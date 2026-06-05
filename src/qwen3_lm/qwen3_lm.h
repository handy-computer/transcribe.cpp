// src/qwen3_lm/qwen3_lm.h - shared Qwen3 LM block math, KV cache,
// and packed gate/up MLP packing.
//
// Both Qwen3-ASR (`Qwen3ASRThinkerTextModel`) and Fun-ASR-Nano
// (the bundled `Qwen3-0.6B` LLM) ship identical Qwen3 decoder
// block code: pre-LN RMSNorm (eps ~1e-6), GQA with per-head Q/K
// RMSNorm before RoPE, NeoX rotate_half RoPE @ θ=1e6 (text-only,
// MRoPE collapses to NeoX), SwiGLU MLP via packed gate+up, tied
// lm_head. Both also pack ffn_gate_w + ffn_up_w into a single
// [hidden, 2·intermediate] tensor at load time so the FFN can
// run with one mul_mat instead of two.
//
// This module exposes those primitives via the same view +
// free-function pattern used by `src/sanm/` and `src/conformer/`:
//
//   - BlockView : nullable-pointer projection over one decoder
//                 block's weights. Each family fills it at the
//                 call site from its own `DecBlock` struct.
//   - BlockParams : per-call scalar shape (n_heads, n_kv_heads,
//                 head_dim, rms_eps, rope_theta, max_position).
//   - block_prefill / block_step : the two block-forward
//                 variants used by the prefill (T_seq > 1) and
//                 autoregressive step (T_seq == 1) graphs.
//   - KvCache + kv_init : self-attention-only KV cache sized
//                 [n_kv_heads · head_dim · n_ctx · n_layer] flat
//                 per K and V. Layout matches what both families
//                 emitted before extraction (so the conversion is
//                 a strict refactor — no graph topology change).
//   - pack_gate_up : load-time helper that allocates the
//                 [hidden, 2·intermediate] packed tensor per
//                 block and copies the gate and up bytes in
//                 (compatible with row-wise quants because
//                 concat-along-dim-1 is just bytes-concat for
//                 those types).
//
// What deliberately stays per-family (audio-LLM call sites):
//
//   - Token embedding lookup, audio injection (different shapes:
//     enc_out for qwen3_asr, adaptor_out for funasr_nano), final
//     RMSNorm + tied lm_head, dump-tensor naming for
//     validate.py parity, prefill / step graph allocation, the
//     autoregressive driver loop, and the chat-template /
//     special-token handling. The block-forward helpers below
//     return raw `ggml_tensor*` so each family keeps full control
//     of its dump points and tensor names.

#pragma once

#include "transcribe.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

struct transcribe_session;

namespace transcribe::qwen3_lm {

// ---------------------------------------------------------------------------
// Block view
// ---------------------------------------------------------------------------

// Nullable-pointer projection over one Qwen3 decoder block's weights.
// Field names match `arch/qwen3_asr/QwenAsrDecBlock` and
// `arch/funasr_nano/DecBlock` so the call-site builder is a struct
// initializer.
//
// All slots are required for block_prefill / block_step EXCEPT
// attn_q_norm / attn_k_norm, which are optional (null = skip, for
// Llama-style decoders without per-head Q/K norm).
struct BlockView {
    ggml_tensor * norm_attn_w   = nullptr;  // input_layernorm (RMSNorm, no bias)
    ggml_tensor * norm_ffn_w    = nullptr;  // post_attention_layernorm
    ggml_tensor * attn_q_w      = nullptr;  // [hidden, n_heads * head_dim]
    ggml_tensor * attn_k_w      = nullptr;  // [hidden, n_kv_heads * head_dim]
    ggml_tensor * attn_v_w      = nullptr;  // [hidden, n_kv_heads * head_dim]
    ggml_tensor * attn_o_w      = nullptr;  // [n_heads * head_dim, hidden]
    // Per-head Q/K RMSNorm (Qwen3). OPTIONAL: leave null for Llama-style
    // decoders (e.g. Voxtral's Ministral backbone) that ship no q/k norm;
    // the block helpers skip the norm when the slot is null.
    ggml_tensor * attn_q_norm   = nullptr;  // [head_dim] per-head Q-norm, or null
    ggml_tensor * attn_k_norm   = nullptr;  // [head_dim] per-head K-norm, or null
    // Packed gate+up: [hidden, 2·intermediate]. Filled by pack_gate_up
    // at load time (see below). The graph runs ONE mul_mat against this
    // tensor + ggml_swiglu instead of two mul_mats + manual silu·mul.
    ggml_tensor * ffn_gate_up_w = nullptr;  // [hidden, 2·intermediate]
    ggml_tensor * ffn_down_w    = nullptr;  // [intermediate, hidden]
    // OPTIONAL per-layer FFN-branch scale applied right after norm_ffn and
    // before the gate_up mul_mat: ff_norm *= ffn_scale (broadcast [hidden,1]
    // over the token axis). Null for every standard caller; used by
    // voxtral_realtime's delay-conditioned adaptive-norm (a per-layer constant
    // `1 + ada(t_cond)` precomputed once per run). See arch/voxtral_realtime.
    ggml_tensor * ffn_scale     = nullptr;  // [hidden] or [hidden,1], or null
};

// Per-call scalar block params.
struct BlockParams {
    int   n_heads      = 0;
    int   n_kv_heads   = 0;
    int   head_dim     = 0;
    int   max_position = 0;     // RoPE max position (passed to ggml_rope_ext)
    float rms_eps      = 0.0f;
    float rope_theta   = 0.0f;
};

// ---------------------------------------------------------------------------
// KV cache (self-attention only)
// ---------------------------------------------------------------------------

// Flat 1D K and V tensors sized [n_kv_heads · head_dim · n_ctx · n_layer].
// Layout (slowest → fastest): layer, position, head, dim. Matches the
// memory the per-family decoder.cpp wrote before extraction.
struct KvCache {
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;

    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    int n_ctx    = 0;
    int n        = 0;     // current fill (high-water mark)
    int head     = 0;     // next write position
    int n_batch  = 1;     // utterances packed along the batch axis (offline
                          // batched decode). 1 == single-shot layout, in which
                          // case the flat tensor is byte-identical to kv_init.

    void free();
};

// Allocate K/V tensors on `backend` and bind them to a fresh context.
// Returns false on failure (only F16 / F32 KV types supported).
bool kv_init(KvCache &      cache,
             ggml_backend_t backend,
             int            n_ctx,
             int            n_kv_heads,
             int            head_dim,
             int            n_layer,
             ggml_type      kv_type);

// Batched variant: allocates [n_kv_heads · head_dim · n_ctx · n_batch · n_layer]
// flat per K and V, laid out (slowest → fastest) layer, batch, position, head,
// dim. With n_batch == 1 the size and layout are identical to kv_init, so the
// single-shot block_step views read byte-identical memory. Sets cache.n_batch.
bool kv_init_batched(KvCache &      cache,
                     ggml_backend_t backend,
                     int            n_ctx,
                     int            n_kv_heads,
                     int            head_dim,
                     int            n_layer,
                     int            n_batch,
                     ggml_type      kv_type);

// ---------------------------------------------------------------------------
// Block forward (prefill: T_seq > 1)
// ---------------------------------------------------------------------------

struct BlockOpts {
    bool use_flash             = true;

    // When true, after the post-attention residual add but before the
    // FFN sub-layer, slice x to the last position only. Output shape
    // becomes [hidden, 1] instead of [hidden, T_seq]. The caller uses
    // this on the LAST block when only the final position's logits are
    // consumed (matches llama.cpp's inp_out_ids optimization).
    bool slice_last_before_ffn = false;

    // Offline batched decode: write/read this utterance's KV in slab
    // `kv_batch_slot` of a batched cache holding `kv_n_batch` slabs. The
    // per-(layer,slot) byte offset becomes
    //   (kv_batch_slot + kv_n_batch * layer_idx) * n_ctx * kv_dim.
    // Defaults (0, 1) reproduce the single-shot offset layer_idx*n_ctx*kv_dim
    // exactly, so single-shot callers are unaffected.
    int  kv_batch_slot = 0;
    int  kv_n_batch    = 1;
};

// Run one Qwen3 block on `x` (ne = [hidden, T_seq]). Writes K/V for
// positions [0, T_seq) at layer `layer_idx` of `kv_cache`. Mask is
// [T_seq, T_seq] f16 (causal, host-prepared); positions are [T_seq] i32.
//
// Returns the post-block hidden state. Shape is [hidden, T_seq] unless
// `opts.slice_last_before_ffn`, in which case [hidden, 1].
//
// The function does NOT name any tensors with ggml_set_name and does
// NOT call ggml_set_output / mark_tensor_for_dump. The family's prefill
// builder owns naming and dump-point selection.
ggml_tensor * block_prefill(
    ggml_context *      ctx,
    ggml_cgraph *       gf,
    ggml_tensor *       x,
    const BlockView &   view,
    const BlockParams & params,
    KvCache &           kv_cache,
    int                 layer_idx,
    int                 T_seq,
    ggml_tensor *       mask,       // [T_seq, T_seq] f16
    ggml_tensor *       positions,  // [T_seq] i32
    BlockOpts           opts = {});

// ---------------------------------------------------------------------------
// Block forward (step: T_seq == 1)
// ---------------------------------------------------------------------------

// Run one Qwen3 block on a single new token. Writes K/V for the row
// indexed by `kv_idx` (i64 [1]) into layer `layer_idx`. Reads the full
// [0, max_n_kv) KV window for attention; `mask` is [max_n_kv, 1] f16
// with zeros at positions ≤ n_past and -inf beyond (host-prepared).
//
// `position` is [1] i32 (RoPE), `kv_idx` is [1] i64 (set_rows index).
// Both are equal at runtime but kept distinct because RoPE wants i32
// and set_rows wants i64.
ggml_tensor * block_step(
    ggml_context *      ctx,
    ggml_cgraph *       gf,
    ggml_tensor *       x,          // [hidden, 1]
    const BlockView &   view,
    const BlockParams & params,
    KvCache &           kv_cache,
    int                 layer_idx,
    int                 max_n_kv,
    ggml_tensor *       mask,       // [max_n_kv, 1] f16
    ggml_tensor *       position,   // [1] i32
    ggml_tensor *       kv_idx,     // [1] i64
    bool                use_flash);

// ---------------------------------------------------------------------------
// Block forward (batched step: B utterances, one new token each)
// ---------------------------------------------------------------------------

// Run one Qwen3 block on B new tokens (x = [hidden, B]), one per utterance
// stepping in lockstep against a batched KV cache (kv_init_batched, n_batch=B).
//
// Layout choice: the batch axis sits on ne[2] (the "position/token" axis)
// through projection + RoPE so each utterance gets its OWN RoPE position from
// `position` [B]; Q is then permuted to [head_dim, 1, n_heads, B] for
// flash_attn_ext, which batches on ne[3]. K/V are written with a single
// batched ggml_set_rows (dest [kv_dim, n_ctx, B], src [kv_dim, 1, B], idx
// [1, B]) so utterance b writes row kv_idx[b] of its own slab.
//
// Inputs:
//   x         [hidden, B]
//   max_n_kv  static KV window (full window read under per-row mask)
//   n_batch   B
//   mask      [max_n_kv, 1, 1, B] f16 — per-utterance causal+pad mask
//   position  [B] i32 — RoPE position per utterance (= that row's n_past)
//   kv_idx    [1, B] i64 — KV write row per utterance (= that row's n_past)
//
// Requires use_flash (the manual GQA path is single-shot only). Returns
// [hidden, B]. With B == 1 this is numerically the batched-of-one analog of
// block_step; the parity gate compares batched row b vs single-shot b.
ggml_tensor * block_step_batched(
    ggml_context *      ctx,
    ggml_cgraph *       gf,
    ggml_tensor *       x,          // [hidden, B]
    const BlockView &   view,
    const BlockParams & params,
    KvCache &           kv_cache,
    int                 layer_idx,
    int                 max_n_kv,
    int                 n_batch,
    ggml_tensor *       mask,       // [max_n_kv, 1, 1, B] f16
    ggml_tensor *       position,   // [B] i32
    ggml_tensor *       kv_idx,     // [1, B] i64
    bool                use_flash);

// ---------------------------------------------------------------------------
// Block forward (batched prefill: B utterances, T tokens each)
// ---------------------------------------------------------------------------

// Run one Qwen3 block over B prompts of T tokens each (x = [hidden, T, B]),
// writing each utterance's K/V to positions [0, T) of its own slab in a
// batched KV cache (kv_init_batched, n_batch=B). Batch rides ne[2]; RoPE
// positions [T] are shared (every prompt starts at position 0); the causal
// mask [T, T] is shared (a real token p < T_prompt[b] only attends [0, p],
// all real). One batched ggml_set_rows writes B*T KV rows (idx[t,b]=t).
//
// Pad tokens (t >= T_prompt[b]) compute harmlessly: their KV lands in pad
// slab positions that the step loop overwrites before attending, and the
// caller gathers only each utterance's real last-position output. Requires
// use_flash (the manual GQA path has no batch axis). Returns [hidden, T, B].
ggml_tensor * block_prefill_batched(
    ggml_context *      ctx,
    ggml_cgraph *       gf,
    ggml_tensor *       x,          // [hidden, T, B]
    const BlockView &   view,
    const BlockParams & params,
    KvCache &           kv_cache,
    int                 layer_idx,
    int                 T_seq,
    int                 n_batch,
    ggml_tensor *       mask,       // [T_seq, T_seq] f16 causal (shared)
    ggml_tensor *       positions,  // [T_seq] i32 (shared)
    ggml_tensor *       kv_idx,     // [T_seq, B] i64 (idx[t,b] = t)
    bool                use_flash);

// ---------------------------------------------------------------------------
// Load-time gate/up packing
// ---------------------------------------------------------------------------

// One block's input pointers (gate_w, up_w) and output slot
// (gate_up_w_out, written by pack_gate_up). gate_w must already be
// allocated and bound to a backend buffer; up_w likewise. The output
// pointer is written with the address of the newly-allocated packed
// tensor inside handles.ctx.
struct GateUpEntry {
    ggml_tensor *  gate_w;
    ggml_tensor *  up_w;
    ggml_tensor ** gate_up_w_out;
};

// Owns the ctx + backend buffer that the packed tensors live in.
// Caller (the model struct) keeps the handles alive for the lifetime
// of the model and calls `free()` on shutdown — including from the
// partial-failure path of `pack_gate_up`, which leaves the handles in
// a state safe to free.
struct PackedGateUpHandles {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    void free();
};

// Allocate a separate packed context + a [hidden, 2·intermediate]
// tensor per block on `backend`, then copy gate's bytes followed by
// up's bytes into each packed tensor. Compatible with row-wise quants
// (Q4/Q5/Q6/Q8) because concat-along-dim-1 is byte-concat for those
// types.
//
// Writes `*entries[i].gate_up_w_out` to point at the new packed tensor.
// Marks the buffer with GGML_BACKEND_BUFFER_USAGE_WEIGHTS. The new
// context lives in `out_handles.ctx` and the backing buffer in
// `out_handles.buffer`.
//
// Returns false on alloc / size-mismatch failure; in that case
// `out_handles` is left in a state safe to free (either fully clean
// or holding a partial ctx that the caller frees on shutdown).
bool pack_gate_up(ggml_backend_t                  backend,
                  int                             hidden,
                  int                             intermediate,
                  const std::vector<GateUpEntry> & entries,
                  PackedGateUpHandles &           out_handles,
                  const char *                    error_tag = "qwen3_lm");

// ---------------------------------------------------------------------------
// Batched greedy step loop (offline transcribe_run_batch decode)
// ---------------------------------------------------------------------------

// The B utterances' batched step graph, as built by a family's
// build_step_graph_batched(). The field NAMES differ across families' build
// structs, so the caller fills this projection at the call site.
struct StepBatchedIO {
    ggml_tensor * input_ids = nullptr;  // [B] i32   — token fed each step
    ggml_tensor * positions = nullptr;  // [B] i32   — RoPE position per row
    ggml_tensor * kv_idx    = nullptr;  // [1, B] i64 — KV write row per utterance
    ggml_tensor * mask      = nullptr;  // [max_n_kv, 1, 1, B] f16 — per-row window
    ggml_tensor * argmax    = nullptr;  // [B] i32   — graph output (next token)
    ggml_cgraph * graph     = nullptr;
};

// Per-row state on entry to the step loop, i.e. just after serial prefill.
//   valid[b]    : the utterance produced a usable prefill
//   next_tok[b] : the prefill's first generated token (fed into step 0)
//   n_past[b]   : == T_prompt[b] (the prompt KV length already written)
// The caller must also have seeded generated[b] with next_tok[b].
struct StepBatchedState {
    std::vector<char>    valid;
    std::vector<int32_t> next_tok;
    std::vector<int>     n_past;
};

struct StepLoopStats {
    int     n_steps = 0;
    int64_t step_us = 0;
};

// Run the lockstep batched greedy decode. Each row steps until it emits
// `eos_id`, accumulates `max_new` generated tokens, or fills the KV window;
// each emitted token is appended to generated[b]. Finished / invalid rows keep
// stepping into their own KV slab (independent memory, so a no-op for live
// rows). Polls session->poll_abort() once per step. The step graph must already
// be built and allocated on `sched`. Returns TRANSCRIBE_ERR_ABORTED on abort,
// TRANSCRIBE_ERR_GGUF on a compute failure, else TRANSCRIBE_OK.
//
// Extracted verbatim from the four qwen3_lm-family run_batch() drivers
// (qwen3_asr / funasr_nano / canary_qwen / granite), which were byte-identical
// here apart from granite hand-coding the f16 mask literals (0x0000 / 0xFC00 ==
// ggml_fp32_to_fp16(0) / ggml_fp32_to_fp16(-inf), so this is bit-identical).
transcribe_status run_batched_step_loop(
    transcribe_session *                session,
    ggml_backend_sched_t                sched,
    const StepBatchedIO &               io,
    int                                 n_batch,
    int                                 max_n_kv,
    int32_t                             eos_id,
    int                                 max_new,
    const StepBatchedState &            state,
    std::vector<std::vector<int32_t>> & generated,
    StepLoopStats *                     stats = nullptr);

} // namespace transcribe::qwen3_lm
