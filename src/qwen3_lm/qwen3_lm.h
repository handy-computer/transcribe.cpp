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

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace transcribe::qwen3_lm {

// ---------------------------------------------------------------------------
// Block view
// ---------------------------------------------------------------------------

// Nullable-pointer projection over one Qwen3 decoder block's weights.
// Field names match `arch/qwen3_asr/QwenAsrDecBlock` and
// `arch/funasr_nano/DecBlock` so the call-site builder is a struct
// initializer.
//
// All slots are required for block_prefill / block_step.
struct BlockView {
    ggml_tensor * norm_attn_w   = nullptr;  // input_layernorm (RMSNorm, no bias)
    ggml_tensor * norm_ffn_w    = nullptr;  // post_attention_layernorm
    ggml_tensor * attn_q_w      = nullptr;  // [hidden, n_heads * head_dim]
    ggml_tensor * attn_k_w      = nullptr;  // [hidden, n_kv_heads * head_dim]
    ggml_tensor * attn_v_w      = nullptr;  // [hidden, n_kv_heads * head_dim]
    ggml_tensor * attn_o_w      = nullptr;  // [n_heads * head_dim, hidden]
    ggml_tensor * attn_q_norm   = nullptr;  // [head_dim] per-head Q-norm
    ggml_tensor * attn_k_norm   = nullptr;  // [head_dim] per-head K-norm
    // Packed gate+up: [hidden, 2·intermediate]. Filled by pack_gate_up
    // at load time (see below). The graph runs ONE mul_mat against this
    // tensor + ggml_swiglu instead of two mul_mats + manual silu·mul.
    ggml_tensor * ffn_gate_up_w = nullptr;  // [hidden, 2·intermediate]
    ggml_tensor * ffn_down_w    = nullptr;  // [intermediate, hidden]
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

    int n_ctx = 0;
    int n     = 0;     // current fill (high-water mark)
    int head  = 0;     // next write position

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

} // namespace transcribe::qwen3_lm
