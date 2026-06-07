// arch/qwen3_asr/decoder.h - Qwen3-ASR LM graph builders (prefill + step).
//
// The LM is a 28-layer Qwen3 causal decoder with:
//   - pre-LN RMSNorm (eps 1e-6, weight * rsqrt(mean(x^2) + eps) * x)
//   - GQA (16 Q heads, 8 KV heads, head_dim=128)
//   - per-head Q/K RMSNorm applied on head_dim AFTER projection but
//     BEFORE RoPE
//   - standard NeoX-style RoPE (rotate_half) at freq_base=1e6
//   - SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
//   - tied lm_head (token_embd.weight transposed)
//
// MRoPE note: the Qwen3-ASR config carries an `mrope_section=[24,20,20]`
// split but for text-only ASR every position has the same (T, H, W)
// coordinate, so the interleaved MRoPE collapses to a plain RoPE on
// the temporal axis. We rely on that reduction and use ggml_rope_ext
// with GGML_ROPE_TYPE_NEOX. If a future variant introduces real
// multi-modal positions we'll switch to ggml_rope_multi.
//
// Audio injection: input_ids contains `audio_token_id` placeholders at
// the positions the encoder's output frames should land. The prefill
// graph embeds input_ids as usual, then `ggml_set_rows` scatters the
// encoder output over the audio positions.

#pragma once

#include "qwen3_asr.h"
#include "causal_lm/causal_lm.h"
#include "weights.h"

#include "ggml.h"

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::qwen3_asr {

struct DecoderDumps {
    ggml_tensor * token_emb       = nullptr;  // embed_tokens output, pre-injection
    ggml_tensor * audio_injected  = nullptr;  // after encoder-row scatter
    ggml_tensor * block_0_out     = nullptr;
    ggml_tensor * block_last_out  = nullptr;
    ggml_tensor * out_before_head = nullptr;  // final RMSNorm output
    ggml_tensor * logits_raw      = nullptr;  // [vocab] for the last position
};

struct PrefillBuild {
    ggml_tensor * input_ids_in    = nullptr;  // [T_prompt] i32 (for dec.token_emb dump)
    ggml_tensor * enc_out_in      = nullptr;  // [enc_output_dim, T_enc] f32
    ggml_tensor * positions_in    = nullptr;  // [T_prompt] i32 for RoPE
    ggml_tensor * mask_in         = nullptr;  // [T_prompt, T_prompt] f32 (causal)
    ggml_tensor * out             = nullptr;  // [vocab_size] — last-position logits
    DecoderDumps  dumps {};
    ggml_cgraph * graph           = nullptr;

    int           T_prompt        = 0;
    int           T_enc           = 0;
    int           prefix_len      = 0;  // # prompt tokens before the audio block
    int           suffix_len      = 0;  // # prompt tokens after the audio block
};

// Build a prefill graph that:
//   1. fetches token embeddings for the full prompt (for dump parity),
//   2. builds the injected-embedding sequence by concatenating
//      [prefix_token_embeddings | encoder_output | suffix_token_embeddings],
//   3. runs the 28 Qwen3 blocks, writing K/V into kv_cache at positions
//      [0, T_prompt),
//   4. applies the final RMSNorm and tied-head projection,
//   5. outputs logits for the last position only.
//
// First port assumption: the audio block is a single contiguous run at
// positions [prefix_len, prefix_len + T_enc). Multi-audio prompts (not
// used by Qwen3-ASR's chat template) would need repeated get_rows +
// concat segments.
//
// The kv_cache is WRITTEN by the graph. Callers should set
// kv_cache.n = T_prompt and kv_cache.head = T_prompt after compute.
// kv_batch_slot / kv_n_batch: for offline batched decode, write this
// utterance's KV into slab `kv_batch_slot` of a batched cache holding
// `kv_n_batch` slabs (see causal_lm::block_prefill / kv_init_batched).
// Defaults (0, 1) reproduce the single-shot KV layout exactly.
PrefillBuild build_prefill_graph(ggml_context *                ctx,
                                 const QwenAsrWeights &        weights,
                                 const QwenAsrHParams &        hp,
                                 transcribe::causal_lm::KvCache & kv_cache,
                                 int                           T_prompt,
                                 int                           T_enc,
                                 int                           prefix_len,
                                 int                           suffix_len,
                                 bool                          use_flash,
                                 bool                          slice_last,
                                 int                           kv_batch_slot = 0,
                                 int                           kv_n_batch    = 1);

// ---------- Step graph (one token) ----------

struct StepBuild {
    ggml_tensor * input_id_in  = nullptr;  // [1] i32
    ggml_tensor * position_in  = nullptr;  // [1] i32, value = n_past
    ggml_tensor * kv_idx_in    = nullptr;  // [1] i64, KV write position
    ggml_tensor * mask_in      = nullptr;  // [max_n_kv, 1] f16
    ggml_tensor * out          = nullptr;  // [1] i32 — argmax token id
    ggml_cgraph * graph        = nullptr;

    int           max_n_kv     = 0;  // static shape sized for whole run
};

// Build a static-shape single-token step graph suitable for reuse
// across every autoregressive step in a run. Topology depends only on
// max_n_kv (the max KV window the LM will see), not on n_past. The
// graph has four input tensors the caller updates per step:
//   input_id_in [1]      — the token to embed
//   position_in [1]      — RoPE position
//   kv_idx_in   [1]      — where to write K/V (via ggml_set_rows)
//   mask_in     [max_n_kv, 1] — attention mask; positions > n_past hold -inf
//
// Reusing the same graph every step eliminates the ggml_free/init +
// rebuild + sched_reset + sched_alloc_graph overhead that per-step
// construction incurs (measured ~0.4 ms/step before this change).
StepBuild build_step_graph(ggml_context *                  ctx,
                           const QwenAsrWeights &          weights,
                           const QwenAsrHParams &          hp,
                           transcribe::causal_lm::KvCache & kv_cache,
                           int                             max_n_kv,
                           bool                            use_flash);

// ---------- Batched prefill graph (B utterances, T tokens each) ----------

struct PrefillBuildBatched {
    ggml_tensor * input_ids_in = nullptr;  // [T_prompt_max, B] i32 (audio_token placeholders)
    // Audio injection via elementwise blend (no set_rows): audio_dense holds the
    // enc_out embeds scattered (host-side) into their prompt positions, zero
    // elsewhere; keep_mask is 0 at audio positions and 1 elsewhere. The block
    // input is x*keep_mask + audio_dense. Elementwise ops cross the CPU/CUDA
    // split (forced by k-quant token_embd get_rows) cleanly, unlike a set_rows.
    ggml_tensor * audio_dense_in = nullptr;  // [dec_hidden, T_prompt_max*B] f32
    ggml_tensor * keep_mask_in   = nullptr;  // [1, T_prompt_max*B] f32 (0=audio,1=keep)
    ggml_tensor * positions_in = nullptr;  // [T_prompt_max] i32 (shared 0..T-1)
    ggml_tensor * mask_in      = nullptr;  // [T_prompt_max, T_prompt_max] f16 causal (shared)
    ggml_tensor * kv_idx_in    = nullptr;  // [T_prompt_max, B] i64 (idx[t,b] = t)
    ggml_tensor * last_idx_in  = nullptr;  // [1, B] i32 (each utterance's last real position)
    ggml_tensor * logits       = nullptr;  // [vocab, B] last-real-position logits
    ggml_tensor * out          = nullptr;  // [B] i32 argmax (first generated token)
    ggml_cgraph * graph        = nullptr;

    int T_prompt_max = 0;
    int T_enc_max    = 0;
    int n_batch      = 0;
};

// Build a batched prefill graph: B prompts of up to T_prompt_max tokens each,
// writing each utterance's K/V into slab b of a batched cache (kv_init_batched).
// Audio is injected by ggml_set_rows scatter of enc_out into the audio
// placeholder positions (per-utterance scatter indices in enc_idx_in). The
// last-real-position logits per utterance are gathered via last_idx_in, so the
// readback is [vocab, B] (not [vocab, T_prompt_max, B]). Requires use_flash.
PrefillBuildBatched build_prefill_graph_batched(
    ggml_context *                  ctx,
    const QwenAsrWeights &          weights,
    const QwenAsrHParams &          hp,
    transcribe::causal_lm::KvCache & kv_cache,
    int                             T_prompt_max,
    int                             T_enc_max,
    int                             n_batch,
    bool                            use_flash);

// ---------- Batched step graph (B utterances, one token each) ----------

struct StepBuildBatched {
    ggml_tensor * input_ids_in = nullptr;  // [B] i32
    ggml_tensor * position_in  = nullptr;  // [B] i32, value = per-row n_past
    ggml_tensor * kv_idx_in    = nullptr;  // [1, B] i64, per-row KV write row
    ggml_tensor * mask_in      = nullptr;  // [max_n_kv, 1, 1, B] f16
    ggml_tensor * logits       = nullptr;  // [vocab, B] f32 (parity/sampling)
    ggml_tensor * out          = nullptr;  // [B] i32 — per-row argmax token id
    ggml_cgraph * graph        = nullptr;

    int           max_n_kv     = 0;
    int           n_batch      = 0;
};

// Build a static-shape batched step graph reused across every step of an
// offline batch of `n_batch` utterances against a batched KV cache
// (causal_lm::kv_init_batched). The four input tensors are updated per step:
//   input_ids_in [B]            — the token to embed for each utterance
//   position_in  [B]            — RoPE position per utterance (= its n_past)
//   kv_idx_in    [1, B]         — KV write row per utterance (= its n_past)
//   mask_in      [max_n_kv,1,1,B] — per-utterance attention mask
// Requires use_flash (block_step_batched has no manual-GQA path).
StepBuildBatched build_step_graph_batched(
    ggml_context *                  ctx,
    const QwenAsrWeights &          weights,
    const QwenAsrHParams &          hp,
    transcribe::causal_lm::KvCache & kv_cache,
    int                             max_n_kv,
    int                             n_batch,
    bool                            use_flash);

} // namespace transcribe::qwen3_asr
