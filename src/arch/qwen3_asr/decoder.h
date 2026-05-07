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
#include "qwen3_lm/qwen3_lm.h"
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
PrefillBuild build_prefill_graph(ggml_context *                ctx,
                                 const QwenAsrWeights &        weights,
                                 const QwenAsrHParams &        hp,
                                 transcribe::qwen3_lm::KvCache & kv_cache,
                                 int                           T_prompt,
                                 int                           T_enc,
                                 int                           prefix_len,
                                 int                           suffix_len,
                                 bool                          use_flash,
                                 bool                          slice_last);

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
                           transcribe::qwen3_lm::KvCache & kv_cache,
                           int                             max_n_kv,
                           bool                            use_flash);

} // namespace transcribe::qwen3_asr
