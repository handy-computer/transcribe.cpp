// arch/moss/decoder.h - MOSS Qwen3 LM graph builders (prefill + step).
//
// 28-layer Qwen3 causal decoder, identical block math to arch/qwen3_asr via
// src/causal_lm/ (pre-LN RMSNorm, GQA with per-head q/k RMSNorm, NeoX RoPE
// theta 1e6, SwiGLU on packed gate+up, tied lm_head).
//
// Audio injection differs from qwen3_asr: MOSS audio-pad positions are NOT
// contiguous (the processor interleaves time-marker digit tokens into the
// span), so both single and batched prefill use the elementwise blend
//   x = token_emb * keep_mask + audio_dense
// where audio_dense holds the adaptor features scattered host-side into the
// audio-pad prompt positions (0 elsewhere) and keep_mask is 0 there / 1 else.

#pragma once

#include "causal_lm/causal_lm.h"
#include "ggml.h"
#include "weights.h"

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::moss {

struct DecoderDumps {
    ggml_tensor * token_emb       = nullptr;
    ggml_tensor * audio_injected  = nullptr;
    ggml_tensor * block_0_out     = nullptr;
    ggml_tensor * block_last_out  = nullptr;
    ggml_tensor * out_before_head = nullptr;
    ggml_tensor * logits_raw      = nullptr;
};

struct PrefillBuild {
    ggml_tensor * input_ids_in   = nullptr;  // [T_prompt] i32
    ggml_tensor * audio_dense_in = nullptr;  // [hidden, T_prompt] f32
    ggml_tensor * keep_mask_in   = nullptr;  // [1, T_prompt] f32 (0=audio, 1=keep)
    ggml_tensor * positions_in   = nullptr;  // [T_prompt] i32
    ggml_tensor * mask_in        = nullptr;  // [T_prompt, T_prompt] f16 causal
    ggml_tensor * out            = nullptr;  // [vocab] last-position logits
    DecoderDumps  dumps{};
    ggml_cgraph * graph = nullptr;
    int           T_prompt = 0;
};

PrefillBuild build_prefill_graph(ggml_context *                   ctx,
                                 const MossWeights &              weights,
                                 const MossHParams &              hp,
                                 transcribe::causal_lm::KvCache & kv_cache,
                                 int                              T_prompt,
                                 bool                             use_flash,
                                 bool                             slice_last);

struct StepBuild {
    ggml_tensor * input_id_in = nullptr;  // [1] i32
    ggml_tensor * position_in = nullptr;  // [1] i32
    ggml_tensor * kv_idx_in   = nullptr;  // [1] i64
    ggml_tensor * mask_in     = nullptr;  // [max_n_kv, 1] f16
    ggml_tensor * out         = nullptr;  // [1] i32 argmax
    ggml_tensor * logits      = nullptr;  // [vocab] (for gen-step dumps)
    ggml_cgraph * graph       = nullptr;
    int           max_n_kv    = 0;
};

StepBuild build_step_graph(ggml_context *                   ctx,
                           const MossWeights &              weights,
                           const MossHParams &              hp,
                           transcribe::causal_lm::KvCache & kv_cache,
                           int                              max_n_kv,
                           bool                             use_flash);

// ---------- Batched (offline run_batch) ----------

struct PrefillBuildBatched {
    ggml_tensor * input_ids_in   = nullptr;  // [T_prompt_max, B] i32
    ggml_tensor * audio_dense_in = nullptr;  // [hidden, T_prompt_max*B] f32
    ggml_tensor * keep_mask_in   = nullptr;  // [1, T_prompt_max*B] f32
    ggml_tensor * positions_in   = nullptr;  // [T_prompt_max] i32
    ggml_tensor * mask_in        = nullptr;  // [T_prompt_max, T_prompt_max] f16
    ggml_tensor * kv_idx_in       = nullptr;  // [T_prompt_max, B] i64
    ggml_tensor * last_idx_in     = nullptr;  // [1, B] i32
    ggml_tensor * logits          = nullptr;  // [vocab, B]
    ggml_tensor * out             = nullptr;  // [B] i32 argmax
    ggml_cgraph * graph           = nullptr;
    int           T_prompt_max    = 0;
    int           n_batch         = 0;
};

PrefillBuildBatched build_prefill_graph_batched(ggml_context *                   ctx,
                                                const MossWeights &              weights,
                                                const MossHParams &              hp,
                                                transcribe::causal_lm::KvCache & kv_cache,
                                                int                              T_prompt_max,
                                                int                              n_batch,
                                                bool                             use_flash);

struct StepBuildBatched {
    ggml_tensor * input_ids_in = nullptr;  // [B] i32
    ggml_tensor * position_in  = nullptr;  // [B] i32
    ggml_tensor * kv_idx_in    = nullptr;  // [1, B] i64
    ggml_tensor * mask_in      = nullptr;  // [max_n_kv, 1, 1, B] f16
    ggml_tensor * logits       = nullptr;  // [vocab, B]
    ggml_tensor * out          = nullptr;  // [B] i32
    ggml_cgraph * graph        = nullptr;
    int           max_n_kv     = 0;
    int           n_batch      = 0;
};

StepBuildBatched build_step_graph_batched(ggml_context *                   ctx,
                                          const MossWeights &              weights,
                                          const MossHParams &              hp,
                                          transcribe::causal_lm::KvCache & kv_cache,
                                          int                              max_n_kv,
                                          int                              n_batch,
                                          bool                             use_flash);

}  // namespace transcribe::moss
