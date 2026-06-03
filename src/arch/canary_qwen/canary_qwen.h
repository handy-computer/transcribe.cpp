// arch/canary_qwen/canary_qwen.h - SALM (FastConformer + Qwen3-1.7B) family.
//
// INTERNAL to src/arch/canary_qwen/.
//
// Composition:
//   pcm
//     -> mel preprocessor (NeMo AudioToMelSpectrogramPreprocessor)
//     -> FastConformer encoder (32 blocks, identical to canary-1b-flash)
//     -> perception projection (nn.Linear(1024, 2048) + bias)
//   prompt = HF chat template applied to
//            "Transcribe the following: <|audioplaceholder|>"
//     -> token-id list (15 ids for the JFK case)
//   audio scatter: replace single <|audioplaceholder|> position with
//                  T_enc audio rows -> input_embeds (T_prompt, hidden=2048)
//     -> Qwen3-1.7B causal LM (28 blocks, GQA 16/8, head_dim 128)
//     -> tied lm_head -> greedy autoregressive loop until EOS or max_new.

#pragma once

#include "weights.h"

#include "qwen3_lm/qwen3_lm.h"
#include "transcribe-backend.h"
#include "transcribe-session.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::canary_qwen {

void apply_family_invariants(transcribe_model & model);

// Resolved chat-template special-token ids (filled at load time from the
// GGUF tokenizer). The HF chat template emits these by name; we look up
// each token id once and reuse.
struct ChatTokens {
    int32_t im_start       = -1;   // "<|im_start|>"
    int32_t im_end         = -1;   // "<|im_end|>"
    int32_t role_user      = -1;   // bpe("user") = single token in Qwen2 BPE
    int32_t role_assistant = -1;   // bpe("assistant") = single token
};

struct CanaryQwenModel final : public transcribe_model {
    Tokenizer            tok;
    CanaryQwenHParams    hparams;
    CanaryQwenWeights    weights;
    ggml_context *       ctx_meta = nullptr;

    transcribe::BackendPlan       plan;
    ggml_backend_buffer_t         backend_buffer = nullptr;

    // BatchNorm fusion in the conformer conv module — same as canary.
    // Each block stores fused (scale, bias) computed at load time;
    // running_mean/running_var are NOT used at inference.
    ggml_context *               bn_fused_ctx    = nullptr;
    ggml_backend_buffer_t        bn_fused_buffer = nullptr;

    // F16/BF16 conv pointwise weights promoted to F32 on CPU backend so
    // ggml's CPU kernels hit the F32 matmul path (avoids unhealthy
    // fp16/bf16 round-tripping inside the inner loop).
    ggml_context *               conv_pw_f32_ctx    = nullptr;
    ggml_backend_buffer_t        conv_pw_f32_buffer = nullptr;

    // BF16 encoder + decoder linear weights promoted to F32 on CPU
    // backend. canary-1b-flash (the source of these encoder weights)
    // shipped F32 GGUF and was validated F32-only; ggml's CPU BF16
    // matmul path was never exercised. Promotion is conservative and
    // matches the reference's F32 inference regime.
    ggml_context *               linear_f32_ctx    = nullptr;
    ggml_backend_buffer_t        linear_f32_buffer = nullptr;

    // Qwen3 packed gate+up MLP weights — same as funasr_nano /
    // qwen3_asr.
    transcribe::qwen3_lm::PackedGateUpHandles packed_gate_up;

    // C++ mel frontend (constructed once at load).
    std::optional<transcribe::MelFrontend> mel;

    // Static prompt segments built once at load:
    //   prefix = [im_start, "user", "\n", bpe("Transcribe the following: ")]
    //   suffix = [im_end, "\n", im_start, "assistant", "\n"]
    // The audio_locator_id appears between them T_enc times at run time.
    std::vector<int32_t> prompt_prefix_ids;
    std::vector<int32_t> prompt_suffix_ids;

    ChatTokens chat_tokens;

    CanaryQwenModel() = default;
    ~CanaryQwenModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct CanaryQwenSession final : public transcribe_session {
    ggml_context *       compute_ctx = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    transcribe::qwen3_lm::KvCache kv_cache;

    // Batched KV cache for offline transcribe_run_batch (n_batch slabs).
    transcribe::qwen3_lm::KvCache kv_cache_batch;
    int                           kv_batch_cap   = 0;
    int                           kv_batch_n_ctx = 0;

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    // Reusable host scratch.
    std::vector<float> mel_buf;       // [num_mels, n_frames]
    std::vector<float> pos_buf;       // [pos_len, d_model] sinusoidal rel-pos
    std::vector<float> pos_div_term;  // [d_model/2] precomputed exp(-2k * ln(10000)/d)
    std::vector<float> enc_host;      // perception output [hidden=2048, T_enc]

    // Reference (NeMo SALM) ran flash off; we follow suit by default for
    // tightest tensor parity. Override with TRANSCRIBE_FLASH_DECODER=1
    // (or =encoder) at runtime.
    bool encoder_use_flash = false;
    bool decoder_use_flash = false;

    CanaryQwenSession() = default;
    ~CanaryQwenSession() override;
};

} // namespace transcribe::canary_qwen
