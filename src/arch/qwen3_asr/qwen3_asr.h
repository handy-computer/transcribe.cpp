// arch/qwen3_asr/qwen3_asr.h - Qwen3-ASR model and context types.
//
// INTERNAL to src/arch/qwen3_asr/. Defines the concrete classes that
// derive from transcribe_model / transcribe_context for the Qwen3-ASR
// family (audio-LLM: audio encoder + Qwen3 causal LM with audio-token
// injection).
//
// First port targets non-streaming ASR transcription. The sibling
// forced-aligner variant (Qwen3-ForcedAligner-0.6B) and the streaming
// stream_transcribe path are tracked separately.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-context.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::qwen3_asr {

void apply_family_invariants(transcribe_capabilities & caps);

// ---------------------------------------------------------------------------
// KV cache for the autoregressive LM.
//
// Self-attention only (no cross-attention — audio features are fused
// into the LM's input embedding stream via token injection). Flat 1D
// tensors for K and V, with per-layer views used during graph
// construction.
//
// Shape notes:
//   K: [n_kv_heads * head_dim * n_ctx * n_layer]
//   V: [n_kv_heads * head_dim * n_ctx * n_layer]
//
// GQA: n_kv_heads may be less than n_heads (Qwen3-ASR-0.6B: 16/8).
// ---------------------------------------------------------------------------

struct QwenAsrKvCache {
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;

    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    int n_ctx = 0;
    int n     = 0;
    int head  = 0;

    void free() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        self_k = nullptr;
        self_v = nullptr;
        n    = 0;
        head = 0;
    }
};

bool kv_cache_init(QwenAsrKvCache & cache,
                   ggml_backend_t   backend,
                   int              n_ctx,
                   int              n_kv_heads,
                   int              head_dim,
                   int              n_layer,
                   ggml_type        kv_type);

// ---------------------------------------------------------------------------
// Model / Context
// ---------------------------------------------------------------------------

// Qwen3 chat-template special-token ids resolved through the loaded
// tokenizer at load time. The prompt renderer is narrow (no Jinja)
// but it still tokenizes to exact ids; resolving at load lets a
// future checkpoint that reorders the vocab fail loudly instead of
// silently corrupting the prompt.
//
// All fields are the *resolved* ids; the reference ids on the 0.6B
// and 1.7B vocab are stable (im_start=151644, im_end=151645,
// newline=198, system=8948, user=872, assistant=77091) but we do
// not hardcode them at runtime.
struct ChatTokens {
    int32_t im_start       = -1;
    int32_t im_end         = -1;
    int32_t newline        = -1;
    int32_t role_system    = -1;
    int32_t role_user      = -1;
    int32_t role_assistant = -1;
};

struct QwenAsrModel final : public transcribe_model {
    Tokenizer       tok;
    QwenAsrHParams  hparams;
    QwenAsrWeights  weights;
    ggml_context *  ctx_meta   = nullptr;
    ggml_context *  ctx_packed = nullptr;  // packed gate_up / qkv tensors

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;
    ggml_backend_buffer_t   packed_buffer  = nullptr;

    std::optional<transcribe::MelFrontend> mel;

    // Jinja chat template. The prompt is rebuilt at run time with the
    // caller's language / context; we only store the template string
    // from the GGUF KV. Empty string means "no template" (the first
    // port will refuse to run without one).
    std::string chat_template;

    // Resolved chat-template token ids (filled at load time, see
    // resolve_chat_tokens in model.cpp). Used by build_prompt_tokens.
    ChatTokens chat_tokens;

    QwenAsrModel() = default;
    ~QwenAsrModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct QwenAsrContext final : public transcribe_context {
    ggml_context *       compute_ctx = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    QwenAsrKvCache kv_cache;

    std::vector<float> mel_buf;
    std::vector<float> enc_host;  // audio encoder output, pre-injection

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    QwenAsrContext() = default;
    ~QwenAsrContext() override;
};

} // namespace transcribe::qwen3_asr
