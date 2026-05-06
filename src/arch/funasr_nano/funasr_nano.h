// arch/funasr_nano/funasr_nano.h - FunASR-Nano family internal model and
// context types.
//
// INTERNAL to src/arch/funasr_nano/.

#pragma once

#include "frontend.h"
#include "weights.h"

#include "transcribe-backend.h"
#include "transcribe-context.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::funasr_nano {

void apply_family_invariants(transcribe_capabilities & caps);

// ---------------------------------------------------------------------------
// KV cache for the autoregressive LM (audio-llm pattern; self-attention
// only, audio fused into input embeddings via row-override).
// ---------------------------------------------------------------------------

struct KvCache {
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;

    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    int n_ctx = 0;
    int n     = 0;
    int head  = 0;

    void free();
};

bool kv_cache_init(KvCache &       cache,
                   ggml_backend_t  backend,
                   int             n_ctx,
                   int             n_kv_heads,
                   int             head_dim,
                   int             n_layer,
                   ggml_type       kv_type);

// ---------------------------------------------------------------------------
// Resolved chat-template special-token ids (filled at load time).
// ---------------------------------------------------------------------------

struct ChatTokens {
    int32_t im_start       = -1;
    int32_t im_end         = -1;
    int32_t newline        = -1;
    int32_t role_system    = -1;
    int32_t role_user      = -1;
    int32_t role_assistant = -1;
};

// ---------------------------------------------------------------------------
// Model / Context
// ---------------------------------------------------------------------------

struct FunAsrNanoModel final : public transcribe_model {
    Tokenizer            tok;
    FunAsrNanoHParams    hparams;
    FunAsrNanoWeights    weights;
    ggml_context *       ctx_meta   = nullptr;
    ggml_context *       ctx_packed = nullptr;  // packed gate_up tensors

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;
    ggml_backend_buffer_t   packed_buffer  = nullptr;

    // Constructed once at load() time; const-after-construction. Kaldi
    // HTK fbank + LFR; CMVN dropped (fe_apply_cmvn=false).
    std::unique_ptr<KaldiFbankFrontend> frontend;

    // Jinja chat template string (verbatim from GGUF KV).
    std::string chat_template;

    // Resolved chat-template token ids.
    ChatTokens chat_tokens;

    FunAsrNanoModel() = default;
    ~FunAsrNanoModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct FunAsrNanoContext final : public transcribe_context {
    ggml_context *       compute_ctx = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    KvCache kv_cache;

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    // Reusable host scratch.
    std::vector<float> frontend_buf;   // [T_lfr, d_input]
    std::vector<float> pe_buf;         // [T_lfr, d_input]  sinusoidal PE
    std::vector<float> enc_host;       // encoder output [d_model, T_lfr]
    std::vector<float> adaptor_host;   // adaptor output [llm_dim, T_lfr]

    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    FunAsrNanoContext() = default;
    ~FunAsrNanoContext() override;
};

} // namespace transcribe::funasr_nano
