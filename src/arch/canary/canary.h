// arch/canary/canary.h - Canary multitask AED model and context types.
//
// This header is INTERNAL to src/arch/canary/. It defines the concrete
// classes that derive from transcribe_model / transcribe_context for
// the NVIDIA Canary family (FastConformer encoder + Transformer decoder
// with 4-slot or 5-slot multitask prompt).
//
// Modeled on src/arch/cohere/cohere.h. Differences from cohere:
//   - encoder linears are bias-FREE (use_bias=False); shape mirrors
//     parakeet rather than cohere
//   - decoder is structurally identical to cohere's but tensor names
//     differ (dec.layer.{i}.norm{1,2,3} / {self_attn,cross_attn,ffn}.{q,k,v,o,up,down})
//   - LM head is UNTIED (explicit dec.head.{weight,bias})
//   - 180m-flash variant has an encoder->decoder projection
//     (enc_d_model=512 -> dec_d_model=1024); other variants share
//     d_model and skip the projection

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

namespace transcribe::canary {

void apply_family_invariants(transcribe_capabilities & caps);

// ---------------------------------------------------------------------------
// KV cache for the autoregressive decoder. Same shape as cohere.
// ---------------------------------------------------------------------------

struct CanaryKvCache {
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;
    ggml_tensor * cross_k = nullptr;
    ggml_tensor * cross_v = nullptr;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    int n_ctx = 0;
    int n     = 0;
    int head  = 0;
    int T_enc = 0;

    bool cross_populated = false;

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
        cross_k = nullptr;
        cross_v = nullptr;
        n = 0;
        head = 0;
        cross_populated = false;
    }
};

bool kv_cache_init(CanaryKvCache & cache,
                   ggml_backend_t  backend,
                   int             n_ctx,
                   int             T_enc,
                   int             n_state,
                   int             n_layer,
                   ggml_type       kv_type);

struct CanaryModel final : public transcribe_model {
    Tokenizer       tok;
    CanaryHParams   hparams;
    CanaryWeights   weights;
    ggml_context *  ctx_meta = nullptr;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Fused BN parameters (same as parakeet/cohere).
    ggml_context *          bn_fused_ctx    = nullptr;
    ggml_backend_buffer_t   bn_fused_buffer = nullptr;

    // CPU-only F16 -> F32 promotion buffer for conformer 1x1 pointwise convs.
    ggml_context *          conv_pw_f32_ctx    = nullptr;
    ggml_backend_buffer_t   conv_pw_f32_buffer = nullptr;

    std::optional<transcribe::MelFrontend> mel;

    CanaryModel() = default;
    ~CanaryModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct CanaryContext final : public transcribe_context {
    ggml_context *        compute_ctx = nullptr;
    ggml_backend_sched_t  sched       = nullptr;
    ggml_tensor *         encoder_out = nullptr;

    CanaryKvCache kv_cache;

    std::vector<float> mel_buf;
    std::vector<float> pos_buf;
    std::vector<float> pos_div_term;
    std::vector<float> enc_host;

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    // Per-stage flash-attention controls. Same rationale as cohere:
    // encoder rel-pos MHSA dk depends on enc_d_model/n_heads (96 for
    // 180m-flash, 128 for the larger variants); decoder dk=128
    // uniformly. ggml's Metal flash kernel currently lacks a path for
    // some encoder dk values, so we default encoder flash off on
    // Metal. The decoder defaults flash on.
    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    CanaryContext() = default;
    ~CanaryContext() override;
};

} // namespace transcribe::canary
