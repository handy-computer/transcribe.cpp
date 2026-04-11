// arch/cohere/cohere.h - Cohere ASR model and context types.
//
// This header is INTERNAL to src/arch/cohere/. It defines the concrete
// classes that derive from transcribe_model / transcribe_context for
// the Cohere ASR family (encoder-decoder conformer + transformer).

#pragma once

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

namespace transcribe::cohere {

void apply_family_invariants(transcribe_capabilities & caps);

// ---------------------------------------------------------------------------
// KV cache for the autoregressive decoder.
//
// Follows the whisper.cpp pattern: flat 1D tensors for K and V, with
// views used during graph construction to read/write per-layer slices.
//
// Self-attention cache: grows with each decode step.
// Cross-attention cache: computed once from encoder output, then reused.
// ---------------------------------------------------------------------------

struct CohereKvCache {
    // Self-attention KV cache.
    // Flat tensors of size [n_state * n_layer * n_ctx].
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;

    // Cross-attention KV cache.
    // Flat tensors of size [n_state * n_layer * T_enc].
    ggml_tensor * cross_k = nullptr;
    ggml_tensor * cross_v = nullptr;

    // ggml context that owns the cache tensor metadata.
    ggml_context * ctx = nullptr;

    // Backend buffer backing all cache tensors.
    ggml_backend_buffer_t buffer = nullptr;

    // Maximum sequence length for self-attention cache.
    int n_ctx = 0;

    // Current number of filled positions in self-attention cache.
    int n = 0;

    // The position at which the next token(s) will be written.
    int head = 0;

    // Number of encoder frames in cross-attention cache.
    int T_enc = 0;

    // Whether cross-attention cache has been populated.
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

// Initialize the KV cache. Must be called after backends are set up.
// n_ctx: max self-attention sequence length.
// T_enc: number of encoder frames (for cross-attention cache).
// n_state: decoder hidden dim.
// n_layer: number of decoder layers.
// kv_type: storage dtype for all four cache tensors (F16 or F32).
bool kv_cache_init(CohereKvCache & cache,
                   ggml_backend_t  backend,
                   int             n_ctx,
                   int             T_enc,
                   int             n_state,
                   int             n_layer,
                   ggml_type       kv_type);

struct CohereModel final : public transcribe_model {
    Tokenizer       tok;
    CohereHParams   hparams;
    CohereWeights   weights;
    ggml_context *  ctx_meta = nullptr;

    std::vector<ggml_backend_t> backends;
    ggml_backend_buffer_t       backend_buffer = nullptr;

    // Fused BN parameters (same as Parakeet).
    ggml_context *          bn_fused_ctx    = nullptr;
    ggml_backend_buffer_t   bn_fused_buffer = nullptr;

    // On CPU primary backend, the conformer 1×1 pointwise conv weights
    // are dequantized from their on-disk F16 form to F32 at load time
    // — Zen 2 class CPUs pay an F16→F32 upconvert on every matmul that
    // erases the bandwidth win. Tensors live in this ctx, and the
    // CohereBlock slots point here instead of the main weight buffer.
    ggml_context *          conv_pw_f32_ctx    = nullptr;
    ggml_backend_buffer_t   conv_pw_f32_buffer = nullptr;

    std::optional<transcribe::MelFrontend> mel;

    CohereModel() = default;
    ~CohereModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct CohereContext final : public transcribe_context {
    ggml_context *        compute_ctx    = nullptr;
    ggml_backend_sched_t  sched          = nullptr;
    ggml_tensor *         encoder_out    = nullptr;

    // KV cache for the decoder.
    CohereKvCache kv_cache;

    std::vector<float> mel_buf;
    std::vector<float> pos_buf;
    std::vector<float> pos_div_term;
    std::vector<float> enc_host;

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    // Flash-attention is controlled per-stage because the encoder and
    // decoder have different head dimensions and therefore different
    // backend support profiles:
    //
    //   - encoder head_dim = 160 -> upstream ggml's Metal backend has
    //     no flash_attn_ext kernel for this dk, so we default it OFF
    //     on Metal. Manual mul_mat + softmax + mul_mat ties or beats
    //     flash at encoder sequence lengths anyway.
    //   - decoder head_dim = 128 -> works with flash_attn_ext on
    //     every backend we ship today, so we default it ON.
    //
    // The TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH env vars apply
    // to both stages at once (the user's intent is "no flash kernels
    // anywhere" or "flash kernels everywhere"). Backend-specific auto
    // disable is per-stage.
    bool               encoder_use_flash = true;
    bool               decoder_use_flash = true;

    CohereContext() = default;
    ~CohereContext() override;
};

} // namespace transcribe::cohere
