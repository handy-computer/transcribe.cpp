// arch/moonshine_streaming/moonshine_streaming.h - Moonshine-Streaming
// model and context types.
//
// This header is INTERNAL to src/arch/moonshine_streaming/. It mirrors
// src/arch/moonshine/moonshine.h since the encoder-decoder + cross-attn
// KV cache shape is shared across both. Differences:
//
//   - `enc_host_for_adapter` holds the post-adapter encoder hidden
//     (instead of the raw encoder output) so the adapter add+proj is
//     applied exactly once per session, mirroring the reference dumper's
//     `apply_adapter()` helper. Cross-KV precompute reads from this
//     buffer, not the encoder output directly.
//   - No streaming session yet: Stage 4 ships one-shot transcribe only.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-context.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
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

namespace transcribe::moonshine_streaming {

void apply_family_invariants(transcribe_capabilities & caps);

// ---------------------------------------------------------------------------
// KV cache for the autoregressive decoder.  Same shape as moonshine /
// cohere / whisper — dual cache: self over [d_model, n_ctx], cross over
// [d_model, T_enc].
// ---------------------------------------------------------------------------

struct MoonshineStreamingKvCache {
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
        T_enc = 0;
        cross_populated = false;
    }
};

bool kv_cache_init(MoonshineStreamingKvCache & cache,
                   ggml_backend_t              backend,
                   int                         n_ctx,
                   int                         T_enc,
                   int                         d_model,
                   int                         n_layer,
                   ggml_type                   kv_type);

// ---------------------------------------------------------------------------
// Model / context
// ---------------------------------------------------------------------------

struct MoonshineStreamingModel final : public transcribe_model {
    Tokenizer                   tok;
    MoonshineStreamingHParams   hparams;
    MoonshineStreamingWeights   weights;
    ggml_context *              ctx_meta = nullptr;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    MoonshineStreamingModel() = default;
    ~MoonshineStreamingModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct MoonshineStreamingContext final : public transcribe_context {
    ggml_context *        compute_ctx = nullptr;
    ggml_backend_sched_t  sched       = nullptr;

    // Host-side mirror of the post-adapter encoder hidden. The adapter
    // pos_emb add (and proj when present) is applied once per session;
    // this host buffer feeds the cross_kv precompute graph.
    std::vector<float> adapter_host;
    int                enc_T = 0;   // T_enc

    MoonshineStreamingKvCache kv_cache;

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    MoonshineStreamingContext() = default;
    ~MoonshineStreamingContext() override;
};

} // namespace transcribe::moonshine_streaming
