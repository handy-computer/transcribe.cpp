// arch/moonshine/moonshine.h - Moonshine ASR model and context types.
//
// This header is INTERNAL to src/arch/moonshine/. It defines the concrete
// classes that derive from transcribe_model / transcribe_context for the
// Moonshine ASR family (encoder-decoder transformer over raw 16 kHz PCM).
//
// Closest in-tree analog: src/arch/whisper/whisper.h. The two families
// share the encoder-decoder + cross-attn KV cache shape; moonshine
// differs in the frontend (raw PCM, no mel) and uses partial RoPE on
// self-attention (whisper uses learned positional embeddings).

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

namespace transcribe::moonshine {

void apply_family_invariants(transcribe_capabilities & caps);

// ---------------------------------------------------------------------------
// KV cache for the autoregressive decoder.
//
// Same dual-cache layout as src/arch/cohere/ and src/arch/whisper/:
//   self_k / self_v   - one [d_model, n_ctx] slab per layer; grows with
//                       each decode step.
//   cross_k / cross_v - one [d_model, T_enc] slab per layer; precomputed
//                       once per session from the encoder output.
// ---------------------------------------------------------------------------

struct MoonshineKvCache {
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;
    ggml_tensor * cross_k = nullptr;
    ggml_tensor * cross_v = nullptr;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    int n_ctx = 0;     // max self-attn sequence length
    int n     = 0;     // current self-attn fill
    int head  = 0;     // write head for next step
    int T_enc = 0;     // encoder frame count in cross cache

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

bool kv_cache_init(MoonshineKvCache & cache,
                   ggml_backend_t     backend,
                   int                n_ctx,
                   int                T_enc,
                   int                d_model,
                   int                n_layer,
                   ggml_type          kv_type);

// ---------------------------------------------------------------------------
// Model / context
// ---------------------------------------------------------------------------

struct MoonshineModel final : public transcribe_model {
    Tokenizer         tok;
    MoonshineHParams  hparams;
    MoonshineWeights  weights;
    ggml_context *    ctx_meta = nullptr;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    MoonshineModel() = default;
    ~MoonshineModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct MoonshineContext final : public transcribe_context {
    ggml_context *        compute_ctx = nullptr;
    ggml_backend_sched_t  sched       = nullptr;

    // Host-side mirror of the encoder output. Required because the
    // cross-KV graph runs in a fresh compute_ctx that does not share
    // tensor handles with the encoder graph.
    std::vector<float> enc_host;
    int                enc_T = 0;   // number of encoder frames

    MoonshineKvCache kv_cache;

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    // Flash-attention defaults — finalized per-backend in init_context.
    //
    // Encoder: ON everywhere. The encoder runs as one big graph with
    // nq=T_enc, and both shipped variants benefit from FA there (tiny
    // especially — ~2x encoder speedup on long audio).
    //
    // Decoder: ON everywhere except Metal. Vulkan/RADV measures ~15%
    // faster on decode_ms with FA on (jfk q8_0: tiny 112→98 ms,
    // base 165→143 ms). Metal is the outlier: moonshine-base's
    // head_dim_padded=56 is NOT in Metal's supported FA head-size set
    // ({32,40,48,64,72,80,96,...}), so the FA op spills per-step to CPU
    // and is ~2x SLOWER than the explicit kq->softmax->v·kq path; tiny's
    // head_dim_padded=40 IS supported on Metal but only via the tile
    // kernel (vec needs head_dim%32==0), roughly a wash. Net: keep
    // decoder FA off on Metal until either (a) we re-pad head_dim to
    // 64 to enable the FA vec kernel, or (b) ggml Metal grows support
    // for these head sizes. TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH
    // still apply on top.
    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    MoonshineContext() = default;
    ~MoonshineContext() override;
};

} // namespace transcribe::moonshine
