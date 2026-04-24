// arch/whisper/whisper.h - Whisper ASR model and context types.
//
// This header is INTERNAL to src/arch/whisper/. It defines the concrete
// classes that derive from transcribe_model / transcribe_context for
// the Whisper ASR family (encoder-decoder transformer with cross-
// attention decoder).
//
// Shape is a direct descendant of src/arch/cohere/cohere.h: the two
// families share the encoder-decoder arch pattern, though their
// encoders are structurally different (cohere is Conformer,
// whisper is a stack of vanilla transformer blocks on top of a
// 2-layer conv stem).

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

namespace transcribe::whisper {

void apply_family_invariants(transcribe_capabilities & caps);

// ---------------------------------------------------------------------------
// KV cache for the autoregressive decoder.
//
// Self-attention KV cache: one entry per decode step, grows until EOS.
// Cross-attention KV cache: computed once from encoder output, reused
// across every decode step.
//
// Layout follows the cohere/whisper.cpp pattern: one flat 1D tensor
// per role; per-layer slices are constructed as views during graph
// build.
// ---------------------------------------------------------------------------

struct WhisperKvCache {
    // Self-attention cache.
    // Flat tensors of size [d_model * n_layer * n_ctx].
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;

    // Cross-attention cache.
    // Flat tensors of size [d_model * n_layer * T_enc].
    ggml_tensor * cross_k = nullptr;
    ggml_tensor * cross_v = nullptr;

    // ggml context that owns the cache tensor metadata.
    ggml_context * ctx = nullptr;

    // Backend buffer backing all cache tensors.
    ggml_backend_buffer_t buffer = nullptr;

    // Maximum self-attention sequence length.
    int n_ctx = 0;

    // Current number of filled positions in the self-attention cache.
    int n = 0;

    // Write head for the next self-attention step.
    int head = 0;

    // Number of encoder frames in the cross-attention cache.
    int T_enc = 0;

    // Whether cross-attention cache has been populated this run.
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

// Allocate cache tensors. n_ctx caps self-attention length; T_enc is
// fixed at 1500 for whisper (max_source_positions after the stride-2
// conv). d_model is the per-layer hidden dim. kv_type is the storage
// dtype for all four cache tensors.
bool kv_cache_init(WhisperKvCache & cache,
                   ggml_backend_t   backend,
                   int              n_ctx,
                   int              T_enc,
                   int              d_model,
                   int              n_layer,
                   ggml_type        kv_type);

// ---------------------------------------------------------------------------
// Model / context
// ---------------------------------------------------------------------------

struct WhisperModel final : public transcribe_model {
    Tokenizer       tok;
    WhisperHParams  hparams;
    WhisperWeights  weights;
    ggml_context *  ctx_meta = nullptr;

    // Runtime backend plan. See transcribe-backend.h.
    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Language token ids keyed by BCP-47 short code read from
    // general.languages. Populated at load time. Used by the decoder
    // to resolve a params.language string to the correct
    // <|lang_xx|> token id (whisper packs these in tokenizer slots
    // 50259 + lang_index).
    std::vector<std::string> lang_codes;   // owned copy; lifetime matches the model
    std::vector<int32_t>     lang_token_ids;

    // C++ mel frontend (per_utterance / hann_periodic / reflect /
    // Slaney filterbank). Constructed at load() time using the
    // filterbank + window baked into the GGUF for parity with the
    // transformers reference. Optional so failures during load can
    // still surface a usable model object for inspection.
    std::optional<transcribe::MelFrontend> mel;

    WhisperModel() = default;
    ~WhisperModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct WhisperContext final : public transcribe_context {
    ggml_context *        compute_ctx = nullptr;
    ggml_backend_sched_t  sched       = nullptr;

    // Encoder output held on host between encoder compute and the
    // first decoder cross-attention precompute. We materialize to
    // host so the scheduler doesn't need to keep the encoder graph
    // alive across the reset() that happens before the next graph.
    std::vector<float> enc_host;
    int                enc_T = 0;  // number of encoder frames (1500)

    // Host buffer for mel + positional inputs.
    std::vector<float> mel_buf;

    // KV cache for the autoregressive decoder.
    WhisperKvCache kv_cache;

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    // Flash-attention policy. Encoder runs at seqlen=1500 and always
    // benefits from flash kernels where available; decoder runs at
    // seqlen ≤ 448 but the cross-attention keys are 1500 long. Head
    // dim is d_model/n_heads = 64 for tiny, which is supported by
    // every backend's flash_attn_ext kernel today. Keep both on by
    // default; TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH env vars
    // still apply.
    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    WhisperContext() = default;
    ~WhisperContext() override;
};

} // namespace transcribe::whisper
