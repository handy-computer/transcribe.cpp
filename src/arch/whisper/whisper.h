// arch/whisper/whisper.h - Whisper ASR model and context types. INTERNAL to
// src/arch/whisper/. Defines the concrete transcribe_model /
// transcribe_session subclasses for the Whisper encoder-decoder family.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe-tokenizer.h"
#include "transcribe/whisper.h"
#include "weights.h"

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

void apply_family_invariants(transcribe_model & model);

// Per-stage timing counters. Always-on (timestamps are negligible); printing
// is opt-in via TRANSCRIBE_PERF_DEBUG. Reset at the top of whisper_run.
struct WhisperPerfStage {
    int64_t total_us = 0;
    int     count    = 0;

    void add(int64_t us) {
        total_us += us;
        count += 1;
    }

    void reset() {
        total_us = 0;
        count    = 0;
    }
};

struct WhisperPerf {
    // Encoder per-chunk path.
    WhisperPerfStage enc_build;
    WhisperPerfStage enc_alloc;
    WhisperPerfStage enc_compute;
    WhisperPerfStage enc_tensor_get;

    // Cross-KV precompute (chunk-scoped).
    WhisperPerfStage cross_build;
    WhisperPerfStage cross_alloc;
    WhisperPerfStage cross_compute;

    // Decoder prompt prefill (per tier).
    WhisperPerfStage prompt_build;
    WhisperPerfStage prompt_alloc;
    WhisperPerfStage prompt_compute;
    WhisperPerfStage prompt_tensor_get;
    WhisperPerfStage prompt_cpu;

    // Decoder steady-state single-token step (per generated token).
    WhisperPerfStage step_build;
    WhisperPerfStage step_alloc;
    WhisperPerfStage step_compute;
    WhisperPerfStage step_tensor_get;
    WhisperPerfStage step_cpu;

    // CPU-section sub-counters. Always populated (the timestamps are
    // negligible) but only printed when TRANSCRIBE_PERF_DEBUG
    // includes "cpu" or "all". Splits the prompt_cpu / step_cpu
    // total into the four sub-stages of post-graph processing.
    WhisperPerfStage prompt_cpu_suppress;
    WhisperPerfStage prompt_cpu_timestamp;
    WhisperPerfStage prompt_cpu_sample;
    WhisperPerfStage prompt_cpu_logprob;
    WhisperPerfStage step_cpu_suppress;
    WhisperPerfStage step_cpu_timestamp;
    WhisperPerfStage step_cpu_sample;
    WhisperPerfStage step_cpu_logprob;

    int chunks = 0;

    void reset() {
        enc_build.reset();
        enc_alloc.reset();
        enc_compute.reset();
        enc_tensor_get.reset();
        cross_build.reset();
        cross_alloc.reset();
        cross_compute.reset();
        prompt_build.reset();
        prompt_alloc.reset();
        prompt_compute.reset();
        prompt_tensor_get.reset();
        prompt_cpu.reset();
        step_build.reset();
        step_alloc.reset();
        step_compute.reset();
        step_tensor_get.reset();
        step_cpu.reset();
        prompt_cpu_suppress.reset();
        prompt_cpu_timestamp.reset();
        prompt_cpu_sample.reset();
        prompt_cpu_logprob.reset();
        step_cpu_suppress.reset();
        step_cpu_timestamp.reset();
        step_cpu_sample.reset();
        step_cpu_logprob.reset();
        chunks = 0;
    }
};

struct WhisperModel final : public transcribe_model {
    Tokenizer      tok;
    WhisperHParams hparams;
    WhisperWeights weights;
    ggml_context * ctx_meta = nullptr;

    // Runtime backend plan. See transcribe-backend.h.
    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Language token ids keyed by BCP-47 short code (from general.languages).
    // Resolves a params.language string to the <|lang_xx|> token id (whisper
    // packs these in tokenizer slots 50259 + lang_index).
    std::vector<std::string> lang_codes;  // owned copy; lifetime matches the model
    std::vector<int32_t>     lang_token_ids;

    // C++ mel frontend (per_utterance / hann_periodic / reflect / Slaney).
    // Built from the filterbank + window baked into the GGUF. Optional so a
    // load failure still surfaces a model object for inspection.
    std::optional<transcribe::MelFrontend> mel;

    WhisperModel() = default;
    ~WhisperModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct WhisperSession final : public transcribe_session {
    ggml_context *       compute_ctx      = nullptr;
    // Currently-allocated capacity of compute_ctx (mem_size). Used by
    // ensure_compute_ctx to decide between ggml_reset (cheap reuse)
    // and ggml_free + ggml_init (only when more space is needed).
    size_t               compute_ctx_size = 0;
    ggml_backend_sched_t sched            = nullptr;

    // Persistent backend-resident encoder output (see WhisperEncOut).
    WhisperEncOut enc_out;

    // Host-side mirror of the encoder output, populated only when a path needs
    // an F32 input fed via ggml_backend_tensor_set (language detection on the
    // first chunk, dump-emitting validation runs). Off the per-chunk hot path.
    std::vector<float> enc_host;
    int                enc_T = 0;  // number of encoder frames (1500)

    // Host buffer for mel + positional inputs.
    std::vector<float> mel_buf;

    // KV cache for the autoregressive decoder.
    WhisperKvCache kv_cache;

    // Flash-attention policy. On by default for both; TRANSCRIBE_NO_FLASH /
    // TRANSCRIBE_FORCE_FLASH still apply.
    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    // Per-chunk decoding trace for the most recent run, exposed via
    // transcribe_get_whisper_chunk_count / _get_whisper_chunk_trace. Cleared
    // at the top of each run alongside cc->clear_result().
    std::vector<transcribe_whisper_chunk_trace> chunk_traces;

    // Per-stage timing counters; summary printed when TRANSCRIBE_PERF_DEBUG is set.
    WhisperPerf perf;

    // Reusable scratch for the multinomial T>0 sampler, sized to vocab_size on
    // first use to avoid a per-call double[vocab] allocation in the hot path.
    std::vector<double> sample_scratch;

    WhisperSession() = default;
    ~WhisperSession() override;
};

}  // namespace transcribe::whisper
