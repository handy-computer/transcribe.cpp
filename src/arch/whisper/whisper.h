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
#include "transcribe/whisper.h"
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

void apply_family_invariants(transcribe_model & model);

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

    // Padded encoder length (= GGML_PAD(T_enc, 256)). The cross K/V
    // tensors are allocated to T_enc_pad rows per layer so the FA op
    // sees a sequence dim that is a multiple of the Metal kernel's
    // block size; the trailing T_enc_pad - T_enc rows are left zero
    // by buffer_clear and never written by build_cross_kv_graph.
    // Layer offsets in the flat cache must use T_enc_pad (not T_enc)
    // so adjacent layers do not overlap.
    int T_enc_pad = 0;

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
        T_enc = 0;
        T_enc_pad = 0;
        cross_populated = false;
    }
};

// ---------------------------------------------------------------------------
// Persistent encoder output tensor.
//
// Backend-resident F32 tensor sized [d_model, T_enc] that holds the
// encoder output across the encoder→cross-KV transition. Avoids the
// GPU→CPU→GPU roundtrip the previous design paid every chunk: the
// encoder graph appends a ggml_cpy into this tensor as its final op,
// and build_cross_kv_graph reads from it via a view.
//
// Lifetime: allocated on first use, sized from hparams.enc_d_model
// and the per-chunk T_enc the encoder emits. Reallocated only when
// shape changes (which does not happen for stock whisper variants —
// T_enc is fixed at hparams.enc_max_source_positions = 1500).
// ---------------------------------------------------------------------------

struct WhisperEncOut {
    ggml_tensor *         tensor = nullptr;
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int                   d_model = 0;
    int                   T_enc   = 0;

    void free() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        tensor  = nullptr;
        d_model = 0;
        T_enc   = 0;
    }
};

bool enc_out_init(WhisperEncOut & enc_out,
                  ggml_backend_t  backend,
                  int             d_model,
                  int             T_enc);

// Active-KV padding for the self-attention step graph. whisper.cpp's
// whisper_kv_cache_get_padding policy: 32 on Metal+FA, 1 otherwise
// (CUDA path uses 256 but is not exercised here yet). Padding the
// active n_kv lets the FA op land on a kernel that prefers shapes
// aligned to 32 — the trailing positions are masked to -inf so they
// do not contribute. Returns 1 (no padding) when use_flash is false.
int kv_pad_self_attn(transcribe::BackendKind kind, bool use_flash);

// Cross-attention cache padding multiple. Universal in whisper.cpp
// (not gated on backend); the cost is 36 unused rows per layer for
// whisper-tiny and the small cross-attn dilution that comes with it.
constexpr int k_cross_kv_pad = 256;

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
// Per-stage timing counters for performance diagnosis.
//
// Always-on (the timestamps themselves are negligible). Printing is
// opt-in via TRANSCRIBE_WHISPER_PROFILE=1 — see whisper_run /
// commit_result. Reset at the top of every whisper_run.
//
// Counters are organized by run-stage so a single summary line per
// stage shows the per-call breakdown of build / alloc / compute /
// tensor_get / cpu — matching the structural targets of Phases 2-4.
// ---------------------------------------------------------------------------

struct WhisperPerfStage {
    int64_t total_us = 0;
    int     count    = 0;

    void add(int64_t us) {
        total_us += us;
        count    += 1;
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
    // negligible) but only printed when TRANSCRIBE_WHISPER_PROFILE
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
        enc_build.reset();         enc_alloc.reset();
        enc_compute.reset();       enc_tensor_get.reset();
        cross_build.reset();       cross_alloc.reset();
        cross_compute.reset();
        prompt_build.reset();      prompt_alloc.reset();
        prompt_compute.reset();    prompt_tensor_get.reset();
        prompt_cpu.reset();
        step_build.reset();        step_alloc.reset();
        step_compute.reset();      step_tensor_get.reset();
        step_cpu.reset();
        prompt_cpu_suppress.reset();  prompt_cpu_timestamp.reset();
        prompt_cpu_sample.reset();    prompt_cpu_logprob.reset();
        step_cpu_suppress.reset();    step_cpu_timestamp.reset();
        step_cpu_sample.reset();      step_cpu_logprob.reset();
        chunks = 0;
    }
};

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
    ggml_context *        compute_ctx      = nullptr;
    // Currently-allocated capacity of compute_ctx (mem_size). Used by
    // ensure_compute_ctx to decide between ggml_reset (cheap reuse)
    // and ggml_free + ggml_init (only when more space is needed).
    size_t                compute_ctx_size = 0;
    ggml_backend_sched_t  sched       = nullptr;

    // Persistent backend-resident encoder output. Encoder graph
    // appends a ggml_cpy into enc_out.tensor as its final op so the
    // cross-KV graph can read the encoder output via a view without
    // a GPU→CPU→GPU roundtrip.
    WhisperEncOut enc_out;

    // Host-side mirror of the encoder output, populated only when a
    // path needs an F32 input fed via ggml_backend_tensor_set
    // (currently: language detection on the first chunk and any
    // dump-emitting validation runs). Kept off the per-chunk hot
    // path so cross-KV no longer requires the materialization.
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

    // Per-chunk decoding trace for the most recent successful run —
    // populated by whisper_run as the chunk loop executes and exposed
    // via transcribe_get_whisper_chunk_count / _get_whisper_chunk_trace.
    // Cleared at the top of each run alongside cc->clear_result().
    std::vector<transcribe_whisper_chunk_trace> chunk_traces;

    // Per-stage timing counters. Always populated; a final summary is
    // printed when TRANSCRIBE_WHISPER_PROFILE=1.
    WhisperPerf perf;

    // Reusable scratch for the multinomial T>0 sampler. Sized to
    // vocab_size on first use; kept across steps to avoid the per-call
    // double[vocab] allocation in the hot path. Argmax / T==0 do not
    // touch this buffer.
    std::vector<double> sample_scratch;

    WhisperContext() = default;
    ~WhisperContext() override;
};

} // namespace transcribe::whisper
