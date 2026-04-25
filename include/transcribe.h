/*
 * transcribe.h - public C API for transcribe.cpp
 *
 * One-header public surface. Callers never need to include <ggml.h>.
 *
 * Threading:
 * - transcribe_model_* functions are thread-safe.
 * - A loaded model may be shared across any number of threads and used to
 *   create multiple contexts in parallel.
 * - A model must outlive every context created from it.
 * - transcribe_model_free() is only valid after all derived contexts have
 *   been freed and no thread is still using the model.
 * - transcribe_context_* functions are not thread-safe.
 * - A context must be used by at most one thread at a time.
 * - A context may be moved between threads if no two threads ever use it
 *   concurrently.
 * - The log callback may be invoked from any thread (including ggml
 *   worker threads); the userdata pointer must be safe for concurrent
 *   access. The callback must not assume it runs on the caller thread.
 * - transcribe_log_set() should be called ONCE at process startup,
 *   before any models are loaded, contexts are created, or worker
 *   threads exist. That is the only supported usage model in 0.x. The
 *   implementation publishes the callback with release semantics so any
 *   thread that later loads it observes both the callback and any state
 *   the installer wrote before publication. This protection covers the
 *   single startup-time install only.
 * - Calling transcribe_log_set() after threads or models exist is
 *   UNSUPPORTED in 0.x. The implementation is data-race-free (the two
 *   atomics ensure no torn reads), but the API does NOT guarantee any
 *   of the following under concurrent or mid-run reconfiguration:
 *     (a) pair-atomic publication of (callback, userdata) - an emitter
 *         may observe a callback from one generation paired with a
 *         userdata from another, especially if two threads call
 *         transcribe_log_set concurrently;
 *     (b) that the old callback or old userdata will not be invoked
 *         after transcribe_log_set returns - an in-flight emission on
 *         another thread may already be holding the previous pair on
 *         its stack and about to call through it.
 *   A caller that needs mid-run reconfiguration must provide its own
 *   external synchronization between transcribe_log_set callers AND
 *   keep every previous callback target and userdata alive until it
 *   can prove no thread can still emit with them. A future revision
 *   may add a properly synchronized reconfiguration entry point with
 *   stronger guarantees.
 *
 * Params and ABI stability:
 * - Every public params struct ships with a transcribe_*_default_params()
 *   factory function. Callers MUST use the factory rather than zero-
 *   initializing the struct themselves. New fields are added only at the
 *   end of structs.
 * - This is the whisper.cpp / llama.cpp discipline. It is NOT a real
 *   forward ABI: it relies on the caller being rebuilt against the same
 *   header version it links against. If you upgrade transcribe.cpp, you
 *   must rebuild your caller. Mixing a newer library with an older
 *   caller's smaller params struct is undefined behavior - the library
 *   will read past the end of the caller's struct.
 * - This is acceptable for transcribe.cpp because the project is at
 *   version 0.x and the deployment story is "vendored or rebuilt by the
 *   binding maintainer", not "system-installed shared library that
 *   ships independently of its consumers". When the project moves toward
 *   a stable shared-library deployment, a struct_size / version field
 *   will be added to each params struct as the first member. That change
 *   is itself an ABI break, which is fine within 0.x.
 *
 * Results:
 * - Results are owned by the context and exposed via accessor functions.
 *   Calling transcribe_run() replaces the previous result. All accessors
 *   are safe to call before the first transcribe_run() and return safe
 *   sentinels ("", 0, NaN) when no result is present.
 * - All timestamps are int64_t milliseconds, relative to the original
 *   input audio (never to internally padded audio).
 * - All counts and indices are int.
 * - Out-of-bounds indices are programmer error: in debug builds the
 *   library may assert; in release builds accessors return safe sentinels
 *   instead of reading invalid memory. Callers should bounds-check with
 *   the corresponding transcribe_n_*() accessor before indexing.
 */

#ifndef TRANSCRIBE_H
#define TRANSCRIBE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef TRANSCRIBE_API
#  if defined(_WIN32) && !defined(__GNUC__)
#    ifdef TRANSCRIBE_BUILD
#      define TRANSCRIBE_API __declspec(dllexport)
#    else
#      define TRANSCRIBE_API __declspec(dllimport)
#    endif
#  else
#    define TRANSCRIBE_API __attribute__((visibility("default")))
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------------- */
/* Status                                                                  */
/* ----------------------------------------------------------------------- */

typedef enum {
    TRANSCRIBE_OK                       = 0,
    TRANSCRIBE_ERR_INVALID_ARG          = 1,
    TRANSCRIBE_ERR_NOT_IMPLEMENTED      = 2,
    TRANSCRIBE_ERR_FILE_NOT_FOUND       = 3,
    TRANSCRIBE_ERR_GGUF                 = 4,
    TRANSCRIBE_ERR_UNSUPPORTED_ARCH     = 5,
    TRANSCRIBE_ERR_UNSUPPORTED_VARIANT  = 6,
    TRANSCRIBE_ERR_OOM                  = 7,
    TRANSCRIBE_ERR_BACKEND              = 8,
    /*
     * Reserved. v1 accepts only 16 kHz mono float32 PCM and the library
     * does not link a resampler, so there is currently no caller-supplied
     * sample rate to validate. This code is preserved at its assigned
     * value for the future "caller declares input rate" entry point.
     */
    TRANSCRIBE_ERR_SAMPLE_RATE          = 9,
    TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE = 10,
    TRANSCRIBE_ERR_UNSUPPORTED_TASK     = 11,
    /*
     * Returned by transcribe_run when the caller asks for a
     * timestamp granularity finer than the model can produce.
     * Compared against transcribe_capabilities::max_timestamp_kind,
     * which is set per family at load time. AUTO never trips this
     * code: the dispatcher passes AUTO through unconditionally and
     * the per-family run() handler resolves it to the model's
     * maximum when it assembles the result.
     */
    TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS = 12,
    /*
     * Returned by transcribe_run when the caller's abort callback
     * returns true during the run. Partial results from chunks that
     * completed before the abort are preserved on the context and
     * readable via the normal result accessors; transcribe_was_aborted
     * distinguishes partial-from-abort from complete.
     */
    TRANSCRIBE_ERR_ABORTED              = 13,
} transcribe_status;

/*
 * Returns a statically allocated, NUL-terminated description of a status
 * code. The returned pointer must not be freed and is valid for the
 * lifetime of the process.
 *
 * The parameter is `int` (not `transcribe_status`) for two reasons:
 *   1. It matches the convention of `strerror(int)` and lets callers pass
 *      an arbitrary integer without forming a bogus enum value.
 *   2. Casting an out-of-range integer to a C++ enum without a fixed
 *      underlying type is undefined behavior. Taking an int means the
 *      "out of range returns 'unknown status'" contract can be exercised
 *      portably without tripping UBSan.
 *
 * Any value of transcribe_status converts implicitly to int at the call
 * site, so existing C and C++ callers continue to compile unchanged.
 */
TRANSCRIBE_API const char * transcribe_status_string(int status);

/* ----------------------------------------------------------------------- */
/* Logging                                                                 */
/* ----------------------------------------------------------------------- */

/*
 * Numeric values mirror GGML_LOG_LEVEL_* exactly so internal pass-through
 * from ggml is a free reinterpret. If ggml ever renumbers, this enum is
 * the source of truth for callers and the internal pass-through layer is
 * responsible for honoring that contract.
 */
typedef enum {
    TRANSCRIBE_LOG_LEVEL_NONE  = 0,
    TRANSCRIBE_LOG_LEVEL_INFO  = 1,
    TRANSCRIBE_LOG_LEVEL_WARN  = 2,
    TRANSCRIBE_LOG_LEVEL_ERROR = 3,
    TRANSCRIBE_LOG_LEVEL_DEBUG = 4,
    TRANSCRIBE_LOG_LEVEL_CONT  = 5, /* continue previous line */
} transcribe_log_level;

typedef void (*transcribe_log_callback)(
    transcribe_log_level level,
    const char *         msg,
    void *               userdata);

/* Global log sink. Pass NULL cb to disable logging. */
TRANSCRIBE_API void transcribe_log_set(transcribe_log_callback cb, void * userdata);

/* ----------------------------------------------------------------------- */
/* Task / timestamps                                                       */
/* ----------------------------------------------------------------------- */

typedef enum {
    TRANSCRIBE_TASK_TRANSCRIBE = 0,
    TRANSCRIBE_TASK_TRANSLATE  = 1,
} transcribe_task;

/*
 * AUTO timestamp policy: AUTO is treated as "equal to the model's
 * max_timestamp_kind." The dispatcher never rejects AUTO, and the
 * per-family run() handler resolves it to the finest granularity
 * the model can actually produce when it assembles the result. A
 * non-AUTO request is treated as a ceiling: if the request is
 * finer than the model's max, transcribe_run returns
 * TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS. If the request is
 * coarser-or-equal, the family handler emits only that granularity
 * and any finer per-run data is elided. The actual granularity
 * returned by a run is reported by
 * transcribe_returned_timestamp_kind(ctx).
 */
typedef enum {
    TRANSCRIBE_TIMESTAMPS_NONE    = 0,
    TRANSCRIBE_TIMESTAMPS_AUTO    = 1,
    TRANSCRIBE_TIMESTAMPS_SEGMENT = 2,
    TRANSCRIBE_TIMESTAMPS_WORD    = 3,
    TRANSCRIBE_TIMESTAMPS_TOKEN   = 4,
} transcribe_timestamp_kind;

/*
 * KV cache type for the encoder's flash-attention path.
 *
 * AUTO:  f16 when model weights are quantized, f32 when weights are f32.
 *        Best default — matches the bandwidth profile of the model.
 * F32:   full-precision KV. Maximum accuracy, highest bandwidth.
 * F16:   half-precision KV. Halves attention bandwidth with minimal
 *        precision loss (~3 decimal digits). Best for bandwidth-bound
 *        backends (integrated GPUs, CPU).
 */
typedef enum {
    TRANSCRIBE_KV_TYPE_AUTO = 0,
    TRANSCRIBE_KV_TYPE_F32  = 1,
    TRANSCRIBE_KV_TYPE_F16  = 2,
} transcribe_kv_type;

/* ----------------------------------------------------------------------- */
/* Handles                                                                 */
/* ----------------------------------------------------------------------- */

struct transcribe_model;
struct transcribe_context;

/* ----------------------------------------------------------------------- */
/* Params                                                                  */
/* ----------------------------------------------------------------------- */

/*
 * Backend request.
 *
 * AUTO    Pick the best available backend. Takes the first GPU/IGPU
 *         device that successfully initializes in ggml's device
 *         registry order — which is build-time prioritized (Metal on
 *         Apple, Vulkan / CUDA / SYCL on Linux, …). Host-memory
 *         accelerators (BLAS, AMX, …) are additionally layered onto
 *         the scheduler when present — they run on the same memory
 *         as the CPU backend and are orthogonal to the GPU/CPU split.
 *         Always succeeds: CPU is the final fallback when no GPU
 *         initializes.
 *
 * CPU     Strict CPU only. No GPU, no IGPU, and no host-memory
 *         accelerators (BLAS/AMX). This is the right choice for
 *         numerical reference runs, cross-platform determinism, and
 *         for exercising the pure-CPU kernels under test. Always
 *         succeeds.
 *
 * CPU_ACCEL  CPU primary plus host-memory accelerators (BLAS/AMX/…
 *            whatever ggml registers as ACCEL). primary_kind stays
 *            Cpu so any CPU-keyed policy still triggers, but mat-mul-
 *            shaped graph nodes that fit the accel backend's gates
 *            (large F32 src1, ne ≥ 32) get dispatched there. Useful
 *            when GPU is unavailable or undesired but you still want
 *            production-grade CPU throughput. Requires the build to
 *            include the relevant accel backend (e.g. GGML_BLAS=ON).
 *            Falls back to plain CPU semantics if no accel device is
 *            registered. Always succeeds.
 *
 * METAL   Require the Metal backend. Returns TRANSCRIBE_ERR_BACKEND
 *         if Metal is not available in this build. Host-memory
 *         accelerators are still layered on when present.
 *
 * VULKAN  Require the Vulkan backend. Returns TRANSCRIBE_ERR_BACKEND
 *         if Vulkan is not available in this build. Host-memory
 *         accelerators are still layered on when present.
 *
 * Callers that need to know which backend they actually landed on
 * can query transcribe_model_backend() after load.
 */
typedef enum {
    TRANSCRIBE_BACKEND_AUTO      = 0,
    TRANSCRIBE_BACKEND_CPU       = 1,
    TRANSCRIBE_BACKEND_METAL     = 2,
    TRANSCRIBE_BACKEND_VULKAN    = 3,
    TRANSCRIBE_BACKEND_CPU_ACCEL = 4,
} transcribe_backend_request;

/*
 * Model load params.
 *
 * backend:    which backend to request. See transcribe_backend_request
 *             for the semantics of each value. Default is AUTO.
 *
 * gpu_device: Reserved for future multi-device selection. MUST be
 *             set to -1 in 0.x; any other value returns
 *             TRANSCRIBE_ERR_INVALID_ARG. AUTO always picks the first
 *             device of the chosen kind in ggml's registry order;
 *             explicit METAL/VULKAN requests likewise pick the first
 *             matching device. There is no per-device selection in
 *             the current release.
 */
struct transcribe_model_params {
    transcribe_backend_request backend;
    int                        gpu_device;
};

TRANSCRIBE_API struct transcribe_model_params transcribe_model_default_params(void);

/*
 * Context init params.
 *
 * n_threads: number of CPU threads for ops that run on CPU. 0 means
 *            "library picks a sensible default".
 *
 * kv_type:   data type for K/V activations in flash attention.
 *            AUTO (default) uses f16 for quantized models, f32 for f32.
 */
struct transcribe_context_params {
    int                n_threads;
    transcribe_kv_type kv_type;
};

TRANSCRIBE_API struct transcribe_context_params transcribe_context_default_params(void);

/*
 * Per-run params.
 *
 * v1 accepts only 16 kHz mono float32 PCM. There is no caller-supplied
 * sample rate; the library does not link a resampler; the caller is
 * responsible for resampling external audio (e.g. with sox or ffmpeg)
 * before calling transcribe_run. A future revision will introduce a
 * caller-declared input rate, at which point TRANSCRIBE_ERR_SAMPLE_RATE
 * (currently reserved) will become observable.
 *
 * task:        TRANSCRIBE or TRANSLATE. The model must declare support
 *              for translate via its capabilities; otherwise the run
 *              returns TRANSCRIBE_ERR_UNSUPPORTED_TASK.
 *
 * timestamps:  requested granularity. Use AUTO to get the finest the
 *              model supports.
 *
 * language:        source language hint as a BCP-47-ish short code, or
 *                  NULL to autodetect (only if the model supports it).
 *
 * target_language: target language for translation tasks, or NULL.
 *
 * strip_special_tags: strip special vocabulary tags (e.g. <|...|>) from
 *                     the returned text fields. Token-level accessors
 *                     still expose the raw token text.
 */
struct transcribe_whisper_params;

struct transcribe_params {
    transcribe_task           task;
    transcribe_timestamp_kind timestamps;
    const char *              language;
    const char *              target_language;
    bool                      strip_special_tags;

    /*
     * Family-specific extension pointer. NULL selects family defaults.
     * Only consulted when the loaded model's architecture matches the
     * pointed-to struct's family; other families ignore it.
     *
     * Placed at the end of the struct so new fields appended after it
     * in future minor versions do not shift the layout of any earlier
     * field. This is source-compatible (callers rebuild against the
     * new header and pick up the expanded struct cleanly) but NOT
     * forward-ABI-safe: a pre-Stage-1 caller linked against an older
     * binary whose struct ended before `whisper` would have the new
     * library read past the end of its stack-allocated params object.
     * See "Params and ABI stability" above — 0.x is vendored/rebuilt,
     * not system-installed-independent, and callers MUST rebuild when
     * they upgrade transcribe.cpp.
     */
    const struct transcribe_whisper_params * whisper;
};

TRANSCRIBE_API struct transcribe_params transcribe_default_params(void);

/* ----------------------------------------------------------------------- */
/* Whisper-specific params                                                 */
/* ----------------------------------------------------------------------- */

/*
 * Sentinels for "disabled" on threshold fields. Never rely on 0.0 as a
 * sentinel — it is a legitimate value for logprob_thold and lies in
 * the normal range of the other thresholds.
 *
 *   Check shape "X > thold"  →  disable with +INF (_THOLD_DISABLED).
 *   Check shape "X < thold"  →  disable with -INF (_LOGPROB_DISABLED).
 */
#define TRANSCRIBE_WHISPER_THOLD_DISABLED   ((float) (1.0 / 0.0))  /* +INF */
#define TRANSCRIBE_WHISPER_LOGPROB_DISABLED ((float) (-1.0 / 0.0)) /* -INF */

/*
 * How an initial prompt composes with condition_on_prev_tokens on
 * long-form runs. Matches HF's prompt_condition_type string knob.
 *
 *   FIRST_SEGMENT (default): the initial prompt sits at the head of
 *     the first chunk's prefix. If condition_on_prev_tokens is true,
 *     subsequent chunks carry the decoded prior-chunk tokens under
 *     <|startofprev|> and the initial prompt is visible only while
 *     it still fits in the sliding 223-token window.
 *
 *   ALL_SEGMENTS: the initial prompt replaces <|startofprev|> as the
 *     persistent BOS of the prev-context window on EVERY chunk.
 *     Requires condition_on_prev_tokens=true; passing this mode
 *     without condition_on_prev_tokens returns
 *     TRANSCRIBE_ERR_INVALID_ARG (mirrors HF's ValueError).
 */
enum transcribe_whisper_prompt_condition {
    TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT = 0,
    TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS  = 1,
};

/*
 * Whisper-family run knobs. Reached via transcribe_params::whisper.
 * A NULL pointer uses transcribe_whisper_default_params() values.
 *
 * The defaults intentionally enable temperature fallback and the
 * compression-ratio / avg-logprob / no-speech safety nets. This
 * matches Whisper's own recipe (OpenAI whisper shipping default;
 * HF's own docstring advisory). Callers that want HF library-wide
 * generate() behavior ("all thresholds disabled") set each _thold
 * field to its _DISABLED sentinel.
 */
struct transcribe_whisper_params {
    /*
     * Initial prompt / prior context.
     *
     *   If prompt_tokens != NULL:
     *       Use prompt_tokens verbatim (caller owns the bytes; they must
     *       NOT include the <|startofprev|> marker — the library prepends
     *       it). initial_prompt is IGNORED.
     *
     *   Else if initial_prompt != NULL:
     *       Tokenize as HF's get_prompt_ids does:
     *           "<|startofprev|>" + " " + initial_prompt.strip()
     *       The leading space is mandatory (matches
     *       transformers tokenization_whisper.py:710-722). Any special
     *       token (<|...|>) found in the tokenized prompt text is
     *       rejected with TRANSCRIBE_ERR_INVALID_ARG, mirroring HF's
     *       own check.
     *
     *   Else: no initial prompt.
     */
    const char *    initial_prompt;
    const int32_t * prompt_tokens;
    size_t          n_prompt_tokens;

    /* See transcribe_whisper_prompt_condition above. Default FIRST_SEGMENT. */
    enum transcribe_whisper_prompt_condition prompt_condition;

    /*
     * Per HF generation_whisper.py:1853-1918. When true and any prior
     * chunk decoded successfully at temperature < 0.5, the tail of the
     * prior chunk's tokens is prepended (under <|startofprev|>) to the
     * next chunk's prefix, capped at max_prev_context_tokens. Auto-
     * disables for the next chunk when the prior chunk was accepted at
     * temperature >= 0.5 (matches HF ":1090-1093").
     */
    bool            condition_on_prev_tokens;

    /* Cap for carried prev tokens; default 223 = max_target_positions/2 - 1. */
    int32_t         max_prev_context_tokens;

    /*
     * Sampling + temperature fallback. Default behavior matches
     * Whisper's own recipe, not HF generate()'s library-default.
     * Use the _DISABLED sentinels to turn off individual thresholds.
     */
    float           temperature;             /* first-tier; default 0.0 */
    float           temperature_inc;         /* default 0.2             */
    float           compression_ratio_thold; /* default 2.4             */
    float           logprob_thold;           /* default -1.0            */
    float           no_speech_thold;         /* default 0.6             */

    /*
     * Seed for the sampler at temperature > 0. 0 = nondeterministic
     * (matches the whisper.cpp convention). A nonzero seed produces
     * reproducible output across runs at matching temperatures.
     * Ignored when temperature == 0.0 (greedy decode is deterministic
     * by construction).
     */
    uint32_t        seed;

    /* Seconds. Caps the first emitted timestamp; default 1.0. */
    float           max_initial_timestamp;
};

TRANSCRIBE_API struct transcribe_whisper_params transcribe_whisper_default_params(void);

/* ----------------------------------------------------------------------- */
/* Whisper decoding trace                                                  */
/* ----------------------------------------------------------------------- */

/*
 * Per-chunk observability: the temperature tier that the fallback loop
 * accepted, the metric values that drove acceptance, and whether the
 * no-speech gate fired. Populated on every successful transcribe_run()
 * call for a Whisper context; one entry per 30-second encoder window
 * that actually decoded (no-speech-skipped chunks still emit an entry).
 *
 * The chunk window `t0_ms .. t1_ms` is the global encoder slice, not a
 * segment boundary. Segments carried in transcribe_n_segments() / -
 * transcribe_segment_*() live inside these windows.
 *
 * Without the trace, every fallback bug is unresolvable — the exact
 * knob the caller twisted may not match the threshold that fired. Keep
 * this surface stable.
 */
struct transcribe_whisper_chunk_trace {
    int64_t  t0_ms;
    int64_t  t1_ms;
    float    temperature_used;    /* the tier that was accepted */
    float    compression_ratio;
    float    avg_logprob;
    float    no_speech_prob;
    bool     no_speech_triggered; /* chunk output was discarded */
    int32_t  n_fallbacks;         /* tiers tried before accept (0 = tier 0 accepted) */
};

/* Number of chunk traces captured on the most recent successful run.
 * Returns 0 before any run or on non-Whisper contexts. */
TRANSCRIBE_API int transcribe_get_whisper_chunk_count(
    const struct transcribe_context * ctx);

/* Trace at index [0, count). Out-of-range indices return a zeroed
 * struct. Non-Whisper contexts return a zeroed struct. */
TRANSCRIBE_API struct transcribe_whisper_chunk_trace
    transcribe_get_whisper_chunk_trace(
        const struct transcribe_context * ctx, int i);

/* ----------------------------------------------------------------------- */
/* Capabilities                                                            */
/* ----------------------------------------------------------------------- */

/*
 * Capabilities are semantic model properties read from GGUF KV (with
 * architecture defaults filling in absent keys). They are immutable
 * after a successful transcribe_model_load_file() and never change
 * based on backend fallback.
 *
 * languages is an array of n_languages NUL-terminated short codes,
 * owned by the model. The pointer is valid until transcribe_model_free.
 *
 * max_timestamp_kind is the finest granularity the family's run()
 * handler can produce. NONE means the model emits text but no
 * alignment data; SEGMENT / WORD / TOKEN expose progressively finer
 * alignment. The dispatcher rejects any transcribe_run request
 * whose timestamps field is finer than this value. AUTO is not
 * validated against this field: the dispatcher passes AUTO through
 * unconditionally and the per-family run() handler resolves AUTO
 * to its own max_timestamp_kind when it assembles the result.
 */
struct transcribe_capabilities {
    int32_t                   native_sample_rate;
    int                       n_languages;
    const char * const *      languages;
    bool                      supports_language_detect;
    bool                      supports_translate;
    bool                      supports_streaming;
    transcribe_timestamp_kind max_timestamp_kind;

    /*
     * Family-level feature advertisements, set by apply_family_invariants
     * and immutable after load. Added at the end of the struct; old
     * callers that only read the fields above keep working unchanged.
     */
    bool                      supports_initial_prompt;
    bool                      supports_temperature_fallback;
    bool                      supports_long_form;
    bool                      supports_cancellation;
};

TRANSCRIBE_API const struct transcribe_capabilities *
    transcribe_model_capabilities(const struct transcribe_model * model);

/*
 * Stable identifier strings read from GGUF and runtime state.
 *
 *   transcribe_model_arch_string(): the general.architecture key,
 *     e.g. "parakeet". Immutable for the lifetime of the model.
 *
 *   transcribe_model_variant_string(): the stt.variant key, e.g.
 *     "tdt-0.6b-v2". Immutable for the lifetime of the model. May be
 *     an empty string if the underlying GGUF did not carry a variant
 *     and the architecture handler had no default to substitute.
 *
 *   transcribe_model_backend(): the runtime backend currently bound
 *     to this model, e.g. "cpu", "metal", "vulkan", "cuda". This is
 *     the mechanism for detecting CPU fallback when GPU was requested.
 *
 *     Returns an empty string when no runtime backend is currently
 *     bound to the model. This covers both (a) model == NULL, the
 *     safe-sentinel pattern used throughout this header, and (b) a
 *     model that has been loaded but is not yet bound to a runtime
 *     backend - for example, a model whose on-disk metadata and
 *     vocabulary have been parsed but whose compute graph has not
 *     been materialized on any backend yet. Callers can distinguish
 *     the two by null-checking the model pointer.
 *
 *     Backend is a runtime fact, reported separately from the
 *     model's semantic capabilities (see
 *     transcribe_model_capabilities), which never change based on
 *     backend state.
 *
 * All three return statically allocated or model-owned strings; the
 * caller must not free them and they remain valid until the model is
 * freed.
 */
TRANSCRIBE_API const char * transcribe_model_arch_string(const struct transcribe_model * model);
TRANSCRIBE_API const char * transcribe_model_variant_string(const struct transcribe_model * model);
TRANSCRIBE_API const char * transcribe_model_backend(const struct transcribe_model * model);

/* ----------------------------------------------------------------------- */
/* Lifecycle                                                               */
/* ----------------------------------------------------------------------- */

/*
 * Load a GGUF model from disk. The _file suffix is preserved to leave
 * room for transcribe_model_load_buffer() later without renaming.
 *
 * On success, *out_model is set and the caller owns it. On failure,
 * *out_model is set to NULL and a non-OK status is returned.
 */
TRANSCRIBE_API transcribe_status transcribe_model_load_file(
    const char *                           path,
    const struct transcribe_model_params * params,
    struct transcribe_model **             out_model);

/*
 * Free a model. Only valid after every context derived from this model
 * has been freed and no thread is still using the model. Passing NULL
 * is a no-op.
 */
TRANSCRIBE_API void transcribe_model_free(struct transcribe_model * model);

/*
 * Initialize a transcription context bound to a loaded model. Multiple
 * contexts may be created from the same model in parallel; each context
 * is single-threaded.
 */
TRANSCRIBE_API transcribe_status transcribe_context_init(
    struct transcribe_model *                model,
    const struct transcribe_context_params * params,
    struct transcribe_context **             out_ctx);

/* Free a context. Passing NULL is a no-op. */
TRANSCRIBE_API void transcribe_context_free(struct transcribe_context * ctx);

/* ----------------------------------------------------------------------- */
/* Run                                                                     */
/* ----------------------------------------------------------------------- */

/*
 * Run one batch transcription.
 *
 * pcm:        mono float32 PCM samples in [-1.0, 1.0] at 16 kHz.
 * n_samples:  number of samples in pcm.
 * params:     run params.
 *
 * v1 supports only 16 kHz mono float32 PCM and does not link a
 * resampler; the caller is responsible for resampling external audio
 * before calling this function.
 *
 * On success, results are populated on the context and may be read via
 * the accessors below. Calling transcribe_run() again replaces the
 * previous result on the same context.
 */
TRANSCRIBE_API transcribe_status transcribe_run(
    struct transcribe_context *      ctx,
    const float *                    pcm,
    int                              n_samples,
    const struct transcribe_params * params);

/* ----------------------------------------------------------------------- */
/* Cancellation                                                            */
/* ----------------------------------------------------------------------- */

/*
 * Abort callback. Polled between chunks AND between decode steps (after
 * each token). Abort latency is therefore one decode step — typically
 * 10-50 ms on CPU, sub-ms on Metal. Mid-encoder abort is not supported
 * today (would require ggml hooks that do not exist).
 *
 * Returning true aborts the in-flight run: transcribe_run returns
 * TRANSCRIBE_ERR_ABORTED and result accessors (transcribe_full_text,
 * transcribe_n_segments, etc.) expose the partial content from chunks
 * that completed before abort. Use transcribe_was_aborted(ctx) to
 * distinguish partial-from-abort from complete.
 *
 * The next successful transcribe_run() replaces the result atomically,
 * as today.
 *
 * The callback fires on the run thread. If the caller wants to trigger
 * abort from another thread, the common pattern is to store an
 * std::atomic<bool> inside user_data and have the callback load it.
 */
typedef bool (*transcribe_abort_callback)(void * user_data);

/*
 * Install or clear the abort callback for a context. Passing cb=NULL
 * clears any previously installed callback. Safe to call before the
 * first transcribe_run. No-op if ctx is NULL.
 *
 * Not thread-safe with respect to an in-flight transcribe_run on the
 * same context — the context is single-threaded-at-a-time per the
 * threading contract above. Callers that need to trigger abort from
 * another thread should do so by flipping state inside the callback's
 * user_data, not by swapping the callback itself.
 */
TRANSCRIBE_API void transcribe_set_abort_callback(
    struct transcribe_context *  ctx,
    transcribe_abort_callback    cb,
    void *                       user_data);

/*
 * True if the most recent transcribe_run was aborted by the installed
 * callback returning true. Reset to false at the top of each
 * transcribe_run. Returns false if ctx is NULL.
 */
TRANSCRIBE_API bool transcribe_was_aborted(const struct transcribe_context * ctx);

/* ----------------------------------------------------------------------- */
/* Tokenization                                                            */
/* ----------------------------------------------------------------------- */

/*
 * Tokenize plain UTF-8 text into the model's vocabulary, with
 * add_special_tokens=False semantics (no BOS/EOS, no <|...|> markers).
 * Special tokens present in the input are not recognized and will be
 * BPE-encoded piece-by-piece, which is almost never what the caller
 * wants — render special tokens via direct id paths and pass only the
 * plain-text fragments through this function.
 *
 * Buffer contract (mirrors whisper.cpp's n_max convention):
 *
 *   >= 0             Number of tokens written to `tokens[0..return-1]`.
 *   negative of N    Buffer too small; N tokens were needed. Caller
 *                    reallocates and retries. `tokens` content is
 *                    unspecified in this case (may be partially
 *                    written).
 *   INT_MIN          Hard error: model or text is NULL, the model's
 *                    tokenizer does not support encode for this
 *                    vocabulary (e.g. a SentencePiece family without
 *                    encode wired), or the vocab file is malformed.
 *
 * Plumbed per family:
 *   - Whisper      (GPT-2 byte-level BPE)
 *   - Qwen3-ASR    (Qwen2 byte-level BPE)
 *   - Parakeet     (SentencePiece; currently returns INT_MIN)
 *   - Cohere ASR   (SentencePiece; currently returns INT_MIN)
 */
TRANSCRIBE_API int transcribe_tokenize(
    const struct transcribe_model * model,
    const char *                    text,
    int32_t *                       tokens,
    size_t                          n_max);

/* ----------------------------------------------------------------------- */
/* Timings                                                                 */
/* ----------------------------------------------------------------------- */

/*
 * Per-call timings collected by the most recent transcribe_run on a
 * context, plus the wall-clock load time of the model the context is
 * bound to. All values are in milliseconds.
 *
 *   load_ms    one-time wall-clock cost of transcribe_model_load_file,
 *              captured on the model and surfaced via every context
 *              derived from it. Not affected by transcribe_reset_timings
 *              (it's a model-scoped fact).
 *   mel_ms     time to compute the mel front-end on the most recent
 *              transcribe_run. Reset to 0 by transcribe_reset_timings.
 *   encode_ms  time to run the encoder forward pass on the most recent
 *              transcribe_run.
 *   decode_ms  time to run the decoder (predictor + joint + token
 *              search) on the most recent transcribe_run. Currently
 *              always 0 since the decoder isn't wired yet; phase 5
 *              fills this in.
 *
 * The struct shape follows whisper.cpp's whisper_timings: a small
 * value type returned by an accessor, no allocation crossing the FFI.
 * New fields will only be added at the end of the struct (the same
 * "factory + add at end" rule the params structs follow); within 0.x
 * this is not a forward-ABI guarantee — see "Params and ABI
 * stability" elsewhere in this header.
 */
struct transcribe_timings {
    float load_ms;
    float mel_ms;
    float encode_ms;
    float decode_ms;
};

/*
 * Read the current timings from a context. Safe to call before any
 * transcribe_run; mel_ms / encode_ms / decode_ms will be 0. load_ms
 * is non-zero as soon as the underlying model is loaded.
 *
 * Returns a zeroed struct if ctx is NULL.
 */
TRANSCRIBE_API struct transcribe_timings
transcribe_get_timings(const struct transcribe_context * ctx);

/*
 * Pretty-print the current timings to the registered log callback at
 * INFO level (or stderr if no callback is installed). No-op if ctx
 * is NULL.
 */
TRANSCRIBE_API void
transcribe_print_timings(const struct transcribe_context * ctx);

/*
 * Reset the per-run timing accumulators (mel_ms, encode_ms,
 * decode_ms) to 0. Does NOT touch load_ms — that's a model-scoped
 * fact. No-op if ctx is NULL.
 */
TRANSCRIBE_API void
transcribe_reset_timings(struct transcribe_context * ctx);

/* ----------------------------------------------------------------------- */
/* Result accessors - top level                                            */
/* ----------------------------------------------------------------------- */

TRANSCRIBE_API const char *               transcribe_full_text(const struct transcribe_context * ctx);
TRANSCRIBE_API transcribe_timestamp_kind  transcribe_returned_timestamp_kind(const struct transcribe_context * ctx);
TRANSCRIBE_API int                        transcribe_n_segments(const struct transcribe_context * ctx);
TRANSCRIBE_API int                        transcribe_n_words(const struct transcribe_context * ctx);
TRANSCRIBE_API int                        transcribe_n_tokens(const struct transcribe_context * ctx);

/* ----------------------------------------------------------------------- */
/* Result accessors - segment level                                        */
/* ----------------------------------------------------------------------- */

TRANSCRIBE_API const char * transcribe_segment_text       (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int64_t      transcribe_segment_t0_ms      (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int64_t      transcribe_segment_t1_ms      (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int          transcribe_segment_first_word (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int          transcribe_segment_n_words    (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int          transcribe_segment_first_token(const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int          transcribe_segment_n_tokens   (const struct transcribe_context * ctx, int i);

/* ----------------------------------------------------------------------- */
/* Result accessors - word level                                           */
/* ----------------------------------------------------------------------- */

TRANSCRIBE_API const char * transcribe_word_text       (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int64_t      transcribe_word_t0_ms      (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int64_t      transcribe_word_t1_ms      (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int          transcribe_word_seg_index  (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int          transcribe_word_first_token(const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int          transcribe_word_n_tokens   (const struct transcribe_context * ctx, int i);

/* ----------------------------------------------------------------------- */
/* Result accessors - token level                                          */
/* ----------------------------------------------------------------------- */

TRANSCRIBE_API int          transcribe_token_id        (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API const char * transcribe_token_text      (const struct transcribe_context * ctx, int i);
/*
 * transcribe_token_p returns the per-token probability when the
 * architecture produces one, or NaN when it does not. The semantic of
 * "token probability" varies by family (joint softmax for transducer,
 * per-step softmax for autoregressive, per-frame argmax probability for
 * CTC) and callers should treat it as a confidence hint, not a
 * calibrated probability.
 */
TRANSCRIBE_API float        transcribe_token_p         (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int64_t      transcribe_token_t0_ms     (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int64_t      transcribe_token_t1_ms     (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int          transcribe_token_seg_index (const struct transcribe_context * ctx, int i);
TRANSCRIBE_API int          transcribe_token_word_index(const struct transcribe_context * ctx, int i);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRANSCRIBE_H */
