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
 * Timestamp policy: transcribe_default_params() requests NONE for
 * text-first transcription. AUTO is an opt-in "richest supported"
 * mode: it is treated as "equal to the model's max_timestamp_kind."
 * The dispatcher never rejects AUTO, and the per-family run() handler
 * resolves it to the finest granularity the model can actually
 * produce when it assembles the result. A non-AUTO request is treated
 * as a ceiling: if the request is finer than the model's max,
 * transcribe_run returns TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS. If the
 * request is coarser-or-equal, the family handler emits only that
 * granularity and any finer per-run data is elided. The actual
 * granularity returned by a run is reported by
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
 * timestamps:  requested granularity. Default params request NONE.
 *              Use AUTO to get the finest granularity the model
 *              supports.
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
struct transcribe_sensevoice_params;
struct transcribe_funasr_nano_params;
struct transcribe_canary_params;

struct transcribe_params {
    transcribe_task           task;
    transcribe_timestamp_kind timestamps;
    const char *              language;
    const char *              target_language;
    bool                      strip_special_tags;

    /*
     * Family-specific extension pointers. NULL selects family defaults.
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
    const struct transcribe_whisper_params *     whisper;
    const struct transcribe_sensevoice_params *  sensevoice;
    const struct transcribe_funasr_nano_params * funasr_nano;
    const struct transcribe_canary_params *      canary;
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
/* SenseVoice-specific params                                              */
/* ----------------------------------------------------------------------- */

/*
 * SenseVoiceSmall family knobs. Reached via transcribe_params::sensevoice.
 * A NULL pointer selects transcribe_sensevoice_default_params() values.
 *
 * SenseVoice is non-autoregressive (encoder + CTC head) and ships four
 * input prefix-embedding slots that condition the encoder on language,
 * audio-event, emotion, and inverse text normalization (ITN). The
 * language slot is driven by the family-agnostic transcribe_params::language
 * field; the audio-event and emotion slots are hard-coded by the upstream
 * inference path so they emerge in the OUTPUT (not as input prefixes).
 * The ITN slot is the only remaining user-facing knob:
 *
 *   use_itn = false (default): encoder receives the `woitn` prefix, and
 *     the CTC head emits raw transcription with no inverse text
 *     normalization. The control token `<|woitn|>` rides along in the
 *     output.
 *
 *   use_itn = true: encoder receives the `withitn` prefix, the CTC head
 *     applies inverse text normalization (numbers/punctuation rendered
 *     in formal form), and `<|withitn|>` rides along in the output.
 *
 * NOTE: the runtime cannot enable ITN for languages the upstream model
 * was not trained to ITN-format. SenseVoiceSmall ships ITN coverage for
 * its five advertised languages (zh, yue, en, ja, ko). Setting use_itn
 * on a non-supported audio is a no-op rather than an error.
 */
struct transcribe_sensevoice_params {
    bool use_itn;
};

TRANSCRIBE_API struct transcribe_sensevoice_params
    transcribe_sensevoice_default_params(void);

/* ----------------------------------------------------------------------- */
/* FunASR-Nano-specific params                                             */
/* ----------------------------------------------------------------------- */

/*
 * FunASR-Nano family knobs. Reached via transcribe_params::funasr_nano.
 * A NULL pointer selects transcribe_funasr_nano_default_params() values.
 *
 * FunASR-Nano is autoregressive (encoder + audio adaptor + bundled
 * Qwen3-0.6B LLM). The user-visible prompt knob is inverse text
 * normalization, controlled via the system prompt the LLM receives:
 *
 *   use_itn = false (default): the LLM prompt appends
 *     "，不进行文本规整" ("do not apply text normalization") so the
 *     decoded transcript stays close to verbatim — no digit/punctuation
 *     formatting, lowercase English. Matches the reference's
 *     `itn=False` Python path.
 *
 *   use_itn = true: the LLM prompt omits the no-ITN suffix and the
 *     decoder is free to render numbers, capitalization, and
 *     punctuation in formal form. Matches `itn=True` upstream.
 *
 * Applies to both fun-asr-nano-2512 (zh / en / ja, plus dialects /
 * accents) and fun-asr-mlt-nano-2512 (31 languages). The family shares
 * one prompt builder; ITN coverage per language is whatever the
 * upstream model was trained on.
 */
struct transcribe_funasr_nano_params {
    bool use_itn;
};

TRANSCRIBE_API struct transcribe_funasr_nano_params
    transcribe_funasr_nano_default_params(void);

/* ----------------------------------------------------------------------- */
/* Canary-specific params                                                  */
/* ----------------------------------------------------------------------- */

/*
 * Canary-family run knobs. Reached via transcribe_params::canary.
 * A NULL pointer selects transcribe_canary_default_params() values.
 *
 * pnc = true (default): the multitask prompt's pnc slot is set to
 *   <|pnc|>, and the AED emits text with punctuation and standard
 *   capitalization. This is the regime under which the upstream model
 *   card WER numbers are reported.
 *
 * pnc = false: the prompt slot is set to <|nopnc|>, and the AED emits
 *   lowercase, de-punctuated text — useful for downstream pipelines
 *   that re-punctuate or that need verbatim-style output.
 *
 * The pnc/nopnc tokens are positional in the multitask prompt; flipping
 * this changes one token id at the prompt-build site, nothing else.
 */
struct transcribe_canary_params {
    bool pnc;
};

TRANSCRIBE_API struct transcribe_canary_params
    transcribe_canary_default_params(void);

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

    /*
     * Streaming timing hints. Consumer-facing advisories: callers may use
     * them to size their audio capture chunks or display latency budgets,
     * but transcribe_stream_feed accepts arbitrary sample counts and
     * buffers internally. Zero means "unsupported or unknown" — a family
     * that advertises supports_streaming may legitimately leave one or
     * both fields zero if the value is not a fixed property of the model.
     * supports_streaming remains the hard capability gate.
     *
     * For multi-lookahead models (e.g. nemotron-speech-streaming-en-0.6b),
     * streaming_lookahead_ms reports the default setting's lookahead and
     * streaming_lookahead_ms_min reports the fastest setting available.
     * When the two differ the caller can pick a lower-latency setting via
     * the family-specific stream params extension (parakeet:
     * att_context_right). When the model has only one setting the two
     * fields are equal.
     */
    int32_t                   streaming_lookahead_ms;
    int32_t                   streaming_chunk_ms;
    int32_t                   streaming_lookahead_ms_min;
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
/* Streaming                                                               */
/* ----------------------------------------------------------------------- */

/*
 * Streaming is a mode on transcribe_context, not a separate handle. A
 * context is in exactly one of four lifecycle states at any time, and
 * the result accessors (transcribe_full_text, segments, words, tokens)
 * return the current cumulative snapshot of the active stream.
 *
 * Result snapshot semantics during an active stream:
 *
 *   result = committed prefix + tentative suffix
 *
 * The committed-count accessors (transcribe_stream_n_committed_*)
 * identify the stable prefix; everything past the boundary may be
 * replaced on the next feed/finalize call. Consumers MUST NOT cache
 * text, timestamps, probabilities, or indices at or beyond the
 * committed-count boundary across calls.
 *
 * Cancellation reuses the existing context abort callback. The
 * callback is polled at chunk boundaries and decode-step boundaries
 * during feed/finalize; on cancellation the active call returns
 * TRANSCRIBE_ERR_ABORTED, partial results remain readable, and the
 * stream transitions to FAILED (transcribe_was_aborted distinguishes
 * caller cancellation from other terminal statuses).
 *
 * Multiple concurrent streams: use the existing model/context
 * threading model. Create multiple contexts from one loaded model,
 * then run at most one stream per context. The model is shared and
 * read-only; each context owns its own stream state.
 *
 * v1 explicitly limits stream params to family-specific extension
 * pointers. Generic latency / partial-output knobs will be added when
 * a shipped family implements them with clear semantics.
 */

enum transcribe_stream_state {
    TRANSCRIBE_STREAM_IDLE     = 0,
    TRANSCRIBE_STREAM_ACTIVE   = 1,
    TRANSCRIBE_STREAM_FINISHED = 2,
    TRANSCRIBE_STREAM_FAILED   = 3,
};

/*
 * Family-specific streaming extensions. Defined per-family alongside
 * the corresponding run-params extension. NULL selects the family
 * default. Only consulted by the matching family; others ignore.
 *
 * Parakeet streaming:
 *
 *   att_context_right
 *
 *     Right-context (lookahead) selector in encoder frames. The
 *     cache-aware streaming variants (e.g. nemotron-speech-streaming-en-0.6b)
 *     are trained on a menu of (left, right) pairs simultaneously —
 *     the user picks one at inference time to trade latency for
 *     accuracy. nemotron's published menu is right ∈ {13, 6, 1, 0},
 *     corresponding to lookahead of {1040, 480, 80, 0} ms at the 80ms
 *     encoder frame rate.
 *
 *     -1 (default): use the model's default setting (first entry of
 *     att_context_size_choices = max-accuracy / max-latency).
 *     >=0: select the corresponding (left, att_context_right) entry
 *     from the model's training menu. transcribe_stream_begin returns
 *     TRANSCRIBE_ERR_INVALID_ARG if the requested right is not in the
 *     menu.
 *
 *     Inspect transcribe_capabilities::streaming_lookahead_ms to see
 *     the default's lookahead in milliseconds, and
 *     transcribe_capabilities::streaming_lookahead_ms_min to see the
 *     fastest setting the model supports.
 */
struct transcribe_parakeet_stream_params {
    int32_t att_context_right;
};

/*
 * Buffered-streaming knobs for the parakeet-unified family.
 *
 * parakeet-unified-en-0.6b is trained with chunked_limited_with_rc
 * attention over a menu of (left, chunk, right) context tuples
 * expressed in 80ms encoder frames. The user picks the active tuple
 * at stream_begin time; the encoder re-runs over each new
 * [left | chunk | right] PCM window. Each field is in MILLISECONDS;
 * the runtime converts to encoder frames via the 80ms frame rate.
 *
 * Use -1 (sentinel) on any field to get the model's "best accuracy"
 * default — L=5600 ms / C=1040 ms / R=1040 ms for unified-en-0.6b,
 * which the published WER numbers correspond to. Non-default tuples
 * are passed through to the encoder's set_default_att_context_size;
 * the runtime validates the resolved (L, C, R) against the model's
 * training menu (stt.parakeet.encoder.att_chunk_{left,chunk,right}_choices).
 * Tuples outside the menu return TRANSCRIBE_ERR_INVALID_ARG.
 */
struct transcribe_parakeet_buffered_stream_params {
    int32_t left_ms;
    int32_t chunk_ms;
    int32_t right_ms;
};

/*
 * Moonshine-Streaming family stream knobs.
 *
 *   min_decode_interval_ms
 *
 *     Minimum gap (in milliseconds of audio) between successive
 *     per-feed AR decoder runs. The runtime extends the encoder /
 *     adapter / cross-KV host buffers on every stream_feed that
 *     advances the committed encoder frame count, but only re-runs
 *     the AR decoder (and only flips update->result_changed) when
 *     at least this many milliseconds of new audio have committed
 *     since the previous decode.
 *
 *     The point of this knob is to decouple the caller's PCM feed
 *     cadence (often dictated by mic buffer size, typically 10-50 ms)
 *     from the partial-transcript update rate (a UX choice, typically
 *     ~250 ms). Without it, a caller pushing 10 ms chunks from a
 *     microphone would trigger a decode every 20 ms of audio for the
 *     entire stream — most of which would be wasted compute that the
 *     consumer never gets a chance to render.
 *
 *     Set 0 to decode on every encoder-frame advance (no throttle).
 *     Set -1 to use the family default (240 ms = one
 *     R_total = ~4 updates/sec after the initial right-context
 *     warmup; balances responsiveness with compute cost).
 *
 *     Note this is a *minimum* gap — the decoder always runs once at
 *     stream_finalize regardless of this value, and a feed that does
 *     not advance the committed encoder frame count never triggers a
 *     decode either way.
 */
struct transcribe_moonshine_streaming_stream_params {
    int32_t min_decode_interval_ms;
};

struct transcribe_stream_params {
    const struct transcribe_parakeet_stream_params *           parakeet;
    const struct transcribe_parakeet_buffered_stream_params *  parakeet_buffered;
    const struct transcribe_moonshine_streaming_stream_params *moonshine_streaming;
};

TRANSCRIBE_API struct transcribe_parakeet_stream_params
    transcribe_parakeet_stream_default_params(void);

TRANSCRIBE_API struct transcribe_parakeet_buffered_stream_params
    transcribe_parakeet_buffered_stream_default_params(void);

TRANSCRIBE_API struct transcribe_moonshine_streaming_stream_params
    transcribe_moonshine_streaming_stream_default_params(void);

/*
 * Optional per-call change metadata returned by feed/finalize.
 *
 *   result_changed     The context result was modified by this call.
 *                      Inspect the accessors after the call to read
 *                      the new snapshot.
 *   is_final           True only on the finalize call's update. Set
 *                      by the dispatcher after the family hook
 *                      returns; family hooks cannot override.
 *   revision           Monotonic counter that increments whenever the
 *                      context result changes (mirrors
 *                      transcribe_stream_revision(ctx) after the call
 *                      returns).
 *   input_received_ms  Total audio received by the stream since begin.
 *   audio_committed_ms Audio whose result is committed (corresponds
 *                      to the committed-prefix boundary).
 *   buffered_ms        Audio still buffered inside the family's
 *                      streaming state (frontend carry +
 *                      lookahead/right-context requirement). Caller
 *                      may use this as a "drain" hint.
 *
 * update is nullable. Passing NULL means the caller will inspect the
 * context directly (revision + committed-count accessors). Audio
 * cursor fields are only available via this struct.
 */
struct transcribe_stream_update {
    bool    result_changed;
    bool    is_final;
    int32_t revision;
    int64_t input_received_ms;
    int64_t audio_committed_ms;
    int64_t buffered_ms;
};

TRANSCRIBE_API struct transcribe_stream_params
    transcribe_stream_default_params(void);

/*
 * Begin a streaming run on ctx.
 *
 * run_params and stream_params MUST be non-null (callers obtain
 * defaults from transcribe_default_params() and
 * transcribe_stream_default_params()).
 *
 * On success the context transitions IDLE/FINISHED/FAILED -> ACTIVE
 * and every result-visible field is cleared: text, segments, words,
 * tokens, detected language, stream revision, committed counts,
 * audio cursors, timings, was_aborted, and last stream status. The
 * installed abort callback, the loaded model, and any reusable
 * stream buffers held by the family are preserved.
 *
 * Returns:
 *   TRANSCRIBE_ERR_INVALID_ARG       NULL arg, ctx in ACTIVE state,
 *                                    or out-of-range enum in run_params.
 *   TRANSCRIBE_ERR_NOT_IMPLEMENTED   Model has no streaming hooks or
 *                                    capabilities advertise
 *                                    supports_streaming == false.
 *                                    All three of stream_begin /
 *                                    stream_feed / stream_finalize must
 *                                    be wired for begin to succeed;
 *                                    stream_reset is optional.
 *   TRANSCRIBE_ERR_UNSUPPORTED_TASK  TRANSCRIBE_TASK_TRANSLATE; v1
 *                                    streaming is transcribe-only.
 *   TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS
 *                                    Requested granularity finer than
 *                                    the model's max_timestamp_kind.
 *   TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE
 *                                    run_params->language not in the
 *                                    model's declared language list.
 *
 * Failure semantics split at the family hook boundary:
 *
 *   Pre-hook failures (every status above) leave the context's
 *   lifecycle state untouched — the call returns without entering
 *   ACTIVE and without clearing the previous result snapshot.
 *
 *   Family-hook failures (any non-OK status returned by the family's
 *   stream_begin hook after the dispatcher has already cleared the
 *   snapshot and entered ACTIVE) transition the stream to FAILED and
 *   preserve the status in transcribe_stream_last_status. The result
 *   snapshot in this case is whatever the hook wrote before failing —
 *   typically empty.
 */
TRANSCRIBE_API transcribe_status transcribe_stream_begin(
    struct transcribe_context *              ctx,
    const struct transcribe_params *         run_params,
    const struct transcribe_stream_params *  stream_params);

/*
 * Feed PCM into the active stream. 16 kHz mono float32, same as
 * transcribe_run.
 *
 * pcm must be non-null and n_samples must be strictly greater than
 * zero. Polling the stream without supplying audio is unsupported —
 * use the stream accessors (transcribe_stream_revision,
 * transcribe_stream_n_committed_*, transcribe_stream_last_status,
 * transcribe_stream_get_state) to inspect state without progressing
 * the stream.
 *
 * update is nullable; when non-null the dispatcher zero-initializes
 * it before calling the family hook, so callers may rely on a clean
 * struct even on early-return error paths.
 *
 * Returns TRANSCRIBE_ERR_INVALID_ARG when ctx is NULL, when state
 * is not ACTIVE, or on malformed input. A terminal non-OK status
 * from the family hook transitions the stream to FAILED and is
 * preserved in transcribe_stream_last_status. TRANSCRIBE_ERR_ABORTED
 * is a terminal status; transcribe_was_aborted distinguishes it from
 * other failures.
 */
TRANSCRIBE_API transcribe_status transcribe_stream_feed(
    struct transcribe_context *        ctx,
    const float *                      pcm,
    int                                n_samples,
    struct transcribe_stream_update *  update);

/*
 * Signal end of input. Flushes buffered audio, satisfies right-
 * context / lookahead requirements, and emits remaining text.
 *
 * On success the context transitions ACTIVE -> FINISHED. On a
 * terminal non-OK status the context transitions to FAILED and the
 * status is preserved in transcribe_stream_last_status.
 *
 * Returns TRANSCRIBE_ERR_INVALID_ARG when ctx is NULL or state is
 * not ACTIVE.
 */
TRANSCRIBE_API transcribe_status transcribe_stream_finalize(
    struct transcribe_context *        ctx,
    struct transcribe_stream_update *  update);

/*
 * Abandon the current stream without finalizing.
 *
 * Always returns the context to IDLE and clears every result-visible
 * field, every stream-snapshot counter, last_status, and the family's
 * per-utterance streaming state. Allocated stream buffers held by the
 * family are preserved for reuse — full memory release is
 * transcribe_context_free.
 *
 * Reset from IDLE clears any stale result/snapshot from a previous
 * stream or run. Reset from FINISHED or FAILED clears the surviving
 * result text and snapshot counters as well.
 *
 * No-op if ctx is NULL.
 */
TRANSCRIBE_API void transcribe_stream_reset(
    struct transcribe_context * ctx);

/*
 * Current stream lifecycle state. Returns TRANSCRIBE_STREAM_IDLE if
 * ctx is NULL.
 */
TRANSCRIBE_API enum transcribe_stream_state
    transcribe_stream_get_state(const struct transcribe_context * ctx);

/*
 * Monotonic revision counter. Increments whenever the context result
 * changes during streaming. Reset to 0 by begin / reset / run.
 * Returns 0 if ctx is NULL.
 */
TRANSCRIBE_API int transcribe_stream_revision(
    const struct transcribe_context * ctx);

/*
 * Committed-prefix counts. tokens[0 .. n_committed_tokens) is the
 * stable prefix; everything beyond may be replaced on the next
 * feed/finalize. The same holds for words and segments. Families
 * that only emit on finalize set committed counts equal to the
 * total counts on the finalize call.
 *
 * Return 0 if ctx is NULL.
 */
TRANSCRIBE_API int transcribe_stream_n_committed_segments(
    const struct transcribe_context * ctx);
TRANSCRIBE_API int transcribe_stream_n_committed_words(
    const struct transcribe_context * ctx);
TRANSCRIBE_API int transcribe_stream_n_committed_tokens(
    const struct transcribe_context * ctx);

/*
 * Last terminal status of the stream. Preserves the failing status
 * after a feed/finalize call transitioned the stream to FAILED, so
 * the caller can inspect it after the fact. Reset to TRANSCRIBE_OK
 * by begin / reset / run.
 *
 * Returns TRANSCRIBE_OK if ctx is NULL or no terminal status has
 * been recorded on the current stream.
 */
TRANSCRIBE_API transcribe_status transcribe_stream_last_status(
    const struct transcribe_context * ctx);

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

/*
 * The language the model itself predicted on the most recent run, as a
 * short ISO code ("en", "zh", "yue", "ja", "ko", ...). Returns an empty
 * string when:
 *   - no successful run has happened on this context yet, or
 *   - the caller passed an explicit `params->language` hint (the
 *     library does not echo hints back through this field; callers
 *     already know what they asked for), or
 *   - the model does not support language detection (English-only
 *     Whispers, families without an LID head), or
 *   - the family's LID head produced a non-language sentinel for this
 *     audio (e.g. SenseVoice's <|nospeech|>).
 *
 * Returned pointer is owned by the context and remains valid until the
 * next transcribe_run() or transcribe_context_free() call.
 */
TRANSCRIBE_API const char *               transcribe_detected_language(const struct transcribe_context * ctx);

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
