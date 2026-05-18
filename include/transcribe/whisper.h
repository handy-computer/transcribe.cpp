/*
 * include/transcribe/whisper.h - Whisper-family public extension surface.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Whisper-specific
 * run params still live in transcribe.h pending the Phase-2 run-params
 * migration; this header currently exposes only the family telemetry
 * (chunk traces) which moved out of transcribe.h in the Phase-4 output-API
 * conversion.
 */

#ifndef TRANSCRIBE_WHISPER_H
#define TRANSCRIBE_WHISPER_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 * segment boundary. Segments carried in transcribe_n_segments() /
 * transcribe_segment_*() live inside these windows.
 *
 * Output struct: caller initializes via TRANSCRIBE_WHISPER_CHUNK_TRACE_INIT
 * (zero-fill). The library writes only fields that fit within struct_size
 * and never touches tail bytes beyond it; every field is designed so the
 * zero value means "absent / unknown / false."
 */
struct transcribe_whisper_chunk_trace {
    size_t   struct_size;
    int64_t  t0_ms;
    int64_t  t1_ms;
    float    temperature_used;    /* the tier that was accepted */
    float    compression_ratio;
    float    avg_logprob;
    float    no_speech_prob;
    bool     no_speech_triggered; /* chunk output was discarded */
    int32_t  n_fallbacks;         /* tiers tried before accept (0 = tier 0 accepted) */
};

#define TRANSCRIBE_WHISPER_CHUNK_TRACE_INIT \
    { sizeof(struct transcribe_whisper_chunk_trace) }

/*
 * Number of chunk traces captured on the most recent successful run.
 * Returns 0 before any run or on non-Whisper contexts.
 */
TRANSCRIBE_API int transcribe_get_whisper_chunk_count(
    const struct transcribe_context * ctx);

/*
 * Read the trace at index [0, count) into caller-owned storage.
 * Out-of-range indices and non-Whisper contexts succeed and leave the
 * caller's struct as initialized (zero-filled).
 *
 * Returns:
 *   TRANSCRIBE_ERR_INVALID_ARG     out_trace is NULL.
 *   TRANSCRIBE_ERR_BAD_STRUCT_SIZE out_trace->struct_size is 0 or
 *                                  smaller than the library's minimum.
 *   TRANSCRIBE_OK                  otherwise. The caller's struct is
 *                                  written when ctx is a Whisper context
 *                                  and i is in range; otherwise it is
 *                                  left as zero-initialized by INIT.
 */
TRANSCRIBE_API transcribe_status transcribe_get_whisper_chunk_trace(
    const struct transcribe_context *       ctx,
    int                                     i,
    struct transcribe_whisper_chunk_trace * out_trace);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRANSCRIBE_WHISPER_H */
