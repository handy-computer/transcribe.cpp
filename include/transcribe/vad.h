/*
 * include/transcribe/vad.h - VAD role: where is there speech.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. For models whose
 * product is voice activity (transcribe_model_roles() has TRANSCRIBE_ROLE_VAD).
 *
 * A VAD model scores fixed-size frames of 16 kHz audio with a speech
 * probability. Two ways to drive it from a transcribe_vad_session:
 *
 *   Offline   transcribe_vad_run scores a whole clip and turns the
 *             probabilities into speech segments.
 *   Streaming transcribe_vad_stream_feed scores audio as it arrives and
 *             returns the probabilities of the frames each call completed;
 *             transcribe_vad_stream_flush scores the zero-padded remainder.
 *             On CPU, Silero VAD's stream probabilities equal an offline run
 *             over the same audio with the same thread count.
 *
 * Usage guide: docs/vad.md. Threading and lifetime rules: docs/roles.md.
 */

#ifndef TRANSCRIBE_VAD_H
#define TRANSCRIBE_VAD_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

struct transcribe_vad_session;

/* Static facts about a VAD model. */
struct transcribe_vad_info {
    uint64_t struct_size;
    int32_t  sample_rate;   /* input PCM rate (16000) */
    int32_t  frame_samples; /* samples per probability (512 = 32 ms) */
};

struct transcribe_vad_session_params {
    uint64_t struct_size;
    int32_t  n_threads; /* 0 = library default */
};

/*
 * Probabilities -> speech segments (transcribe_vad_run only). The fields
 * and defaults are those of the reference implementation (Silero's
 * get_speech_timestamps); transcribe_vad_params_init sets the defaults.
 * The thresholds are double, as the reference's are, so a frame on a
 * threshold is classified the same way.
 */
struct transcribe_vad_params {
    uint64_t struct_size;
    /* A frame with p >= threshold is speech (default 0.5). [0, 1]. */
    double   threshold;
    /* Inside speech, a frame with p < neg_threshold is silence. Negative
     * (the default, -1) means max(threshold - 0.15, 0.01). Otherwise
     * [0, threshold]. */
    double   neg_threshold;
    /* Segments of at most this length are dropped (default 250), except
     * those emitted by max-duration splitting. */
    int32_t  min_speech_ms;
    /* Silence this long ends a segment (default 100). */
    int32_t  min_silence_ms;
    /* Each segment is extended by this much on both sides, or by half the
     * gap when two segments are closer than twice this (default 30). */
    int32_t  speech_pad_ms;
    /* Segments longer than this are split; 0 (the default) = no limit. A
     * split prefers a silence longer than min_silence_at_max_speech_ms. */
    int32_t  max_speech_ms;
    /* Shortest silence a max_speech_ms split may use (default 98). */
    int32_t  min_silence_at_max_speech_ms;
    /* At a max_speech_ms split, cut at the longest qualifying silence in
     * the segment (true, the default) or at the last one (false). */
    bool     use_max_possible_silence;
};

/* Summary of the last run, feed or flush. */
struct transcribe_vad_result {
    uint64_t struct_size;
    /* Probabilities produced by the call (transcribe_vad_probs). */
    int32_t  n_probs;
    /* Stream position of the first of them, in frames: 0 after
     * transcribe_vad_run; for a feed or flush, the number of frames the
     * stream had scored before the call. Frame f covers samples
     * [f * frame_samples, (f + 1) * frame_samples). */
    int64_t  first_frame;
    /* Speech segments (transcribe_vad_get_segment); transcribe_vad_run only,
     * 0 after a feed or flush. */
    int32_t  n_segments;
};

/* One speech segment, in samples of the run's input: [start, end). */
struct transcribe_vad_segment {
    uint64_t struct_size;
    int64_t  start_sample;
    int64_t  end_sample;
};

TRANSCRIBE_API void transcribe_vad_info_init(struct transcribe_vad_info * out);
TRANSCRIBE_API void transcribe_vad_session_params_init(struct transcribe_vad_session_params * params);
/* Zero-fills, then sets the reference defaults documented above. */
TRANSCRIBE_API void transcribe_vad_params_init(struct transcribe_vad_params * params);
TRANSCRIBE_API void transcribe_vad_result_init(struct transcribe_vad_result * out);
TRANSCRIBE_API void transcribe_vad_segment_init(struct transcribe_vad_segment * out);

/* UNSUPPORTED_ROLE when the model does not serve TRANSCRIBE_ROLE_VAD. */
TRANSCRIBE_API transcribe_status transcribe_vad_get_info(const struct transcribe_model * model,
                                                         struct transcribe_vad_info *    out);

/* params may be NULL for defaults. UNSUPPORTED_ROLE when the model does not
 * serve TRANSCRIBE_ROLE_VAD. On failure *out_session is NULL. */
TRANSCRIBE_API transcribe_status transcribe_vad_session_init(struct transcribe_model *                    model,
                                                             const struct transcribe_vad_session_params * params,
                                                             struct transcribe_vad_session **             out_session);

/* NULL is a no-op. */
TRANSCRIBE_API void transcribe_vad_session_free(struct transcribe_vad_session * session);

/* Polled during a run or feed; returning true stops it with
 * TRANSCRIBE_ERR_ABORTED. */
TRANSCRIBE_API void transcribe_vad_set_abort_callback(struct transcribe_vad_session * session,
                                                      transcribe_abort_callback       cb,
                                                      void *                          user_data);

/*
 * Score one clip and segment it: 16 kHz mono float32 PCM, every sample
 * finite. params may be NULL for defaults. The clip is scored from a fresh
 * model state in frames of frame_samples, the last one zero padded; segment
 * ends never pass n_samples. Resets any stream in progress. Replaces the
 * previous result. Malformed input (NULL pointers, n_samples <= 0, NaN /
 * Inf, out-of-range params) returns an error before the previous result or
 * the stream is touched.
 */
TRANSCRIBE_API transcribe_status transcribe_vad_run(struct transcribe_vad_session *      session,
                                                    const float *                        pcm,
                                                    int                                  n_samples,
                                                    const struct transcribe_vad_params * params);

/*
 * Append audio to the session's stream (n_samples may be 0) and score every
 * frame it completes; a partial frame waits for the next call. The result
 * holds those frames' probabilities. The stream continues across calls
 * until transcribe_vad_stream_reset, transcribe_vad_stream_flush or
 * transcribe_vad_run. Malformed input (NULL session, NULL pcm with
 * n_samples > 0, n_samples < 0, NaN / Inf) returns an error before the
 * stream or the previous result is touched. Any other failure (aborted,
 * backend error) resets the stream.
 */
TRANSCRIBE_API transcribe_status transcribe_vad_stream_feed(struct transcribe_vad_session * session,
                                                            const float *                   pcm,
                                                            int                             n_samples);

/*
 * End the stream: score the buffered partial frame, zero padded (the result
 * then holds one probability, or none when nothing was buffered), and
 * reset. The next feed starts a new stream at frame 0.
 */
TRANSCRIBE_API transcribe_status transcribe_vad_stream_flush(struct transcribe_vad_session * session);

/* Drop the stream (model state, buffered samples, frame count) without
 * scoring. The last result is kept. NULL is a no-op. */
TRANSCRIBE_API void transcribe_vad_stream_reset(struct transcribe_vad_session * session);

/*
 * The last call's summary. Zeroed before any call and after a call that
 * passed input validation but then failed (aborted, backend error).
 */
TRANSCRIBE_API transcribe_status transcribe_vad_get_result(const struct transcribe_vad_session * session,
                                                           struct transcribe_vad_result *        out);

/* The last call's n_probs probabilities, or NULL when there are none.
 * Session-owned; valid until the next run, feed, flush or session free. */
TRANSCRIBE_API const float * transcribe_vad_probs(const struct transcribe_vad_session * session);

/* Segment i of the last run, in time order. An out-of-range index returns
 * OK with a zeroed row. */
TRANSCRIBE_API transcribe_status transcribe_vad_get_segment(const struct transcribe_vad_session * session,
                                                            int                                   i,
                                                            struct transcribe_vad_segment *       out);

/* load_ms plus the last call's stage times (encode_ms: front end and
 * encoder; decode_ms: the recurrent decoder). */
TRANSCRIBE_API transcribe_status transcribe_vad_get_timings(const struct transcribe_vad_session * session,
                                                            struct transcribe_timings *           out);

/* Optional model-independent live policy: Silero VADIterator at 16 kHz.
 * Separate from the inference session; never resets a model on speech END.
 * One iterator may be used by one thread at a time. */
struct transcribe_vad_iterator;

struct transcribe_vad_iterator_params {
    uint64_t struct_size;
    /* Speech starts at p >= threshold; silence is p < threshold - 0.15.
     * Double comparisons, with no offline-style 0.01 floor. [0, 1]. */
    double   threshold;      /* default 0.5 */
    int32_t  min_silence_ms; /* default 100; >= 0 */
    int32_t  speech_pad_ms;  /* default 30; >= 0 */
};

typedef enum transcribe_vad_event_type {
    TRANSCRIBE_VAD_EVENT_START = 0,
    TRANSCRIBE_VAD_EVENT_END   = 1,
} transcribe_vad_event_type;

struct transcribe_vad_event {
    uint64_t                  struct_size;
    transcribe_vad_event_type type;
    /* Boundary in 16 kHz samples, not the time the event was detected.
     * START is clamped to 0; END includes padding and may exceed the
     * samples scored so far. Padding may overlap adjacent events. */
    int64_t                   sample;
};

struct transcribe_vad_iterator_result {
    uint64_t struct_size;
    int32_t  n_events;       /* events from the last feed, in detection order */
    int64_t  current_sample; /* number of probabilities consumed * frame_samples */
    bool     triggered;      /* speech is active (possibly awaiting silence) */
};

TRANSCRIBE_API void transcribe_vad_iterator_params_init(struct transcribe_vad_iterator_params * params);
TRANSCRIBE_API void transcribe_vad_event_init(struct transcribe_vad_event * out);
TRANSCRIBE_API void transcribe_vad_iterator_result_init(struct transcribe_vad_iterator_result * out);

/* frame_samples > 0, normally from transcribe_vad_get_info. params may be
 * NULL for defaults. Copies params; owns no model. On any failure *out is
 * NULL. The caller owns the iterator and must free it. */
TRANSCRIBE_API transcribe_status transcribe_vad_iterator_init(int32_t frame_samples,
                                                              const struct transcribe_vad_iterator_params * params,
                                                              struct transcribe_vad_iterator **             out);
TRANSCRIBE_API void              transcribe_vad_iterator_free(struct transcribe_vad_iterator * iterator);

/* Consume sequential per-frame probabilities, each finite and in [0, 1].
 * Arbitrary grouping, including n_probs == 0 (probs may then be NULL).
 * Every successful feed replaces the event result; an empty feed clears
 * events without advancing time. Any failure leaves state AND events intact.
 * Overflow of the sample counter (including padding headroom) is INVALID_ARG.
 * Middle probabilities neither cancel pending silence nor emit END; only
 * p >= threshold cancels it. No min/max speech-duration filtering.
 * EOF does NOT emit END, even for active speech, exactly as upstream. There
 * is no finish operation. A zero-padded final probability still advances a
 * full frame; callers wanting EOF closure/clamping must use the actual audio
 * length, not current_sample. Reset discards active speech without END. */
TRANSCRIBE_API transcribe_status transcribe_vad_iterator_feed(struct transcribe_vad_iterator * iterator,
                                                              const float *                    probs,
                                                              int32_t                          n_probs);

/* Reset time, hysteresis and last events. NULL is a no-op, as with free. */
TRANSCRIBE_API void              transcribe_vad_iterator_reset(struct transcribe_vad_iterator * iterator);
TRANSCRIBE_API transcribe_status transcribe_vad_iterator_get_result(const struct transcribe_vad_iterator *  iterator,
                                                                    struct transcribe_vad_iterator_result * out);
/* Copy event i; out-of-range indices return INVALID_ARG without writing out.
 * No borrowed event pointers: the copy remains valid after feed/reset/free. */
TRANSCRIBE_API transcribe_status transcribe_vad_iterator_get_event(const struct transcribe_vad_iterator * iterator,
                                                                   int32_t                                i,
                                                                   struct transcribe_vad_event *          out);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_VAD_H */
