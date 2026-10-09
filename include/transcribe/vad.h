/*
 * Voice activity detection (VAD): find speech in audio.
 *
 * Requires a model that supports TRANSCRIBE_ROLE_VAD. Works in C and C++.
 * Audio must be 16 kHz mono floats.
 *
 * The model checks short blocks of audio ("frames") and returns a speech
 * probability for each: 0 means no speech; 1 means speech.
 *
 * transcribe_vad_run finds speech segments in a whole clip.
 * transcribe_vad_stream_feed returns probabilities for incoming audio.
 * transcribe_vad_stream_flush processes any leftover audio.
 * transcribe_vad_iterator turns those probabilities into START/END events.
 *
 * Usage: docs/vad.md. Threading and object lifetimes: docs/roles.md.
 */

#ifndef TRANSCRIBE_VAD_H
#define TRANSCRIBE_VAD_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

struct transcribe_vad_session;

/* Audio requirements of a VAD model. */
struct transcribe_vad_info {
    uint64_t struct_size;   /* struct size in bytes; set by the init function */
    int32_t  sample_rate;   /* required audio sample rate: 16000 Hz */
    int32_t  frame_samples; /* samples per frame (512 samples = 32 ms) */
};

struct transcribe_vad_session_params {
    uint64_t struct_size; /* struct size in bytes; set by the init function */
    int32_t  n_threads;   /* number of processing threads; 0 uses the default */
};

/* Settings for transcribe_vad_run. Durations are in milliseconds and
 * must be nonnegative. Call transcribe_vad_params_init to set the defaults. */
struct transcribe_vad_params {
    uint64_t struct_size; /* struct size in bytes; set by the init function */
    /* Probability at or above which audio is classified as speech.
     * Default: 0.5. Range: 0 to 1. */
    double   threshold;
    /* Probability below which ongoing speech is treated as silence.
     * Default: -1, which uses max(threshold - 0.15, 0.01).
     * Any negative value uses this default; otherwise use 0 to threshold. */
    double   neg_threshold;
    /* Discard speech segments this short or shorter. Default: 250 ms.
     * Does not apply to segments split by max_speech_ms. */
    int32_t  min_speech_ms;
    /* How long to wait after silence starts before ending speech.
     * Default: 100 ms. */
    int32_t  min_silence_ms;
    /* Extra audio to keep before and after speech. Default: 30 ms.
     * Nearby segments share the gap so they do not overlap. */
    int32_t  speech_pad_ms;
    /* Maximum speech segment length. Longer segments are split,
     * at the longest pause when possible. Default: 0 (no limit). */
    int32_t  max_speech_ms;
};

/* Summary of the last run, feed or flush. */
struct transcribe_vad_result {
    uint64_t struct_size; /* struct size in bytes; set by the init function */
    /* Number of probabilities from the last call (transcribe_vad_probs). */
    int32_t  n_probs;
    /* Index of the first returned frame, counting from 0. Always 0 after
     * run. For feed or flush, counts frames processed before this call.
     * A frame starts at sample: frame index * frame_samples. */
    int64_t  first_frame;
    /* Number of speech segments (transcribe_vad_get_segment).
     * Always 0 after feed or flush. */
    int32_t  n_segments;
};

/* Speech position in the input audio. Start is included; end is not. */
struct transcribe_vad_segment {
    uint64_t struct_size;  /* struct size in bytes; set by the init function */
    int64_t  start_sample; /* first sample in the speech segment */
    int64_t  end_sample;   /* first sample after the speech segment */
};

/* Call each struct's init function before using it. */
TRANSCRIBE_API void transcribe_vad_info_init(struct transcribe_vad_info * out);
TRANSCRIBE_API void transcribe_vad_session_params_init(struct transcribe_vad_session_params * params);
/* Clear the struct and set the default settings documented above. */
TRANSCRIBE_API void transcribe_vad_params_init(struct transcribe_vad_params * params);
TRANSCRIBE_API void transcribe_vad_result_init(struct transcribe_vad_result * out);
TRANSCRIBE_API void transcribe_vad_segment_init(struct transcribe_vad_segment * out);

/* Get the model's required sample rate and frame size.
 * Returns UNSUPPORTED_ROLE if the model cannot detect speech. */
TRANSCRIBE_API transcribe_status transcribe_vad_get_info(const struct transcribe_model * model,
                                                         struct transcribe_vad_info *    out);

/* Create a session for processing audio. NULL params uses defaults.
 * Returns UNSUPPORTED_ROLE if the model cannot detect speech.
 * On failure, *out_session is NULL. */
TRANSCRIBE_API transcribe_status transcribe_vad_session_init(struct transcribe_model *                    model,
                                                             const struct transcribe_vad_session_params * params,
                                                             struct transcribe_vad_session **             out_session);

/* Free the session. NULL does nothing. */
TRANSCRIBE_API void transcribe_vad_session_free(struct transcribe_vad_session * session);

/* Check this callback during run or feed. Return true to stop processing
 * with TRANSCRIBE_ERR_ABORTED. */
TRANSCRIBE_API void transcribe_vad_set_abort_callback(struct transcribe_vad_session * session,
                                                      transcribe_abort_callback       cb,
                                                      void *                          user_data);

/*
 * Find speech segments in a whole audio clip.
 * Input must be 16 kHz mono floats with no NaN or infinite values.
 * NULL params uses defaults. n_samples must be greater than 0.
 *
 * Starts fresh, discards any live stream and replaces the last result.
 * Fills the last frame with zeros if needed. Speech segments never
 * extend past the input audio.
 *
 * Invalid input (NULL session or pcm, invalid n_samples, NaN, infinity
 * or out-of-range settings) leaves the stream and last result unchanged.
 */
TRANSCRIBE_API transcribe_status transcribe_vad_run(struct transcribe_vad_session *      session,
                                                    const float *                        pcm,
                                                    int                                  n_samples,
                                                    const struct transcribe_vad_params * params);

/*
 * Process incoming audio. Keep incomplete frames for the next call.
 * Input must be 16 kHz mono floats with no NaN or infinite values.
 * Returns a probability for each completed frame through transcribe_vad_probs.
 * n_samples may be 0; pcm may be NULL in that case.
 *
 * The stream continues until stream_reset, stream_flush or run.
 * Invalid input (NULL session, NULL pcm with n_samples > 0, negative
 * n_samples, NaN or infinity) leaves the stream and last result unchanged.
 * An abort or processing error resets the stream and clears the result.
 */
TRANSCRIBE_API transcribe_status transcribe_vad_stream_feed(struct transcribe_vad_session * session,
                                                            const float *                   pcm,
                                                            int                             n_samples);

/* Process leftover audio, filling the last frame with zeros, then end
 * the stream. Returns one probability, or none if no audio was left over.
 * The next feed starts a new stream at frame 0. */
TRANSCRIBE_API transcribe_status transcribe_vad_stream_flush(struct transcribe_vad_session * session);

/* Discard the stream and leftover audio without processing them.
 * Reset the model state and frame count. Keep the last result.
 * NULL does nothing. */
TRANSCRIBE_API void transcribe_vad_stream_reset(struct transcribe_vad_session * session);

/* Get the summary of the last run, feed or flush.
 * Empty before any processing, and after an abort or processing error.
 * Invalid input leaves the previous result unchanged. */
TRANSCRIBE_API transcribe_status transcribe_vad_get_result(const struct transcribe_vad_session * session,
                                                           struct transcribe_vad_result *        out);

/* Get the last call's n_probs probabilities, or NULL if there are none.
 * Do not free this pointer. It is valid until the next run, feed, flush
 * or session free. */
TRANSCRIBE_API const float * transcribe_vad_probs(const struct transcribe_vad_session * session);

/* Get speech segment i from the last run, ordered by start time.
 * An out-of-range index returns OK with a cleared output. */
TRANSCRIBE_API transcribe_status transcribe_vad_get_segment(const struct transcribe_vad_session * session,
                                                            int                                   i,
                                                            struct transcribe_vad_segment *       out);

/* Get model loading and last-call processing times in milliseconds.
 * load_ms: model loading; encode_ms: audio preparation and encoder;
 * decode_ms: recurrent decoder. */
TRANSCRIBE_API transcribe_status transcribe_vad_get_timings(const struct transcribe_vad_session * session,
                                                            struct transcribe_timings *           out);

/* Turn speech probabilities into live START and END events.
 * Uses Silero's VADIterator rules for 16 kHz audio.
 * Does not own or reset a model. Use from one thread at a time. */
struct transcribe_vad_iterator;

struct transcribe_vad_iterator_params {
    uint64_t struct_size; /* struct size in bytes; set by the init function */
    /* Probability at or above which audio is classified as speech.
     * Default: 0.5. Range: 0 to 1. */
    double   threshold;
    /* Probability below which ongoing speech is treated as silence.
     * Default: -1, which uses threshold - 0.15 (unlike run, no 0.01 minimum).
     * Any negative value uses this default; otherwise use 0 to threshold. */
    double   neg_threshold;
    /* How long to wait after silence starts before sending END.
     * Default: 100 ms. Must be nonnegative. */
    int32_t  min_silence_ms;
    /* Extra audio to include before START and after END.
     * Default: 30 ms. Must be nonnegative. */
    int32_t  speech_pad_ms;
};

typedef enum transcribe_vad_event_type {
    TRANSCRIBE_VAD_EVENT_START = 0, /* speech started */
    TRANSCRIBE_VAD_EVENT_END   = 1, /* speech ended */
} transcribe_vad_event_type;

struct transcribe_vad_event {
    uint64_t                  struct_size; /* struct size in bytes; set by the init function */
    transcribe_vad_event_type type;        /* START or END */
    /* Sample index where speech starts or ends in 16 kHz audio, including
     * padding. This is the speech position, not when the event was reported.
     * START cannot be below 0. END can extend past the processed audio.
     * Padding from nearby events may overlap. */
    int64_t                   sample;
};

struct transcribe_vad_iterator_result {
    uint64_t struct_size; /* struct size in bytes; set by the init function */
    int32_t  n_events;    /* number of events from the last feed, in the order detected */
    /* Audio samples covered so far. Each probability counts as one full frame. */
    int64_t  current_sample;
    /* True while speech is active, including while waiting for silence to end it. */
    bool     triggered;
};

/* Call each struct's init function before using it. */
TRANSCRIBE_API void transcribe_vad_iterator_params_init(struct transcribe_vad_iterator_params * params);
TRANSCRIBE_API void transcribe_vad_event_init(struct transcribe_vad_event * out);
TRANSCRIBE_API void transcribe_vad_iterator_result_init(struct transcribe_vad_iterator_result * out);

/* Create an iterator. Use the model's frame size from transcribe_vad_get_info;
 * frame_samples must be greater than 0. NULL params uses defaults.
 * Copies the settings and does not own a model. On failure, *out is NULL.
 * Free the iterator with transcribe_vad_iterator_free when done. */
TRANSCRIBE_API transcribe_status transcribe_vad_iterator_init(int32_t frame_samples,
                                                              const struct transcribe_vad_iterator_params * params,
                                                              struct transcribe_vad_iterator **             out);
/* Free the iterator. NULL does nothing. */
TRANSCRIBE_API void              transcribe_vad_iterator_free(struct transcribe_vad_iterator * iterator);

/*
 * Process speech probabilities in audio order and produce START/END events.
 * Feed any number at once. Each must be finite and between 0 and 1.
 *
 * Each successful feed replaces the previous events.
 * Feeding 0 probabilities clears events without advancing time;
 * probs may be NULL. Errors leave state and events unchanged.
 * A sample position too large to represent, including padding, returns
 * INVALID_ARG.
 *
 * Once silence starts, a probability at or above threshold cancels the
 * pending END. Values between the two thresholds do not cancel it
 * or produce END.
 *
 * There is no minimum or maximum speech length.
 * The end of the audio does not automatically produce END; there is no
 * finish function. Close unfinished speech yourself using the actual
 * audio length, not current_sample. The final frame counts as a full
 * frame, even if filled with zeros.
 * Reset discards active speech without producing END.
 */
TRANSCRIBE_API transcribe_status transcribe_vad_iterator_feed(struct transcribe_vad_iterator * iterator,
                                                              const float *                    probs,
                                                              int32_t                          n_probs);

/* Clear speech state and events, and start again at sample 0.
 * NULL does nothing. */
TRANSCRIBE_API void              transcribe_vad_iterator_reset(struct transcribe_vad_iterator * iterator);
/* Get the event count, audio position and whether speech is active. */
TRANSCRIBE_API transcribe_status transcribe_vad_iterator_get_result(const struct transcribe_vad_iterator *  iterator,
                                                                    struct transcribe_vad_iterator_result * out);
/* Copy event i from the last feed. Out-of-range indices return INVALID_ARG
 * without changing out. The copy remains valid after feed, reset or free. */
TRANSCRIBE_API transcribe_status transcribe_vad_iterator_get_event(const struct transcribe_vad_iterator * iterator,
                                                                   int32_t                                i,
                                                                   struct transcribe_vad_event *          out);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_VAD_H */
