/*
 * End-of-turn detection: has the speaker finished talking?
 *
 * Requires a model that supports TRANSCRIBE_ROLE_TURN. Works in C and C++.
 * Audio must be 16 kHz mono floats.
 *
 * A voice assistant uses this to decide when to answer: a pause alone is a
 * poor signal, as people pause mid-sentence. The model returns the
 * probability that the turn is over: 0 means the speaker will keep talking;
 * 1 means they are done.
 *
 * transcribe_turn_run scores one clip; every turn model supports it.
 * transcribe_turn_stream_feed scores incoming audio frame by frame, for
 * models that stream (transcribe_turn_info::frame_samples > 0). When to
 * answer is your code's decision.
 *
 * Usage: docs/turn.md. Threading and object lifetimes: docs/roles.md.
 */

#ifndef TRANSCRIBE_TURN_H
#define TRANSCRIBE_TURN_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

struct transcribe_turn_session;

/* What a turn model needs and supports. */
struct transcribe_turn_info {
    uint64_t struct_size; /* struct size in bytes; set by the init function */
    int32_t  sample_rate; /* required audio sample rate: 16000 Hz */
    /* Longest audio transcribe_turn_run looks at; longer audio is scored on
     * its last max_audio_ms. 0 means no limit. */
    int32_t  max_audio_ms;
    /* Samples per frame for streaming; 0 means the model cannot stream. */
    int32_t  frame_samples;
    /* True if the model also returns a speech probability for each frame
     * (transcribe_turn_speech_probs), so no separate VAD is needed. */
    bool     has_speech;
    /* The model's recommended cut-off: the default threshold. */
    double   threshold;
};

struct transcribe_turn_session_params {
    uint64_t struct_size; /* struct size in bytes; set by the init function */
    int32_t  n_threads;   /* number of processing threads; 0 uses the default */
};

/* Settings for transcribe_turn_run. Call transcribe_turn_params_init to set
 * the defaults. */
struct transcribe_turn_params {
    uint64_t struct_size; /* struct size in bytes; set by the init function */
    /* Probability at or above which the turn counts as ended. Default: -1;
     * any negative value uses transcribe_turn_info::threshold. Otherwise use
     * 0 to 1. */
    double   threshold;
};

/* Summary of the last run, feed or flush. */
struct transcribe_turn_result {
    uint64_t struct_size; /* struct size in bytes; set by the init function */
    /* After run: the probability that the turn has ended at the end of the
     * clip, and whether p reached the threshold. 0 and false after feed or
     * flush. */
    float    p;
    bool     ended;
    /* Number of per-frame turn probabilities (transcribe_turn_probs);
     * probability i is for the end of frame first_frame + i. Always 0 for
     * models that cannot stream. */
    int32_t  n_probs;
    /* Index of the first returned frame, counting from 0. Always 0 after
     * run. For feed or flush, counts frames processed before this call.
     * A frame starts at sample: frame index * frame_samples. */
    int64_t  first_frame;
    /* Number of per-frame speech probabilities
     * (transcribe_turn_speech_probs) and the index of their first frame.
     * Always 0 when has_speech is false.
     *
     * A model may judge speech a fixed number of frames late (lookahead;
     * SAVA: 3). In a stream, the speech probability of frame t then arrives
     * with the turn probability of frame t + lookahead; after a feed that
     * returns speech probabilities, lookahead =
     * (first_frame + n_probs) - (speech_first_frame + n_speech_probs).
     * Flush returns the speech probabilities still missing. After run,
     * every frame has both. */
    int32_t  n_speech_probs;
    int64_t  speech_first_frame;
};

/* Call each struct's init function before using it. */
TRANSCRIBE_API void transcribe_turn_info_init(struct transcribe_turn_info * out);
TRANSCRIBE_API void transcribe_turn_session_params_init(struct transcribe_turn_session_params * params);
/* Clear the struct and set the default settings documented above. */
TRANSCRIBE_API void transcribe_turn_params_init(struct transcribe_turn_params * params);
TRANSCRIBE_API void transcribe_turn_result_init(struct transcribe_turn_result * out);

/* Get what the model needs and supports.
 * Returns UNSUPPORTED_ROLE if the model cannot detect turn ends. */
TRANSCRIBE_API transcribe_status transcribe_turn_get_info(const struct transcribe_model * model,
                                                          struct transcribe_turn_info *   out);

/* Create a session for processing audio. NULL params uses defaults.
 * Returns UNSUPPORTED_ROLE if the model cannot detect turn ends.
 * On failure, *out_session is NULL. */
TRANSCRIBE_API transcribe_status transcribe_turn_session_init(struct transcribe_model *                     model,
                                                              const struct transcribe_turn_session_params * params,
                                                              struct transcribe_turn_session ** out_session);

/* Free the session. NULL does nothing. */
TRANSCRIBE_API void transcribe_turn_session_free(struct transcribe_turn_session * session);

/* Check this callback during run, feed, feed_batch or flush. Return true to
 * stop processing with TRANSCRIBE_ERR_ABORTED. */
TRANSCRIBE_API void transcribe_turn_set_abort_callback(struct transcribe_turn_session * session,
                                                       transcribe_abort_callback        cb,
                                                       void *                           user_data);

/*
 * Score one clip: has the turn ended at the end of this audio? Pass the
 * current turn's audio, from where the speaker started, with no NaN or
 * infinite values; n_samples > 0. NULL params uses defaults.
 *
 * Discards any live stream and replaces the last result. Audio longer than
 * max_audio_ms is scored on its last max_audio_ms. Models that can stream
 * also return a probability for every frame.
 *
 * Invalid input (NULL session or pcm, invalid n_samples, NaN, infinity or
 * out-of-range settings) leaves the stream and last result unchanged.
 */
TRANSCRIBE_API transcribe_status transcribe_turn_run(struct transcribe_turn_session *      session,
                                                     const float *                         pcm,
                                                     int                                   n_samples,
                                                     const struct transcribe_turn_params * params);

/*
 * Process incoming audio with no NaN or infinite values, keeping an
 * incomplete frame for the next call. n_samples may be 0; pcm may be NULL
 * then. Returns a turn probability for each frame completed during this
 * call and a speech probability for each frame the model judged.
 * Returns NOT_IMPLEMENTED if the model cannot stream.
 *
 * The stream continues until stream_reset, stream_flush or run.
 * Invalid input (NULL session, NULL pcm with n_samples > 0, negative
 * n_samples, NaN or infinity) leaves the stream and last result unchanged.
 * An abort or processing error resets the stream and clears the result.
 */
TRANSCRIBE_API transcribe_status transcribe_turn_stream_feed(struct transcribe_turn_session * session,
                                                             const float *                    pcm,
                                                             int                              n_samples);

/*
 * Feed several streams in one call: when many users talk to one model, it
 * processes them together, much faster than one feed each. Each session
 * ends up exactly as after transcribe_turn_stream_feed(sessions[i], pcm[i],
 * n_samples[i]); read each result from its own session. Uses the thread
 * count of sessions[0] and checks every session's abort callback. Returns
 * NOT_IMPLEMENTED if the model cannot stream.
 *
 * n >= 1; the sessions must be distinct and of one model. Invalid input
 * (as for feed, or NULL arrays, a repeated session, mixed models) leaves
 * every stream and result unchanged. An abort or processing error resets
 * every listed stream and clears every result.
 */
TRANSCRIBE_API transcribe_status transcribe_turn_stream_feed_batch(struct transcribe_turn_session * const * sessions,
                                                                   const float * const *                    pcm,
                                                                   const int *                              n_samples,
                                                                   int                                      n);

/* Process leftover audio, zero padding the last frame, then end the stream.
 * Returns that frame's turn probability (if audio was left over) and the
 * speech probabilities of every frame not yet judged. The next feed starts
 * a new stream at frame 0. */
TRANSCRIBE_API transcribe_status transcribe_turn_stream_flush(struct transcribe_turn_session * session);

/* Discard the stream and leftover audio without processing them; the next
 * feed starts at frame 0. Keeps the last result. NULL does nothing. */
TRANSCRIBE_API void transcribe_turn_stream_reset(struct transcribe_turn_session * session);

/* Get the summary of the last run, feed or flush.
 * Empty before any processing, and after an abort or processing error.
 * Invalid input leaves the previous result unchanged. */
TRANSCRIBE_API transcribe_status transcribe_turn_get_result(const struct transcribe_turn_session * session,
                                                            struct transcribe_turn_result *        out);

/* Get the last call's n_probs turn probabilities or n_speech_probs speech
 * probabilities, or NULL if there are none. Do not free these pointers.
 * They are valid until the next run, feed, flush or session free. */
TRANSCRIBE_API const float * transcribe_turn_probs(const struct transcribe_turn_session * session);
TRANSCRIBE_API const float * transcribe_turn_speech_probs(const struct transcribe_turn_session * session);

/* Get model load and last-call processing times in milliseconds. load_ms
 * is the load time; the others cover the last run, feed or flush (0 after
 * a failed one):
 *   mel_ms     a separate audio front end (Smart Turn's log-mel); else 0.
 *   encode_ms  the model's compute (SAVA: its front end included). After
 *              feed_batch, the time of the batched computes it took part in.
 *   decode_ms  0. */
TRANSCRIBE_API transcribe_status transcribe_turn_get_timings(const struct transcribe_turn_session * session,
                                                             struct transcribe_timings *            out);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_TURN_H */
