/*
 * include/transcribe/turn.h - TURN role: has the speaker finished their turn?
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. For models with
 * TRANSCRIBE_ROLE_TURN. Input is 16 kHz mono float32 PCM.
 *
 * transcribe_turn_run scores one clip: "has the turn ended at the end of this
 * audio?". Every TURN model supports it. Pass the current turn's audio (from
 * where the speaker started); the model pads or crops it itself.
 *
 * transcribe_turn_stream_feed scores live audio frame by frame. Only models
 * with transcribe_turn_info::frame_samples > 0 stream; the others return
 * TRANSCRIBE_ERR_NOT_IMPLEMENTED. A stream returns scores only; deciding when
 * a live turn ends is the caller's.
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

/* Static facts about a turn model. */
struct transcribe_turn_info {
    uint64_t struct_size;
    int32_t  sample_rate; /* input PCM rate (16000) */
    /* transcribe_turn_run scores the LAST max_audio_ms of longer input
     * (no error). 0 = no limit. Smart Turn: 8000. */
    int32_t  max_audio_ms;
    /* Samples per stream frame; 0 = the model cannot stream. SAVA: 640. */
    int32_t  frame_samples;
    /* The model also scores speech: the speech track and turn segments
     * exist. False: those are always empty. */
    bool     has_speech;
    /* Frames the speech track lags the audio: the speech score of frame t
     * is available once frame t + speech_lookahead_frames has been fed.
     * 0 when has_speech is false. SAVA: 3. */
    int32_t  speech_lookahead_frames;
    /* The model's recommended cut-off: the default of
     * transcribe_turn_params::threshold. */
    double   threshold;
};

struct transcribe_turn_session_params {
    uint64_t struct_size;
    int32_t  n_threads; /* 0 = library default */
};

/* Settings for transcribe_turn_run. Call transcribe_turn_params_init to set
 * the defaults. */
struct transcribe_turn_params {
    uint64_t                      struct_size;
    /* p >= threshold sets transcribe_turn_result::ended (and closes a turn
     * segment). Default: -1, which uses the model's
     * transcribe_turn_info::threshold. Any negative value uses the model's;
     * otherwise 0 to 1. */
    double                        threshold;
    /* Typed family extension on TRANSCRIBE_EXT_SLOT_TURN_RUN, or NULL; for
     * example the model's turn-segment rule (include/transcribe/sava.h). */
    const struct transcribe_ext * family;
};

/* Summary of the last run, feed or flush. */
struct transcribe_turn_result {
    uint64_t struct_size;
    /* Run only (0 / false after feed or flush): the score that the turn has
     * ended at the end of the clip, and p >= threshold. */
    float    p;
    bool     ended;
    /* Per-frame turn scores (transcribe_turn_probs): score i says the turn
     * has ended by the end of frame first_frame + i. Frame t covers samples
     * [t * frame_samples, (t + 1) * frame_samples). 0 for models that
     * cannot stream. After run, first_frame is 0 and every frame of the clip
     * is scored. */
    int32_t  n_probs;
    int64_t  first_frame;
    /* Per-frame speech scores (transcribe_turn_speech_probs): score i is the
     * speech probability of frame speech_first_frame + i. Only frames the
     * model has judged are returned, so this track trails the turn track by
     * speech_lookahead_frames. 0 when has_speech is false. */
    int32_t  n_speech_probs;
    int64_t  speech_first_frame;
    /* Turns found by the last run (transcribe_turn_get_segment); has_speech
     * models only. Always 0 after feed or flush. */
    int32_t  n_segments;
};

/* Why a turn segment ended. Append-only. */
typedef enum {
    TRANSCRIBE_TURN_END_MODEL   = 0, /* the turn score reached threshold */
    TRANSCRIBE_TURN_END_SILENCE = 1, /* the pause reached max_silence_ms */
    TRANSCRIBE_TURN_END_AUDIO   = 2, /* the audio ended with the turn open */
} transcribe_turn_end;

/* A turn in the input audio. Start is included; end is not. */
struct transcribe_turn_segment {
    uint64_t struct_size;
    int64_t  start_sample;
    int64_t  end_sample;
    int32_t  end_reason; /* transcribe_turn_end */
};

TRANSCRIBE_API void transcribe_turn_info_init(struct transcribe_turn_info * out);
TRANSCRIBE_API void transcribe_turn_session_params_init(struct transcribe_turn_session_params * params);
TRANSCRIBE_API void transcribe_turn_params_init(struct transcribe_turn_params * params);
TRANSCRIBE_API void transcribe_turn_result_init(struct transcribe_turn_result * out);
TRANSCRIBE_API void transcribe_turn_segment_init(struct transcribe_turn_segment * out);

/* UNSUPPORTED_ROLE when the model does not serve TRANSCRIBE_ROLE_TURN. */
TRANSCRIBE_API transcribe_status transcribe_turn_get_info(const struct transcribe_model * model,
                                                          struct transcribe_turn_info *   out);

/* params may be NULL for defaults. UNSUPPORTED_ROLE when the model does not
 * serve TRANSCRIBE_ROLE_TURN. On failure *out_session is NULL. */
TRANSCRIBE_API transcribe_status transcribe_turn_session_init(struct transcribe_model *                     model,
                                                              const struct transcribe_turn_session_params * params,
                                                              struct transcribe_turn_session ** out_session);

/* NULL is a no-op. */
TRANSCRIBE_API void transcribe_turn_session_free(struct transcribe_turn_session * session);

/* Polled during run, feed and flush; returning true stops the call with
 * TRANSCRIBE_ERR_ABORTED. */
TRANSCRIBE_API void transcribe_turn_set_abort_callback(struct transcribe_turn_session * session,
                                                       transcribe_abort_callback        cb,
                                                       void *                           user_data);

/*
 * Score one clip from a fresh state: does the speaker's turn end at the end
 * of this audio? Every sample finite; n_samples > 0. params may be NULL for
 * defaults. Input longer than max_audio_ms is scored on its last
 * max_audio_ms. Discards any live stream and replaces the last result.
 * Malformed input (NULL pointers, n_samples <= 0, NaN / Inf, out-of-range
 * settings, a bad extension) returns an error before the stream or the last
 * result is touched.
 */
TRANSCRIBE_API transcribe_status transcribe_turn_run(struct transcribe_turn_session *      session,
                                                     const float *                         pcm,
                                                     int                                   n_samples,
                                                     const struct transcribe_turn_params * params);

/*
 * Process live audio; keep an incomplete frame for the next call. Returns the
 * scores of the frames this call completes (and the speech scores of the
 * frames it lets the model judge). n_samples may be 0; pcm may be NULL then.
 * NOT_IMPLEMENTED when the model cannot stream. Invalid input leaves the
 * stream and last result unchanged; an abort or processing error resets the
 * stream and clears the result.
 */
TRANSCRIBE_API transcribe_status transcribe_turn_stream_feed(struct transcribe_turn_session * session,
                                                             const float *                    pcm,
                                                             int                              n_samples);

/* End the stream: pad the last partial frame with zeros, then let the model
 * judge every remaining frame. Returns the turn score of the padded frame
 * (if there was leftover audio) and every remaining speech score. The next
 * feed starts a new stream at frame 0. */
TRANSCRIBE_API transcribe_status transcribe_turn_stream_flush(struct transcribe_turn_session * session);

/*
 * Feed n live streams at once: sessions[i] gets pcm[i][0, n_samples[i]), as
 * if transcribe_turn_stream_feed(sessions[i], pcm[i], n_samples[i]) had been
 * called for each, in one compute. Each session's stream, result and result
 * pointers end exactly as after its own feed (bit for bit on the CPU),
 * whatever the other sessions are. Read each result from its session.
 *
 * For a service with several live users on one streaming model: a step reads
 * the weights once for all of them. The sessions must be distinct and belong
 * to the same model; n >= 1; n_samples[i] may be 0 (pcm[i] may then be NULL).
 * Uses sessions[0]'s thread count. Every listed session's abort callback is
 * polled. NOT_IMPLEMENTED when the model cannot stream. Invalid input (NULL
 * arrays or sessions, a repeated session, sessions of different models, a
 * bad length, NaN or infinity) leaves every stream and result unchanged. An
 * abort or processing error resets every listed stream and clears every
 * listed result. The call is one compute on the model (docs/roles.md).
 */
TRANSCRIBE_API transcribe_status transcribe_turn_stream_feed_batch(struct transcribe_turn_session * const * sessions,
                                                                   const float * const *                    pcm,
                                                                   const int *                              n_samples,
                                                                   int                                      n);

/* Discard the stream without processing it; keep the last result. NULL does
 * nothing. */
TRANSCRIBE_API void transcribe_turn_stream_reset(struct transcribe_turn_session * session);

/* The last run's, feed's or flush's summary. Empty before any processing and
 * after an abort or processing error. */
TRANSCRIBE_API transcribe_status transcribe_turn_get_result(const struct transcribe_turn_session * session,
                                                            struct transcribe_turn_result *        out);

/* The last call's per-frame scores, or NULL if there are none. Valid until
 * the next run, feed, flush or session free. Do not free. */
TRANSCRIBE_API const float * transcribe_turn_probs(const struct transcribe_turn_session * session);
TRANSCRIBE_API const float * transcribe_turn_speech_probs(const struct transcribe_turn_session * session);

/* Turn segment i of the last run, ordered by start. Segments follow the
 * model's reference rule exactly and may overlap (a turn's pre-roll can reach
 * back into the previous one). An out-of-range index
 * returns OK with a cleared output. */
TRANSCRIBE_API transcribe_status transcribe_turn_get_segment(const struct transcribe_turn_session * session,
                                                             int                                    i,
                                                             struct transcribe_turn_segment *       out);

/* load_ms plus the last call's stage times. */
TRANSCRIBE_API transcribe_status transcribe_turn_get_timings(const struct transcribe_turn_session * session,
                                                             struct transcribe_timings *            out);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_TURN_H */
