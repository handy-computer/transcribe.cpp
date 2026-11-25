// transcribe-turn.h - internal TURN role surface: the session base, the
// per-arch ops table, and the shared turn segmenter.
//
// Clip models (frame_samples == 0) score one already cropped clip. Streaming
// models step whole frames and carry their own state; the role dispatcher
// (transcribe-turn.cpp) validates input, buffers partial frames, aligns the
// lagged speech track, flushes, and cuts a whole run into turns.

#pragma once

#include "transcribe-session-core.h"
#include "transcribe/turn.h"

#include <cstdint>
#include <vector>

namespace transcribe {

// transcribe_turn_params after defaulting and validation. Also the model's
// defaults (TurnModelInfo::defaults).
struct TurnPolicy {
    double  threshold                   = 0.5;
    double  speech_threshold            = 0.5;
    int32_t pre_roll_ms                 = 0;
    int32_t min_silence_before_check_ms = 0;
    int32_t max_silence_ms              = 0;
    int32_t min_speech_ms               = 0;
    int32_t min_confident_speech_ms     = 0;
    double  confident_speech_threshold  = 0.9;
};

// A model's static turn facts, read from its GGUF at load.
struct TurnModelInfo {
    int32_t    max_audio_samples       = 0;  // run crops to the last this many; 0 = no limit
    int32_t    frame_samples           = 0;  // 0 = clip model, no stream
    bool       has_speech              = false;
    int32_t    speech_lookahead_frames = 0;
    TurnPolicy defaults;  // threshold always; the segmenter fields when has_speech
};

struct TurnSegmentEntry {
    int64_t start_sample = 0;
    int64_t end_sample   = 0;
    int32_t end_reason   = TRANSCRIBE_TURN_END_MODEL;
};

// Validate params (NULL = defaults) against a TURN model: the threshold
// (negative = the model's), and a family extension, which the model's
// apply_run_ext validates and folds into the policy. Starts from the model's
// defaults. INVALID_ARG on a non-finite or out-of-range value or an
// extension the model does not accept; BAD_STRUCT_SIZE on a short struct.
transcribe_status resolve_turn_params(const transcribe_model *       model,
                                      const transcribe_turn_params * params,
                                      TurnPolicy &                   out);

// Inworld SAVA's reference TurnSegmenter (inworld.sava.segmenter, fed the way
// sava.segment feeds it), ported verbatim. speech[t] judges frame t of a
// clip of n_samples samples; turn_at_step[s] is the turn score the stream
// emitted at step s, for s in [0, n_frames + lag) (the last lag steps are
// the flush's silent steps). Kept turns only, in samples, ends clipped to
// n_samples. Replaces out.
void turn_frames_to_segments(const float *                   speech,
                             const float *                   turn_at_step,
                             int64_t                         n_frames,
                             int64_t                         n_samples,
                             int32_t                         frame_samples,
                             int32_t                         lag,
                             const TurnPolicy &              policy,
                             std::vector<TurnSegmentEntry> & out);

struct TurnOps {
    const TurnModelInfo & (*info)(const transcribe_model * model);
    transcribe_turn_session * (*new_session)();

    // Clip models (info.frame_samples == 0), else nullptr. Score validated,
    // already cropped PCM (n_samples > 0): write the turn probability.
    // Poll session->poll_abort() and return TRANSCRIBE_ERR_ABORTED when it
    // fires. Set session->t_mel_us / t_encode_us / t_decode_us.
    transcribe_status (*score_clip)(transcribe_turn_session * session, const float * pcm, int64_t n_samples, float * p);

    // Streaming models (info.frame_samples > 0), else nullptr.
    // reset: return the model state to the start of a stream. Also called
    // once on a new session.
    void (*reset)(transcribe_turn_session * session);
    // step: run n_frames whole frames, pcm[0, n_frames * frame_samples), as
    // the continuation of the stream. For each step write the on-time turn
    // score to turn[i] and, when has_speech, the speech score the model
    // emits at that step to speech[i] (it judges frame step -
    // speech_lookahead_frames; the dispatcher drops the ones before frame 0).
    // speech is nullptr when has_speech is false. Abort and timings as above.
    transcribe_status (
        *step)(transcribe_turn_session * session, const float * pcm, int64_t n_frames, float * turn, float * speech);
    // Optional (nullptr = the dispatcher steps the sessions one by one): step
    // one frame, frame_pcm[i][0, frame_samples), for each of count distinct
    // sessions of this model in one compute, using sessions[0]'s threads.
    // turn[i] / speech[i] as step writes them for sessions[i]. Must give
    // exactly the values step would. Poll every session's poll_abort().
    transcribe_status (*step_batch)(transcribe_turn_session * const * sessions,
                                    int                               count,
                                    const float * const *             frame_pcm,
                                    float *                           turn,
                                    float *                           speech);
    // Optional (nullptr = no extension accepted on TURN_RUN): validate a
    // family extension that transcribe_model_accepts_ext_kind accepted, and
    // apply it to policy. Must not touch anything else.
    transcribe_status (*apply_run_ext)(const transcribe_model * model, const transcribe_ext * ext, TurnPolicy & policy);
};

}  // namespace transcribe

struct transcribe_turn_session : transcribe::SessionCore {
    // Stream state owned by the dispatcher.
    std::vector<float> pending;              // buffered partial frame, < frame_samples
    int64_t            frames_done = 0;      // frames stepped since the stream began
    bool               needs_reset = false;  // retry a reset that threw before stepping again

    // Last call's result; empty / zero otherwise.
    float                                     p     = 0.0f;
    bool                                      ended = false;
    std::vector<float>                        probs;
    int64_t                                   first_frame = 0;
    std::vector<float>                        speech_probs;
    int64_t                                   speech_first_frame = 0;
    std::vector<transcribe::TurnSegmentEntry> segments;
};
