// transcribe-turn.h - internal TURN role surface: the session base and the
// per-arch ops table.
//
// Clip models (frame_samples == 0) score one already cropped clip. Streaming
// models step whole frames and carry their own state; the role dispatcher
// (transcribe-turn.cpp) validates input, buffers partial frames, aligns the
// lagged speech track and flushes.

#pragma once

#include "transcribe-session-core.h"
#include "transcribe/turn.h"

#include <cstdint>
#include <vector>

namespace transcribe {

// A model's static turn facts, read from its GGUF at load.
struct TurnModelInfo {
    int32_t max_audio_samples       = 0;    // run crops to the last this many; 0 = no limit
    int32_t frame_samples           = 0;    // 0 = clip model, no stream
    bool    has_speech              = false;
    int32_t speech_lookahead_frames = 0;    // frames the speech scores lag the turn scores
    double  threshold               = 0.5;  // the model's recommended cut-off
};

// Validate params (NULL = defaults) against a TURN model and return the
// threshold to use (negative = the model's). INVALID_ARG on a non-finite or
// out-of-range value; BAD_STRUCT_SIZE on a short struct.
transcribe_status resolve_turn_threshold(const transcribe_model *       model,
                                         const transcribe_turn_params * params,
                                         double &                       out);

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
    // sessions of this model in one compute, using n_threads threads (the
    // dispatcher passes the caller's sessions[0]'s, which need not be listed
    // here). turn[i] / speech[i] as step writes them for sessions[i]. Must
    // give exactly the values step would, for any count. Poll every session's
    // poll_abort().
    transcribe_status (*step_batch)(transcribe_turn_session * const * sessions,
                                    int                               count,
                                    int                               n_threads,
                                    const float * const *             frame_pcm,
                                    float *                           turn,
                                    float *                           speech);
};

}  // namespace transcribe

struct transcribe_turn_session : transcribe::SessionCore {
    // Stream state owned by the dispatcher.
    std::vector<float> pending;              // buffered partial frame, < frame_samples
    int64_t            frames_done = 0;      // frames stepped since the stream began
    bool               needs_reset = false;  // retry a reset that threw before stepping again

    // Last call's result; empty / zero otherwise.
    float              p     = 0.0f;
    bool               ended = false;
    std::vector<float> probs;
    int64_t            first_frame = 0;
    std::vector<float> speech_probs;
    int64_t            speech_first_frame = 0;
};
