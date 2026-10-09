// transcribe-vad.h - internal VAD role surface: the session base, the
// per-arch ops table, and the shared probabilities -> segments step.
//
// Families score whole frames and carry their own recurrent state; the role
// dispatcher (transcribe-vad.cpp) validates input, buffers partial frames for
// streaming, zero pads the last frame of an offline run, and turns
// probabilities into speech segments.

#pragma once

#include "transcribe-session-core.h"
#include "transcribe/vad.h"

#include <cstdint>
#include <vector>

namespace transcribe {

struct VadSegmentEntry {
    int64_t start_sample = 0;
    int64_t end_sample   = 0;
};

// transcribe_vad_params after defaulting and validation.
struct VadSegmentParams {
    double  threshold                    = 0.5;
    double  neg_threshold                = 0.35;
    int32_t min_speech_ms                = 250;
    int32_t min_silence_ms               = 100;
    int32_t speech_pad_ms                = 30;
    int32_t max_speech_ms                = 0;  // 0 = no limit
    int32_t min_silence_at_max_speech_ms = 98;
    bool    use_max_possible_silence     = true;
};

// Validate params (NULL = defaults) and resolve them. INVALID_ARG on a
// non-finite or out-of-range field; BAD_STRUCT_SIZE on a short struct.
transcribe_status resolve_vad_params(const transcribe_vad_params * params, VadSegmentParams & out);

// Silero's get_speech_timestamps_from_probs, ported verbatim: probs[i]
// scores samples [i * frame_samples, (i + 1) * frame_samples) of a clip of
// n_samples samples at 16 kHz. Replaces out.
void vad_probs_to_segments(const float *                  probs,
                           int64_t                        n_probs,
                           int64_t                        n_samples,
                           int32_t                        frame_samples,
                           const VadSegmentParams &       params,
                           std::vector<VadSegmentEntry> & out);

struct VadOps {
    int32_t (*frame_samples)(const transcribe_model * model);
    transcribe_vad_session * (*new_session)();
    // Return the model state (recurrent state, audio context) to the start of
    // a stream. Also called once on a new session.
    void (*reset)(transcribe_vad_session * session);
    // Score n_frames whole frames, pcm[0, n_frames * frame_samples), as the
    // continuation of the current stream; write one probability per frame.
    // Poll session->poll_abort() and return TRANSCRIBE_ERR_ABORTED when it
    // fires. Add to session->t_encode_us / t_decode_us.
    transcribe_status (*score)(transcribe_vad_session * session, const float * pcm, int64_t n_frames, float * probs);
};

}  // namespace transcribe

struct transcribe_vad_session : transcribe::SessionCore {
    // Stream state owned by the dispatcher.
    std::vector<float> pending;              // buffered partial frame, < frame_samples
    int64_t            frames_done = 0;      // frames scored since the stream began
    bool               needs_reset = false;  // retry a reset that threw before scoring again

    // Last call's result; empty / zero otherwise.
    std::vector<float>                       probs;
    int64_t                                  first_frame = 0;
    std::vector<transcribe::VadSegmentEntry> segments;
};
