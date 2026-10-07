// transcribe-diarize.h - internal DIARIZE role surface: the session base,
// the per-arch ops table, and the shared probs -> segments step.
//
// Families compute frame probabilities; the role dispatcher
// (transcribe-diarize.cpp) turns them into speaker segments.

#pragma once

#include "transcribe-session-core.h"
#include "transcribe/diarize.h"

#include <vector>

struct transcribe_diarize_session : transcribe::SessionCore {
    std::vector<transcribe::SpeakerSegmentEntry> segments;  // last successful run's rows
};

namespace transcribe {

// One run's family output: probs is row-major [n_frames, n_speakers].
struct DiarizeProbs {
    std::vector<float> probs;
    int                n_frames   = 0;
    int                n_speakers = 0;
    double             frame_ms   = 0.0;
};

struct DiarizeOps {
    int (*max_speakers)(const transcribe_model * model);
    transcribe_diarize_session * (*new_session)();
    // Pure pre-clear check of params->family (already kind/size-checked);
    // NULL = nothing to check.
    transcribe_status (*run_validate)(const transcribe_diarize_params * params);
    transcribe_status (*run)(transcribe_diarize_session *      session,
                             const float *                     pcm,
                             int                               n_samples,
                             const transcribe_diarize_params * params,
                             DiarizeProbs &                    out);
};

// Append one segment per contiguous run of probs > 0.5 per speaker,
// speaker-major and time-ordered (speaker_id 1-based, p = NaN).
void probs_to_segments(const float *                      probs,
                       int                                n_frames,
                       int                                n_speakers,
                       double                             frame_ms,
                       std::vector<SpeakerSegmentEntry> & out);

}  // namespace transcribe
