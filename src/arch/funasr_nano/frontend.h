// arch/funasr_nano/frontend.h - kaldi-style HTK mel fbank + LFR stack
// (no CMVN), host-side, single-utterance.
//
// Pruned fork of arch/sensevoice/frontend.h. Identical kaldi fbank +
// LFR chain; the trailing per-feature CMVN is dropped because Fun-ASR-Nano
// trains on raw LFR features (`apply_cmvn=false` in the model config).
// CLAUDE.md policy #8: "No refactoring during a port — note awkwardness,
// ship correctness first." A future cleanup will refactor SAN-M block +
// KaldiFbankFrontend into a shared module.

#pragma once

#include "weights.h"

#include <cstddef>
#include <vector>

namespace transcribe::funasr_nano {

class KaldiFbankFrontend {
public:
    explicit KaldiFbankFrontend(const FunAsrNanoHParams & hp);

    // Compute LFR'd features from raw [-1, 1] f32 PCM. Output is row-major
    // [T_lfr, d_input] stored in `out_features`. Returns the number of LFR
    // frames; 0 if the input is too short.
    int compute(const float *        pcm,
                size_t               n_samples,
                std::vector<float> & out_features) const;

    int d_input() const { return d_input_; }
    int n_mels()  const { return n_mels_; }
    int lfr_m()   const { return lfr_m_; }
    int lfr_n()   const { return lfr_n_; }

private:
    int n_mels_      = 0;
    int sample_rate_ = 0;
    int win_length_  = 0;
    int hop_length_  = 0;
    int padded_n_fft_ = 0;
    int lfr_m_       = 0;
    int lfr_n_       = 0;
    int d_input_     = 0;
    bool upscale_samples_ = true;

    std::vector<float> mel_fb_;
    std::vector<float> window_;
};

} // namespace transcribe::funasr_nano
