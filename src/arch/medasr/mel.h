// arch/medasr/mel.h - MedASR (LASR) log-mel feature extractor.
//
// Mirrors transformers.models.lasr.feature_extraction_lasr.LasrFeatureExtractor:
//   - 16 kHz mono, 128 HTK-formula mel bins (kaldi formula in float64),
//     n_fft=512, win_length=400, hop_length=160.
//   - center=False (no symmetric STFT padding): output frame n covers
//     samples [n*hop, n*hop + win_length).
//   - Manual unfold + rfft, NOT torch.stft (zero-pads each 400-sample
//     window to 512 before fft).
//   - Symmetric Hann window (torch.hann_window(periodic=False)).
//   - NO pre-emphasis, NO dither, NO per-feature CMVN.
//   - Mel filterbank: kaldi formula in float64, HTK-style DC bin
//     excluded (first row zero), 125-7500 Hz, 128 bands, unnormalized
//     triangular filters.
//   - Log compression: y = log(clamp(power, min=1e-5)) (natural log).
//
// The window and filterbank are baked into the GGUF by convert-medasr.py
// (`frontend.window` and `frontend.mel_filterbank` tensors) — see the
// gigaam frontend's "use the baked bytes, do not recompute" reasoning.

#pragma once

#include "transcribe.h"

#include <cstddef>
#include <vector>

namespace transcribe::medasr {

struct MedAsrHParams;
struct MedAsrWeights;

struct MedAsrMelFrontend {
    int n_freq      = 0;   // n_fft / 2 + 1
    int n_mels      = 0;
    int n_fft       = 0;
    int win_length  = 0;
    int hop_length  = 0;
    float log_clamp_min = 1e-5f;
    std::vector<float> hann;   // [win_length] symmetric
    std::vector<float> mel_fb; // [n_mels, n_freq] row-major HTK

    void init(const MedAsrHParams & hp, const MedAsrWeights & w);

    // n_frames under center=False: floor((n_samples - win_length) / hop) + 1.
    int n_frames_for(size_t n_samples) const;

    // Compute log-mel. Output is row-major [n_mels, n_frames].
    transcribe_status compute(const float *        pcm,
                              size_t               n_samples,
                              std::vector<float> & out_mel,
                              int &                out_n_frames) const;
};

} // namespace transcribe::medasr
