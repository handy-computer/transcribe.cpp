// arch/gigaam/mel.h - GigaAM log-mel feature extractor.
//
// Mirrors torchaudio.transforms.MelSpectrogram(...) + log(clamp(x, 1e-9))
// as configured by gigaam.preprocess.FeatureExtractor: 16 kHz mono, 64
// HTK mel bins, n_fft = win_length = 320, hop = 160, center=False, no
// pre-emphasis, periodic Hann window, power=2.

#pragma once

#include "transcribe.h"

#include <cstddef>
#include <vector>

namespace transcribe::gigaam {

struct GigaamHParams;
struct GigaamWeights;

// Load the baked HTK mel filterbank + periodic Hann window from the
// GGUF at load time. hp drives the scalar shape parameters.
//
// The filterbank and window are emitted by the converter from the
// upstream torchaudio state_dict (`preprocessor.featurizer.0.*`), so
// the C++ MelFrontend uses bit-identical buffers — recomputing them
// from htk_hz_to_mel / Hann formulas drifts at high-freq bins because
// torchaudio uses Float32 throughout while a C++ recomputation in
// fp64 produces slightly different bin boundaries.
struct GigaamMelFrontend {
    int n_freq;                       // n_fft/2 + 1
    int n_mels;
    int n_fft;
    int win_length;
    int hop_length;
    std::vector<float> hann;          // [win_length] periodic
    std::vector<float> mel_fb;        // [n_mels, n_freq] row-major HTK

    void init(const GigaamHParams & hp, const GigaamWeights & w);

    // n_frames for a given audio length under center=False.
    // = floor((n_samples - win_length) / hop) + 1, or 0 if too short.
    int n_frames_for(size_t n_samples) const;

    // Compute log-mel. Output is row-major [n_mels, n_frames].
    transcribe_status compute(const float *        pcm,
                              size_t               n_samples,
                              std::vector<float> & out_mel,
                              int &                out_n_frames) const;
};

} // namespace transcribe::gigaam
