// arch/sensevoice/frontend.h - kaldi-style HTK mel fbank + LFR stack
// + per-feature CMVN, host-side, single-utterance.
//
// Mirrors funasr.frontends.wav_frontend.WavFrontend exactly so the
// post-frontend tensor `frontend.fbank.lfr.cmvn.out` is bit-comparable
// to the reference dump. Key differences from src/transcribe-mel.h
// (which targets Whisper-style slaney mels):
//
//   * Hamming window (kaldi's symmetric "hamming": 0.54 - 0.46·cos(2πn/(N-1))).
//   * Per-frame preemphasis coefficient 0.97 with kaldi's special
//     boundary condition `x[0] = x[0] * (1 - 0.97)`.
//   * Per-frame DC offset removal before preemph.
//   * Sample upscale ×32768 before framing (WavFrontend treats audio
//     as int16 magnitude in float storage).
//   * snip_edges=True: drop trailing partial frames.
//   * HTK mel filterbank (triangular bins peak=1, no area
//     normalization), low_freq=20 Hz, high_freq=Nyquist.
//   * round_to_power_of_two=True: per-frame FFT size = next_pow2(400) = 512,
//     even though stt.frontend.n_fft is the un-padded 400.
//   * Low-frame-rate stack (m=7, n=6): output frame i is the
//     concatenation of input frames [6i .. 6i+7).
//   * Per-feature CMVN: (x + shift) * scale, with the 560-d shift +
//     scale tensors baked under frontend.cmvn.{shift,scale}.
//
// The frontend is built once at load time (window + mel filterbank +
// CMVN cached) and run once per transcribe_run.

#pragma once

#include "weights.h"

#include <cstddef>
#include <vector>

namespace transcribe::sensevoice {

class KaldiFbankFrontend {
public:
    KaldiFbankFrontend(const SenseVoiceHParams &  hp,
                       const SenseVoiceWeights &  weights);

    // Compute LFR + CMVN'd features from raw [-1, 1] f32 PCM. Output is
    // row-major [T_lfr, d_input] stored in `out_features`. Returns the
    // number of LFR frames; 0 if the input is too short to produce any
    // LFR output (which means the model cannot run on it).
    int compute(const float *        pcm,
                size_t               n_samples,
                std::vector<float> & out_features) const;

    // Constants the runtime needs.
    int d_input()  const { return d_input_; }
    int n_mels()   const { return n_mels_; }
    int lfr_m()    const { return lfr_m_; }
    int lfr_n()    const { return lfr_n_; }

private:
    // Architecture parameters copied from hparams.
    int n_mels_      = 0;   // 80
    int sample_rate_ = 0;   // 16000
    int win_length_  = 0;   // 400
    int hop_length_  = 0;   // 160
    int padded_n_fft_ = 0;  // next_pow2(win_length_) = 512
    int lfr_m_       = 0;   // 7
    int lfr_n_       = 0;   // 6
    int d_input_     = 0;   // n_mels_ * lfr_m_ = 560
    bool upscale_samples_ = true;

    // Cached mel filterbank, [n_mels, padded_n_fft/2 + 1] row-major,
    // f32. HTK triangular bins, no slaney normalization.
    std::vector<float> mel_fb_;

    // Cached symmetric Hamming window of length win_length_.
    std::vector<float> window_;

    // Borrowed CMVN buffers (life-bound to the model's ggml context).
    // shift_ and scale_ are read at construction time into host arrays
    // because compute() runs in user thread / no scheduler.
    std::vector<float> cmvn_shift_;
    std::vector<float> cmvn_scale_;
};

} // namespace transcribe::sensevoice
