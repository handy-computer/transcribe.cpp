// arch/medasr/mel.cpp - MedASR (LASR) log-mel feature extractor.
//
// Reference: transformers.models.lasr.feature_extraction_lasr.LasrFeatureExtractor
// at the pinned commit.
//
//   x = audio (16 kHz mono fp32)
//   frames = unfold(x, win_length=400, hop=160)        # center=False, no pad
//   frames *= hann_symmetric(400)
//   stft = rfft(frames, n=512)                         # zero-pad 400 -> 512
//   power = |stft|^2
//   mel = power @ filterbank.T                         # filterbank shape [n_freq=257, n_mels=128]
//   log_mel = log(clamp(mel, min=1e-5))                # natural log
//
// Window and filterbank are baked into the GGUF by convert-medasr.py
// (`frontend.window` and `frontend.mel_filterbank`); this class copies
// them out of the backend buffer at init time and reuses the same FFT
// shape gigaam does (mixed-radix Cooley-Tukey, radix-2 to N=2 leaves).

#include "mel.h"

#include "weights.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace transcribe::medasr {

namespace {

// Naive O(N^2) DFT leaf for the mixed-radix path; reached when N is odd.
// For n_fft=512 (= 2^9) the recursion bottoms out at N=1 and never hits
// this path, but keep it for robustness if a future variant changes
// n_fft to a non-power-of-two.
void dft_naive(const float * in, int N, float * out) {
    for (int k = 0; k < N; ++k) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; ++n) {
            const double theta = -2.0 * M_PI * k * n / N;
            re += in[n] * std::cos(theta);
            im += in[n] * std::sin(theta);
        }
        out[2 * k    ] = static_cast<float>(re);
        out[2 * k + 1] = static_cast<float>(im);
    }
}

// Mixed-radix Cooley-Tukey FFT. Recursively halves while N is even;
// falls back to dft_naive on the odd leaf. Same pattern as gigaam/mel.cpp.
//
// Buffers:
//   in : 2N floats (first N: input, second N: even/odd scratch).
//   out: 8N floats (top-level result in first 2N).
void mixed_radix_fft(float * in, int N, float * out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    if (N & 1) {
        dft_naive(in, N, out);
        return;
    }
    const int half_N = N / 2;
    float * even = in + N;
    for (int i = 0; i < half_N; ++i) even[i] = in[2 * i];
    float * even_fft = out + 2 * N;
    mixed_radix_fft(even, half_N, even_fft);

    float * odd = even;
    for (int i = 0; i < half_N; ++i) odd[i] = in[2 * i + 1];
    float * odd_fft = even_fft + N;
    mixed_radix_fft(odd, half_N, odd_fft);

    for (int k = 0; k < half_N; ++k) {
        const double theta = -2.0 * M_PI * k / N;
        const float w_re = static_cast<float>(std::cos(theta));
        const float w_im = static_cast<float>(std::sin(theta));
        const float re_odd = odd_fft[2 * k    ];
        const float im_odd = odd_fft[2 * k + 1];
        out[2 * k    ]            = even_fft[2 * k    ] + w_re * re_odd - w_im * im_odd;
        out[2 * k + 1]            = even_fft[2 * k + 1] + w_re * im_odd + w_im * re_odd;
        out[2 * (k + half_N)    ] = even_fft[2 * k    ] - w_re * re_odd + w_im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - w_re * im_odd - w_im * re_odd;
    }
}

} // namespace

void MedAsrMelFrontend::init(const MedAsrHParams & hp,
                             const MedAsrWeights & w) {
    n_mels        = hp.fe_num_mels;
    n_fft         = hp.fe_n_fft;
    win_length    = hp.fe_win_length;
    hop_length    = hp.fe_hop_length;
    n_freq        = n_fft / 2 + 1;
    log_clamp_min = hp.fe_log_clamp_min;

    hann.assign(win_length, 0.0f);
    if (w.frontend_window != nullptr) {
        ggml_backend_tensor_get(w.frontend_window, hann.data(), 0,
                                hann.size() * sizeof(float));
    }

    // The converter emits the filterbank with numpy shape (n_freq, n_mels);
    // GGUF reverses to ggml ne=[n_mels, n_freq] (fast=n_mels). Read it into
    // a host buffer of the same orientation, then accumulate per-mel by
    // walking [n_freq] strides — see compute().
    mel_fb.assign(static_cast<size_t>(n_mels) * n_freq, 0.0f);
    if (w.frontend_mel_filterbank != nullptr) {
        ggml_backend_tensor_get(w.frontend_mel_filterbank, mel_fb.data(), 0,
                                mel_fb.size() * sizeof(float));
    }
}

int MedAsrMelFrontend::n_frames_for(size_t n_samples) const {
    if (static_cast<int>(n_samples) < win_length) return 0;
    return static_cast<int>((n_samples - win_length) / hop_length) + 1;
}

transcribe_status MedAsrMelFrontend::compute(const float *        pcm,
                                             size_t               n_samples,
                                             std::vector<float> & out_mel,
                                             int &                out_n_frames) const
{
    if (pcm == nullptr) return TRANSCRIBE_ERR_INVALID_ARG;

    const int n_frames = n_frames_for(n_samples);
    if (n_frames <= 0) return TRANSCRIBE_ERR_INVALID_ARG;
    out_n_frames = n_frames;

    // Output layout: [T_mel, n_mels] row-major — matches the reference
    // dump bytes byte-for-byte so the encoder's `mel.in` tensor (ne=[n_mels,
    // T_mel]) can receive it via a single ggml_backend_tensor_set call.
    out_mel.assign(static_cast<size_t>(n_frames) * n_mels, 0.0f);

    // Scratch buffers for the FFT.
    std::vector<float> fft_in (2 * n_fft);
    std::vector<float> fft_out(8 * n_fft);
    std::vector<float> power_spec(n_freq);

    for (int t = 0; t < n_frames; ++t) {
        const size_t start = static_cast<size_t>(t) * hop_length;
        // Zero-pad fft_in[0..n_fft); fill first win_length samples
        // with pcm[start..start+win_length) * hann.
        std::fill(fft_in.begin(), fft_in.begin() + n_fft, 0.0f);
        for (int k = 0; k < win_length; ++k) {
            fft_in[k] = pcm[start + k] * hann[k];
        }
        mixed_radix_fft(fft_in.data(), n_fft, fft_out.data());

        // power_spec[k] = |X[k]|^2 (power=2).
        for (int k = 0; k < n_freq; ++k) {
            const float re = fft_out[2 * k    ];
            const float im = fft_out[2 * k + 1];
            power_spec[k] = re * re + im * im;
        }

        // mel[m] = sum_k filterbank[k, m] * power_spec[k], then log(clamp).
        // mel_fb is laid out with ne[0]=n_mels (fast), ne[1]=n_freq (slow);
        // host buffer index = freq * n_mels + mel. For a fixed mel m,
        // walking k means stride n_mels.
        const size_t row_off = static_cast<size_t>(t) * n_mels;
        for (int m = 0; m < n_mels; ++m) {
            float acc = 0.0f;
            for (int k = 0; k < n_freq; ++k) {
                acc += mel_fb[static_cast<size_t>(k) * n_mels + m] *
                       power_spec[k];
            }
            if (acc < log_clamp_min) acc = log_clamp_min;
            out_mel[row_off + m] = std::log(acc);
        }
    }
    return TRANSCRIBE_OK;
}

} // namespace transcribe::medasr
