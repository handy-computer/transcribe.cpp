// arch/funasr_nano/frontend.cpp - kaldi HTK fbank + LFR (no CMVN).
//
// Pruned fork of arch/sensevoice/frontend.cpp. Identical kaldi fbank
// chain + LFR stack; CMVN add+mul is removed because Fun-ASR-Nano was
// trained on raw LFR fbank.

#include "frontend.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace transcribe::funasr_nano {

namespace {

constexpr float kPreemphasis = 0.97f;
constexpr float kMelLowFreq  = 20.0f;
constexpr float kMelHighFreq = 0.0f;  // 0 → use sample_rate / 2

int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

double hz_to_mel(double f) { return 1127.0 * std::log(1.0 + f / 700.0); }
double mel_to_hz(double m) { return 700.0 * (std::exp(m / 1127.0) - 1.0); }

std::vector<float> build_htk_mel_filterbank(int sample_rate,
                                            int padded_n_fft,
                                            int n_mels,
                                            float low_freq,
                                            float high_freq)
{
    const int n_freq = padded_n_fft / 2 + 1;
    if (high_freq <= 0.0f) {
        high_freq = static_cast<float>(sample_rate) * 0.5f;
    }
    const double mel_low  = hz_to_mel(low_freq);
    const double mel_high = hz_to_mel(high_freq);
    const double mel_step = (mel_high - mel_low) / (n_mels + 1);
    const double fft_bin_width = static_cast<double>(sample_rate) /
                                 static_cast<double>(padded_n_fft);

    std::vector<float> fb(static_cast<size_t>(n_mels) * n_freq, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const double left_mel   = mel_low + m * mel_step;
        const double center_mel = mel_low + (m + 1) * mel_step;
        const double right_mel  = mel_low + (m + 2) * mel_step;
        const double left_hz   = mel_to_hz(left_mel);
        const double center_hz = mel_to_hz(center_mel);
        const double right_hz  = mel_to_hz(right_mel);
        for (int b = 0; b < n_freq; ++b) {
            const double bin_hz = b * fft_bin_width;
            double w = 0.0;
            if (bin_hz > left_hz && bin_hz < right_hz) {
                if (bin_hz <= center_hz) {
                    w = (bin_hz - left_hz) / (center_hz - left_hz);
                } else {
                    w = (right_hz - bin_hz) / (right_hz - center_hz);
                }
            }
            if (w < 0.0) w = 0.0;
            fb[static_cast<size_t>(m) * n_freq + b] = static_cast<float>(w);
        }
    }
    return fb;
}

std::vector<float> build_hamming_window(int N) {
    std::vector<float> w(static_cast<size_t>(N));
    if (N <= 1) {
        for (auto & v : w) v = 1.0f;
        return w;
    }
    const double a = 2.0 * M_PI / (N - 1);
    for (int i = 0; i < N; ++i) {
        w[static_cast<size_t>(i)] =
            static_cast<float>(0.54 - 0.46 * std::cos(a * i));
    }
    return w;
}

void fft_radix2(double * data, int n) {
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[2 * i],     data[2 * j]);
            std::swap(data[2 * i + 1], data[2 * j + 1]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang     = -2.0 * M_PI / static_cast<double>(len);
        const double wlen_re = std::cos(ang);
        const double wlen_im = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;
            for (int k = 0; k < len / 2; ++k) {
                const int a = 2 * (i + k);
                const int b = 2 * (i + k + len / 2);
                const double u_re = data[a];
                const double u_im = data[a + 1];
                const double v_re = data[b]     * w_re - data[b + 1] * w_im;
                const double v_im = data[b]     * w_im + data[b + 1] * w_re;
                data[a]     = u_re + v_re;
                data[a + 1] = u_im + v_im;
                data[b]     = u_re - v_re;
                data[b + 1] = u_im - v_im;
                const double tmp_re = w_re * wlen_re - w_im * wlen_im;
                const double tmp_im = w_re * wlen_im + w_im * wlen_re;
                w_re = tmp_re;
                w_im = tmp_im;
            }
        }
    }
}

} // namespace

KaldiFbankFrontend::KaldiFbankFrontend(const FunAsrNanoHParams & hp)
    : n_mels_(hp.fe_num_mels)
    , sample_rate_(hp.fe_sample_rate)
    , win_length_(hp.fe_win_length)
    , hop_length_(hp.fe_hop_length)
    , padded_n_fft_(next_pow2(hp.fe_win_length))
    , lfr_m_(hp.fe_lfr_m)
    , lfr_n_(hp.fe_lfr_n)
    , d_input_(hp.enc_d_input)
    , upscale_samples_(hp.fe_upscale_samples)
{
    mel_fb_ = build_htk_mel_filterbank(
        sample_rate_, padded_n_fft_, n_mels_, kMelLowFreq, kMelHighFreq);
    window_ = build_hamming_window(win_length_);
}

int KaldiFbankFrontend::compute(const float *        pcm,
                                size_t               n_samples,
                                std::vector<float> & out_features) const
{
    if (pcm == nullptr || n_samples < static_cast<size_t>(win_length_)) {
        out_features.clear();
        return 0;
    }
    const int T_frames = 1 + static_cast<int>(
        (n_samples - static_cast<size_t>(win_length_)) /
        static_cast<size_t>(hop_length_));
    if (T_frames <= 0) {
        out_features.clear();
        return 0;
    }

    const int n_freq    = padded_n_fft_ / 2 + 1;
    const float upscale = upscale_samples_ ? 32768.0f : 1.0f;

    std::vector<double> frame(static_cast<size_t>(padded_n_fft_) * 2, 0.0);
    std::vector<float>  frame_f(static_cast<size_t>(win_length_), 0.0f);
    std::vector<float>  power(static_cast<size_t>(n_freq), 0.0f);
    std::vector<float>  mel(static_cast<size_t>(T_frames) * n_mels_, 0.0f);

    for (int t = 0; t < T_frames; ++t) {
        const size_t base = static_cast<size_t>(t) * hop_length_;
        for (int i = 0; i < win_length_; ++i) {
            frame_f[i] = pcm[base + i] * upscale;
        }
        double sum = 0.0;
        for (int i = 0; i < win_length_; ++i) sum += frame_f[i];
        const float mean = static_cast<float>(sum / win_length_);
        for (int i = 0; i < win_length_; ++i) frame_f[i] -= mean;
        for (int i = win_length_ - 1; i >= 1; --i) {
            frame_f[i] -= kPreemphasis * frame_f[i - 1];
        }
        frame_f[0] *= (1.0f - kPreemphasis);
        for (int i = 0; i < win_length_; ++i) frame_f[i] *= window_[i];

        for (int i = 0; i < padded_n_fft_; ++i) {
            const float re = i < win_length_ ? frame_f[i] : 0.0f;
            frame[2 * i]     = static_cast<double>(re);
            frame[2 * i + 1] = 0.0;
        }
        fft_radix2(frame.data(), padded_n_fft_);

        for (int b = 0; b < n_freq; ++b) {
            const double re = frame[2 * b];
            const double im = frame[2 * b + 1];
            power[b] = static_cast<float>(re * re + im * im);
        }

        float * mel_row = mel.data() + static_cast<size_t>(t) * n_mels_;
        for (int m = 0; m < n_mels_; ++m) {
            const float * fb_row = mel_fb_.data() +
                                   static_cast<size_t>(m) * n_freq;
            float acc = 0.0f;
            for (int b = 0; b < n_freq; ++b) {
                acc += fb_row[b] * power[b];
            }
            constexpr float kMelEpsilon = 1.1920928955078125e-07f;
            mel_row[m] = std::log(std::max(acc, kMelEpsilon));
        }
    }

    // LFR stacking with kaldi-style centered left padding (lfr_m=7, lfr_n=6).
    const int left_pad = (lfr_m_ - 1) / 2;
    const int T_lfr    = (T_frames + lfr_n_ - 1) / lfr_n_;
    out_features.assign(static_cast<size_t>(T_lfr) * d_input_, 0.0f);
    for (int t_lfr = 0; t_lfr < T_lfr; ++t_lfr) {
        float * out_row = out_features.data() +
                          static_cast<size_t>(t_lfr) * d_input_;
        for (int k = 0; k < lfr_m_; ++k) {
            const int padded_idx = t_lfr * lfr_n_ + k;
            int src;
            if (padded_idx < left_pad) {
                src = 0;
            } else {
                src = padded_idx - left_pad;
                if (src >= T_frames) src = T_frames - 1;
            }
            const float * src_row = mel.data() +
                                    static_cast<size_t>(src) * n_mels_;
            std::memcpy(out_row + static_cast<size_t>(k) * n_mels_,
                        src_row, sizeof(float) * n_mels_);
        }
        // No CMVN — Fun-ASR-Nano uses raw LFR features.
    }
    return T_lfr;
}

} // namespace transcribe::funasr_nano
