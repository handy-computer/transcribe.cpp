// arch/ecapa_tdnn/mel.cpp - SpeechBrain log-mel front end.
//
// See mel.h for the SpeechBrain pipeline this reproduces and the
// per-step numerical constants. Implementation notes:
//
//   * The FFT is the mixed-radix fp32 routine from transcribe-mel.cpp
//     (itself lifted from whisper.cpp). n_fft = 400 = 2^4 * 5^2, so it recurses four
//     radix-2 levels down to 25-point naive DFT leaves. There is no
//     power-of-two path here: this front end is non-pow2 only.
//   * Accumulation policy: the window and the FFT run in fp32 (matching
//     torch's fp32 STFT); the mel matmul, the log, the top-dB max and the
//     per-bin mean all accumulate in fp64 and round once at storage. That
//     is strictly more accurate than SpeechBrain's fp32 matmul, which is
//     the direction we want — the residual is far inside the fe.mel
//     tolerance.
//   * Output is frame-major [T, n_mels]. Every frame is written by
//     exactly one thread into its own slot, so the result is bit-
//     identical for any thread count.

#include "mel.h"

#include "transcribe-batch-util.h"
#include "transcribe-log.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

namespace transcribe::ecapa_tdnn {

namespace {

// Naive O(N^2) DFT leaf for the mixed-radix path. Used when N hits an odd
// factor during recursive halving (N=400 -> 200 -> 100 -> 50 -> 25 odd ->
// leaf). Uses the per-frontend fp32 LUT (lut_size divisible by N by
// construction).
void dft_naive_f32(const float * in, int N, const float * cos_lut, const float * sin_lut, int lut_size, float * out) {
    const int stride = lut_size / N;
    for (int k = 0; k < N; ++k) {
        float     re   = 0.0f;
        float     im   = 0.0f;
        // idx = (k * n * stride) % lut_size, advanced incrementally (same
        // indices, same accumulation order, no division per term).
        const int step = (k * stride) % lut_size;
        int       idx  = 0;
        for (int n = 0; n < N; ++n) {
            re += in[n] * cos_lut[idx];
            im -= in[n] * sin_lut[idx];
            idx += step;
            if (idx >= lut_size) {
                idx -= lut_size;
            }
        }
        out[2 * k]     = re;
        out[2 * k + 1] = im;
    }
}

// Cooley-Tukey mixed-radix FFT. Recursively halves while N is even and
// falls back to the naive DFT on odd leaves.
//
// Buffer contract:
//   in    : 2*N floats. First N hold the real input; second N are used as
//           scratch for the even/odd split at this level.
//   out   : 8*N floats. The top-level result lands in the first 2*N;
//           recursion uses the remainder as intermediate output.
void mixed_radix_fft_f32(float * in, int N, const float * cos_lut, const float * sin_lut, int lut_size, float * out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    if (N & 1) {
        dft_naive_f32(in, N, cos_lut, sin_lut, lut_size, out);
        return;
    }
    const int half_N = N / 2;
    float *   even   = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i] = in[2 * i];
    }
    float * even_fft = out + 2 * N;
    mixed_radix_fft_f32(even, half_N, cos_lut, sin_lut, lut_size, even_fft);

    float * odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2 * i + 1];
    }
    float * odd_fft = even_fft + N;
    mixed_radix_fft_f32(odd, half_N, cos_lut, sin_lut, lut_size, odd_fft);

    const int step = lut_size / N;
    for (int k = 0; k < half_N; ++k) {
        const int   idx           = k * step;
        const float w_re          = cos_lut[idx];
        const float w_im          = -sin_lut[idx];
        const float re_odd        = odd_fft[2 * k];
        const float im_odd        = odd_fft[2 * k + 1];
        out[2 * k]                = even_fft[2 * k] + w_re * re_odd - w_im * im_odd;
        out[2 * k + 1]            = even_fft[2 * k + 1] + w_re * im_odd + w_im * re_odd;
        out[2 * (k + half_N)]     = even_fft[2 * k] - w_re * re_odd + w_im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - w_re * im_odd - w_im * re_odd;
    }
}

// Twiddle LUT shared by the constructor and the test hook: computed in
// fp64 for accuracy, stored fp32 to match the FFT precision. lut_size ==
// n_fft, so every recursion level divides it exactly.
void build_twiddle_lut(int n_fft, std::vector<float> & cos_lut, std::vector<float> & sin_lut) {
    cos_lut.resize(static_cast<size_t>(n_fft));
    sin_lut.resize(static_cast<size_t>(n_fft));
    for (int i = 0; i < n_fft; ++i) {
        const double theta              = 2.0 * M_PI * i / static_cast<double>(n_fft);
        cos_lut[static_cast<size_t>(i)] = static_cast<float>(std::cos(theta));
        sin_lut[static_cast<size_t>(i)] = static_cast<float>(std::sin(theta));
    }
}

}  // namespace

// ---------- MelFrontend ----------

MelFrontend::MelFrontend(const MelConfig & cfg) : cfg_(cfg) {
    auto reject = [this](const char * field, const char * why) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "mel: invalid config field '%s': %s", field, why);
        status_ = TRANSCRIBE_ERR_GGUF;
    };

    if (cfg_.n_fft <= 0 || (cfg_.n_fft & 1) != 0) {
        // The mixed-radix FFT halves while N is even; an odd n_fft would
        // degenerate to a single O(N^2) leaf, and n_fft/2 padding would
        // no longer be exact.
        reject("n_fft", "must be a positive even integer");
    }
    if (cfg_.n_mels <= 0) {
        reject("n_mels", "must be positive");
    }
    if (cfg_.hop_length <= 0) {
        reject("hop_length", "must be positive");
    }
    if (cfg_.sample_rate <= 0) {
        reject("sample_rate", "must be positive");
    }
    if (cfg_.win_length != cfg_.n_fft) {
        // SpeechBrain passes win_length == n_fft, so there is no
        // sub-window zero-pad placement to get wrong. Reject rather than
        // silently guess where the window sits inside the FFT buffer.
        reject("win_length", "must equal n_fft");
    }
    if (cfg_.window_type != "hamming_periodic") {
        reject("window_type", "only \"hamming_periodic\" is implemented");
    }
    if (cfg_.pad_mode != "constant") {
        reject("pad_mode", "only \"constant\" is implemented");
    }
    if (cfg_.normalize != "sentence_mean" && cfg_.normalize != "none") {
        reject("normalize", "only \"sentence_mean\" and \"none\" are implemented");
    }

    if (status_ != TRANSCRIBE_OK) {
        return;
    }

    n_freq_ = cfg_.n_fft / 2 + 1;

    if (cfg_.filterbank.size() != static_cast<size_t>(cfg_.n_mels) * static_cast<size_t>(n_freq_)) {
        // The filterbank is checkpoint-provided (`frontend.mel_filterbank`);
        // SpeechBrain's triangle geometry is never rebuilt here.
        reject("filterbank", "must hold exactly n_mels * (n_fft/2 + 1) weights");
        return;
    }

    // Periodic Hamming: w[n] = 0.54 - 0.46*cos(2*pi*n/N), N = n_fft.
    // torch.hamming_window defaults to periodic=True. Computed in fp64,
    // stored fp32 (the STFT runs fp32).
    window_.resize(static_cast<size_t>(cfg_.n_fft));
    for (int n = 0; n < cfg_.n_fft; ++n) {
        const double v                  = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / static_cast<double>(cfg_.n_fft));
        window_[static_cast<size_t>(n)] = static_cast<float>(v);
    }

    build_twiddle_lut(cfg_.n_fft, cos_lut_, sin_lut_);

    fb_lo_.assign(static_cast<size_t>(cfg_.n_mels), 0);
    fb_hi_.assign(static_cast<size_t>(cfg_.n_mels), 0);
    for (int m = 0; m < cfg_.n_mels; ++m) {
        const float * row = cfg_.filterbank.data() + static_cast<size_t>(m) * static_cast<size_t>(n_freq_);
        int           lo  = n_freq_;
        int           hi  = 0;
        for (int k = 0; k < n_freq_; ++k) {
            if (row[k] != 0.0f) {
                lo = std::min(lo, k);
                hi = k + 1;
            }
        }
        fb_lo_[static_cast<size_t>(m)] = lo < hi ? lo : 0;
        fb_hi_[static_cast<size_t>(m)] = lo < hi ? hi : 0;
    }
}

int MelFrontend::n_frames_for(int64_t n_samples) const {
    if (n_samples < 0 || cfg_.hop_length <= 0) {
        return 0;
    }
    // torch center=True: the signal is padded by n_fft/2 on both sides and
    // framed at every hop, giving floor(n / hop) + 1 frames. The trailing
    // frames run into the zero pad; SpeechBrain keeps them.
    return static_cast<int>(n_samples / static_cast<int64_t>(cfg_.hop_length)) + 1;
}

transcribe_status MelFrontend::compute(const float *        pcm,
                                       int64_t              n_samples,
                                       std::vector<float> & out,
                                       int &                out_n_frames,
                                       int                  n_threads) const {
    if (status_ != TRANSCRIBE_OK) {
        // Constructor already logged the offending field.
        return status_;
    }
    if (pcm == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "mel: pcm is NULL");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (n_samples < 1) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "mel: n_samples = %lld, need at least 1",
                static_cast<long long>(n_samples));
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int n_fft   = cfg_.n_fft;
    const int hop     = cfg_.hop_length;
    const int n_mels  = cfg_.n_mels;
    const int n_freq  = n_freq_;
    const int pad     = n_fft / 2;
    const int n_frame = n_frames_for(n_samples);

    // ---- 1. Zero pad (torch center=True, pad_mode="constant") ----
    //
    // Frame t reads padded[t*hop .. t*hop + n_fft). The last frame starts
    // at (n_frame-1)*hop = floor(n/hop)*hop <= n, so it ends at most at
    // n + n_fft = n + 2*pad — exactly the buffer size, with no slack. The
    // check below turns an off-by-one into an error instead of a heap
    // over-read.
    const size_t padded_len = static_cast<size_t>(n_samples) + 2 * static_cast<size_t>(pad);
    if (static_cast<size_t>(n_frame - 1) * static_cast<size_t>(hop) + static_cast<size_t>(n_fft) > padded_len) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "mel: internal framing overflow (n_frames=%d, n_samples=%lld)", n_frame,
                static_cast<long long>(n_samples));
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    std::vector<float> padded(padded_len, 0.0f);
    std::memcpy(padded.data() + pad, pcm, static_cast<size_t>(n_samples) * sizeof(float));

    // ---- 2. STFT -> power -> mel -> log, per frame ----
    //
    // Output frame-major: out[t * n_mels + m]. Each thread owns a disjoint
    // set of frames (strided assignment keeps the load balanced), so the
    // result does not depend on the thread count.
    out.assign(static_cast<size_t>(n_frame) * static_cast<size_t>(n_mels), 0.0f);

    const double log_floor = static_cast<double>(cfg_.log_floor);

    int stft_threads = n_threads;
    if (stft_threads <= 0) {
        stft_threads = default_n_threads();
    }
    if (stft_threads > n_frame) {
        stft_threads = std::max(1, n_frame);
    }

    auto worker = [&](int tid) {
        std::vector<float> fft_in(2 * static_cast<size_t>(n_fft), 0.0f);
        std::vector<float> fft_out(8 * static_cast<size_t>(n_fft), 0.0f);
        std::vector<float> power(static_cast<size_t>(n_freq), 0.0f);

        for (int t = tid; t < n_frame; t += stft_threads) {
            const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
            for (int n = 0; n < n_fft; ++n) {
                fft_in[static_cast<size_t>(n)] =
                    padded[start + static_cast<size_t>(n)] * window_[static_cast<size_t>(n)];
            }
            mixed_radix_fft_f32(fft_in.data(), n_fft, cos_lut_.data(), sin_lut_.data(),
                                static_cast<int>(cos_lut_.size()), fft_out.data());
            for (int k = 0; k < n_freq; ++k) {
                const float re                = fft_out[2 * static_cast<size_t>(k)];
                const float im                = fft_out[2 * static_cast<size_t>(k) + 1];
                power[static_cast<size_t>(k)] = re * re + im * im;
            }

            float * dst = out.data() + static_cast<size_t>(t) * static_cast<size_t>(n_mels);
            for (int m = 0; m < n_mels; ++m) {
                const float * fb_row = cfg_.filterbank.data() + static_cast<size_t>(m) * static_cast<size_t>(n_freq);
                double        sum    = 0.0;
                const int     k_hi   = fb_hi_[static_cast<size_t>(m)];
                for (int k = fb_lo_[static_cast<size_t>(m)]; k < k_hi; ++k) {
                    sum += static_cast<double>(fb_row[k]) * static_cast<double>(power[static_cast<size_t>(k)]);
                }
                // SpeechBrain Filterbank.log_mel: 10*log10(max(x, amin))
                // with amin = 1e-10. ref_value is 1.0, so the usual
                // "- 10*log10(ref)" term is exactly zero.
                if (sum < log_floor) {
                    sum = log_floor;
                }
                dst[m] = static_cast<float>(10.0 * std::log10(sum));
            }
        }
    };
    run_on_threads(stft_threads, worker);

    const size_t total = out.size();

    // ---- 3. Top-dB floor ----
    //
    // Per utterance over time AND frequency, applied BEFORE the mean
    // normalisation (SpeechBrain clamps inside Filterbank, and
    // InputNormalization runs afterwards in compute_features).
    if (cfg_.top_db > 0.0f) {
        double max_all = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < total; ++i) {
            const double v = static_cast<double>(out[i]);
            if (v > max_all) {
                max_all = v;
            }
        }
        const double floor_val = max_all - static_cast<double>(cfg_.top_db);
        for (size_t i = 0; i < total; ++i) {
            if (static_cast<double>(out[i]) < floor_val) {
                out[i] = static_cast<float>(floor_val);
            }
        }
    }

    // ---- 4. Sentence-mean normalisation ----
    //
    // InputNormalization(norm_type="sentence", std_norm=False): subtract
    // each mel bin's mean over ALL frames. fp64 accumulator, one rounding
    // at storage.
    if (cfg_.normalize == "sentence_mean") {
        std::vector<double> mean(static_cast<size_t>(n_mels), 0.0);
        for (int t = 0; t < n_frame; ++t) {
            const float * row = out.data() + static_cast<size_t>(t) * static_cast<size_t>(n_mels);
            for (int m = 0; m < n_mels; ++m) {
                mean[static_cast<size_t>(m)] += static_cast<double>(row[m]);
            }
        }
        for (int m = 0; m < n_mels; ++m) {
            mean[static_cast<size_t>(m)] /= static_cast<double>(n_frame);
        }
        for (int t = 0; t < n_frame; ++t) {
            float * row = out.data() + static_cast<size_t>(t) * static_cast<size_t>(n_mels);
            for (int m = 0; m < n_mels; ++m) {
                row[m] = static_cast<float>(static_cast<double>(row[m]) - mean[static_cast<size_t>(m)]);
            }
        }
    }

    out_n_frames = n_frame;
    return TRANSCRIBE_OK;
}

// --- test hook ---------------------------------------------------------

void mel_test_fft(const float * in400, float * out_re201, float * out_im201) {
    constexpr int      kN = 400;
    std::vector<float> cos_lut;
    std::vector<float> sin_lut;
    build_twiddle_lut(kN, cos_lut, sin_lut);

    std::vector<float> fft_in(2 * static_cast<size_t>(kN), 0.0f);
    std::vector<float> fft_out(8 * static_cast<size_t>(kN), 0.0f);
    std::memcpy(fft_in.data(), in400, static_cast<size_t>(kN) * sizeof(float));

    mixed_radix_fft_f32(fft_in.data(), kN, cos_lut.data(), sin_lut.data(), kN, fft_out.data());

    for (int k = 0; k <= kN / 2; ++k) {
        out_re201[k] = fft_out[2 * static_cast<size_t>(k)];
        out_im201[k] = fft_out[2 * static_cast<size_t>(k) + 1];
    }
}

}  // namespace transcribe::ecapa_tdnn
