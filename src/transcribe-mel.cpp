// transcribe-mel.cpp - native C++ log-mel feature extractor.
//
// Implementation notes
// --------------------
//
// The pipeline below matches NeMo's AudioToMelSpectrogramPreprocessor
// (FilterbankFeatures.forward in nemo/collections/asr/parts/preprocessing/
// features.py) as exported in the Parakeet 0.6B nemo128.onnx
// preprocessor. Phase 3 preflight cross-checked the algorithm three
// ways: NeMo source, ONNX op graph + initializers, and a standalone
// PyTorch reproduction. Constants and ordering decisions:
//
//   * Pre-emphasis alpha: read from MelConfig::pre_emphasis (0.97 for
//     real Parakeet; 0.0 disables the step entirely).
//
//   * Reflect padding by n_fft/2 on each side. The NeMo source on
//     main says pad_mode="constant" but the ONNX export uses
//     reflect, and the released checkpoint was packaged with the
//     reflect path; reflect is what produces working transcriptions.
//     Trust the ONNX, ignore the source on this one point.
//
//   * STFT in fp64. Matches the ONNX which casts the input to f64
//     before STFT and back to f32 after. Eliminates precision drift
//     against the golden tensor.
//
//   * Window: symmetric Hann length win_length, zero-padded to n_fft.
//     The NeMo source explicitly passes periodic=False
//     (features.py:316). The MLX reference is wrong on this point.
//
//   * Power spectrum: re^2 + im^2 (matches ONNX ReduceSumSquare).
//     mag_power=2.0 is collapsed into the squaring; we never compute
//     the magnitude itself.
//
//   * Mel filterbank: Slaney-normalized librosa formula computed at
//     constructor time from (sr, n_fft, n_mels, fmin, fmax). Verified
//     to match the ONNX-baked filterbank to ~3e-7 in fp32.
//
//   * Log epsilon: 2^-24 (NeMo log_zero_guard_value default).
//     Hardcoded; not in the schema. Different from the 1e-5 we
//     initially planned for; the value comes straight from
//     features.py:256.
//
//   * Per-feature normalize uses the UNBIASED variance estimator,
//     dividing by (n_frames - 1) not n_frames. This is what NeMo's
//     normalize_batch does (features.py line 79) and what the ONNX
//     graph reproduces (Sub-1 op before the Div). The biased
//     estimator would silently drift the test by ~1/n_frames.
//
//   * Normalize epsilon: 1e-5. Matches NeMo's CONSTANT.
//
//   * Dither: never applied. NeMo gates dither on self.training,
//     so inference always sees a deterministic frontend. We don't
//     even read MelConfig::dither.
//
// Output layout: [num_mels, n_frames] row-major float32. The encoder
// builder in phase 4 will copy this into a ggml_tensor at the
// backend boundary; the frontend itself stays backend-free.

#include "transcribe-mel.h"

#ifdef __APPLE__
#  include <Accelerate/Accelerate.h>
#elif TRANSCRIBE_HAS_BLAS
#  include <cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace transcribe {

namespace {

// Hardcoded NeMo numerical constants. See file-level comment.
constexpr float kLogEps  = 5.9604644775390625e-08f;  // 2^-24
constexpr float kNormEps = 1.0e-05f;                  // NeMo CONSTANT

// ---------- Slaney mel scale (matches librosa.filters.mel) ----------

constexpr double kSlaneyFsp        = 200.0 / 3.0;
constexpr double kSlaneyMinLogHz   = 1000.0;
// kSlaneyLogStep = log(6.4) / 27.0; not constexpr because std::log
// isn't constexpr in C++17. Computed inline below.

inline double slaney_hz_to_mel(double hz) {
    const double min_log_mel = kSlaneyMinLogHz / kSlaneyFsp;
    const double logstep     = std::log(6.4) / 27.0;
    if (hz < kSlaneyMinLogHz) {
        return hz / kSlaneyFsp;
    }
    return min_log_mel + std::log(hz / kSlaneyMinLogHz) / logstep;
}

inline double slaney_mel_to_hz(double mel) {
    const double min_log_mel = kSlaneyMinLogHz / kSlaneyFsp;
    const double logstep     = std::log(6.4) / 27.0;
    if (mel < min_log_mel) {
        return mel * kSlaneyFsp;
    }
    return kSlaneyMinLogHz * std::exp(logstep * (mel - min_log_mel));
}

// Build the librosa Slaney-normalized mel filterbank.
// Output: [num_mels * n_freq] row-major float32.
// Matches librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax,
//                              norm='slaney') to within ~3e-7 in
// fp32 (verified against the ONNX-baked filterbank in
// nemo128.onnx).
void build_mel_filterbank_slaney(
    int sr, int n_fft, int n_mels, double fmin, double fmax,
    std::vector<float> & out)
{
    const int n_freq = n_fft / 2 + 1;
    out.assign(static_cast<size_t>(n_mels) * n_freq, 0.0f);

    // FFT bin frequencies in Hz: linspace(0, sr/2, n_freq) = k * sr / n_fft
    // (librosa uses np.linspace which gives identical values for this case).
    std::vector<double> fft_freqs(n_freq);
    for (int k = 0; k < n_freq; ++k) {
        fft_freqs[k] = static_cast<double>(sr) * k / static_cast<double>(n_fft);
    }

    // Mel boundary frequencies: n_mels + 2 points spaced uniformly in mel
    // scale. The first and last bound the entire bank; the middle n_mels
    // are the centers of the triangular filters.
    const double mel_min = slaney_hz_to_mel(fmin);
    const double mel_max = slaney_hz_to_mel(fmax);
    std::vector<double> hz_freqs(n_mels + 2);
    for (int m = 0; m < n_mels + 2; ++m) {
        const double mel = mel_min +
            (mel_max - mel_min) * m / static_cast<double>(n_mels + 1);
        hz_freqs[m] = slaney_mel_to_hz(mel);
    }

    // Triangular filters with Slaney area normalization.
    // For mel bin m: rises linearly from hz_freqs[m] to hz_freqs[m+1],
    // then falls linearly to hz_freqs[m+2]. The Slaney enorm scales
    // by 2 / (right - left) so each filter has unit area.
    std::vector<double> fdiff(n_mels + 1);
    for (int m = 0; m < n_mels + 1; ++m) {
        fdiff[m] = hz_freqs[m + 1] - hz_freqs[m];
    }

    for (int m = 0; m < n_mels; ++m) {
        const double enorm = 2.0 / (hz_freqs[m + 2] - hz_freqs[m]);
        for (int k = 0; k < n_freq; ++k) {
            const double lower = (fft_freqs[k]   - hz_freqs[m])     / fdiff[m];
            const double upper = (hz_freqs[m + 2] - fft_freqs[k])   / fdiff[m + 1];
            const double w     = std::max(0.0, std::min(lower, upper));
            out[static_cast<size_t>(m) * n_freq + k] =
                static_cast<float>(w * enorm);
        }
    }
}

// Build a symmetric Hann window of length win_length, zero-padded to
// n_fft on both sides. Matches torch.hann_window(win_length,
// periodic=False) followed by torch.stft's automatic centering of a
// shorter window inside the n_fft frame.
void build_hann_window_symmetric_padded(
    int win_length, int n_fft, std::vector<double> & out)
{
    out.assign(n_fft, 0.0);
    const int pad_each = (n_fft - win_length) / 2;
    for (int k = 0; k < win_length; ++k) {
        out[pad_each + k] =
            0.5 - 0.5 * std::cos(2.0 * M_PI * k /
                                 static_cast<double>(win_length - 1));
    }
}

// In-place radix-2 Cooley-Tukey FFT, fp64. Operates on n complex
// numbers stored as interleaved (re, im, re, im, ...). n must be a
// power of 2. For n=512 (Parakeet) this is ~80 lines and runs in
// ~10 us per call; the entire jfk.wav frontend (1101 frames) takes
// well under 50 ms on M-series.
//
// Why not vendor pocketfft / KISS FFT: n_fft=512 is small, the
// frontend is one-time per audio buffer (not on the inference hot
// path), and a single hand-rolled radix-2 keeps the dependency
// surface zero. Drop in pocketfft only if profiling later shows the
// frontend is a bottleneck.
void fft_radix2(double * data, int n) {
    // Bit-reversal permutation.
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
    // Cooley-Tukey butterflies. Twiddle factors are computed
    // incrementally per butterfly group; this drifts by a few ULPs
    // for very large n but is bit-stable for n=512.
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

// ---------- MelFrontend ----------

MelFrontend::MelFrontend(const MelConfig & cfg) : cfg_(cfg) {
    n_freq_ = cfg.n_fft / 2 + 1;
    build_hann_window_symmetric_padded(cfg.win_length, cfg.n_fft, window_);
    build_mel_filterbank_slaney(
        cfg.sample_rate, cfg.n_fft, cfg.num_mels,
        static_cast<double>(cfg.f_min),
        static_cast<double>(cfg.f_max),
        mel_fb_);
}

int MelFrontend::n_frames_for(size_t n_samples) const {
    // Matches NeMo features_lens = (waveforms_lens / hop_length) + 1.
    // The +1 accounts for the centered first frame.
    return static_cast<int>(n_samples / static_cast<size_t>(cfg_.hop_length)) + 1;
}

transcribe_status MelFrontend::compute(
    const float * pcm,
    size_t n_samples,
    std::vector<float> & out_mel,
    int & out_n_mels,
    int & out_n_frames) const
{
    if (pcm == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int n_fft  = cfg_.n_fft;
    const int hop    = cfg_.hop_length;
    const int n_mels = cfg_.num_mels;
    const int n_freq = n_freq_;
    const int pad    = n_fft / 2;

    const int n_frames =
        static_cast<int>(n_samples / static_cast<size_t>(hop)) + 1;

    // Per-feature normalize divides by (n_frames - 1) and the
    // reflect pad needs at least pad+1 input samples to reflect
    // without going off the end. Both constraints fold into:
    if (n_frames < 2 || n_samples < static_cast<size_t>(pad + 1)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // ---- 1. Pre-emphasis (in fp64; one-tap filter, ULP drift vs fp32 negligible) ----
    std::vector<double> emph(n_samples);
    if (cfg_.pre_emphasis != 0.0f) {
        const double alpha = static_cast<double>(cfg_.pre_emphasis);
        emph[0] = static_cast<double>(pcm[0]);
        for (size_t i = 1; i < n_samples; ++i) {
            emph[i] = static_cast<double>(pcm[i]) -
                      alpha * static_cast<double>(pcm[i - 1]);
        }
    } else {
        for (size_t i = 0; i < n_samples; ++i) {
            emph[i] = static_cast<double>(pcm[i]);
        }
    }

    // ---- 2. Reflect-pad by n_fft/2 on each side ----
    // For input [a, b, c, d] padded by 2: [c, b, a, b, c, d, c, b].
    // Boundary samples are NOT repeated (reflect not symmetric).
    std::vector<double> padded(n_samples + 2 * static_cast<size_t>(pad));
    for (int i = 0; i < pad; ++i) {
        padded[i] = emph[static_cast<size_t>(pad - i)];
    }
    std::memcpy(
        padded.data() + pad,
        emph.data(),
        n_samples * sizeof(double));
    for (int i = 0; i < pad; ++i) {
        padded[static_cast<size_t>(pad) + n_samples + i] =
            emph[n_samples - 2 - static_cast<size_t>(i)];
    }
    // Free emph eagerly; it's not needed past this point.
    std::vector<double>().swap(emph);

    // ---- 3. STFT in fp64. Output power spectrum [n_frames * n_freq] f32 row-major. ----
    std::vector<float>  power(static_cast<size_t>(n_frames) * n_freq);

#ifdef __APPLE__
    // vDSP real-input FFT path (fp64, vectorized). Same precision as
    // the hand-rolled radix-2 — both are fp64 throughout — but ~5-10x
    // faster via Accelerate's SIMD kernels. The loader guarantees
    // n_fft is a power of 2.
    int log2n = 0;
    { int tmp = n_fft; while (tmp > 1) { tmp >>= 1; ++log2n; } }

    FFTSetupD fft_setup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);

    // Split-complex buffers reused across all frames.
    const size_t half_n = static_cast<size_t>(n_fft / 2);
    std::vector<double> fft_real(half_n);
    std::vector<double> fft_imag(half_n);
    DSPDoubleSplitComplex split = { fft_real.data(), fft_imag.data() };

    for (int t = 0; t < n_frames; ++t) {
        const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
        // Window-multiply and pack into split-complex form:
        // realp[k] = windowed[2k], imagp[k] = windowed[2k+1].
        for (size_t k = 0; k < half_n; ++k) {
            fft_real[k] = padded[start + 2 * k]     * window_[2 * k];
            fft_imag[k] = padded[start + 2 * k + 1] * window_[2 * k + 1];
        }

        vDSP_fft_zripD(fft_setup, &split, 1, log2n, FFT_FORWARD);

        // vDSP output is 2x the standard DFT. Power = re^2 + im^2,
        // so the raw squared values are 4x. Multiply by 0.25.
        //
        // Output packing: realp[0]=DC*2, imagp[0]=Nyquist*2,
        // realp[k]+i*imagp[k] = X[k]*2 for k=1..N/2-1.
        float * pwr_row =
            power.data() + static_cast<size_t>(t) * n_freq;
        pwr_row[0]         = static_cast<float>(fft_real[0] * fft_real[0] * 0.25);
        pwr_row[n_fft / 2] = static_cast<float>(fft_imag[0] * fft_imag[0] * 0.25);
        for (size_t k = 1; k < half_n; ++k) {
            pwr_row[k] = static_cast<float>(
                (fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k]) * 0.25);
        }
    }

    vDSP_destroy_fftsetupD(fft_setup);
#else
    // Hand-rolled radix-2 Cooley-Tukey FFT fallback for non-Apple.
    std::vector<double> frame(2 * static_cast<size_t>(n_fft));   // interleaved complex

    for (int t = 0; t < n_frames; ++t) {
        const size_t start = static_cast<size_t>(t) * static_cast<size_t>(hop);
        // Window-multiply real input into the complex frame buffer.
        for (int k = 0; k < n_fft; ++k) {
            frame[2 * static_cast<size_t>(k)]     =
                padded[start + static_cast<size_t>(k)] * window_[k];
            frame[2 * static_cast<size_t>(k) + 1] = 0.0;
        }
        fft_radix2(frame.data(), n_fft);
        // Power spectrum = re^2 + im^2 over the first n_freq bins.
        float * pwr_row =
            power.data() + static_cast<size_t>(t) * n_freq;
        for (int k = 0; k < n_freq; ++k) {
            const double re = frame[2 * static_cast<size_t>(k)];
            const double im = frame[2 * static_cast<size_t>(k) + 1];
            pwr_row[k] = static_cast<float>(re * re + im * im);
        }
    }
    std::vector<double>().swap(frame);
#endif
    std::vector<double>().swap(padded);

    // ---- 4. Mel filterbank matmul + log ----
    //
    // Computes log_mel = log(mel_fb @ power^T + eps).
    //   mel_fb: [n_mels, n_freq] row-major
    //   power:  [n_frames, n_freq] row-major
    //   result: [n_mels, n_frames] row-major
    //
    // When BLAS is available, this is a single sgemm call; otherwise
    // falls back to the scalar triple loop.
    std::vector<float> log_mel(
        static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));

#if defined(__APPLE__) || TRANSCRIBE_HAS_BLAS
    // C = A @ B^T  where A=[n_mels, n_freq], B=[n_frames, n_freq].
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_mels, n_frames, n_freq,
                1.0f, mel_fb_.data(), n_freq,
                power.data(), n_freq,
                0.0f, log_mel.data(), n_frames);
    {
        const size_t total = static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames);
        for (size_t i = 0; i < total; ++i) {
            log_mel[i] = std::log(log_mel[i] + kLogEps);
        }
    }
#else
    for (int t = 0; t < n_frames; ++t) {
        const float * pwr = power.data() + static_cast<size_t>(t) * n_freq;
        for (int m = 0; m < n_mels; ++m) {
            const float * fb_row =
                mel_fb_.data() + static_cast<size_t>(m) * n_freq;
            float acc = 0.0f;
            for (int k = 0; k < n_freq; ++k) {
                acc += fb_row[k] * pwr[k];
            }
            log_mel[static_cast<size_t>(m) * n_frames + t] =
                std::log(acc + kLogEps);
        }
    }
#endif
    std::vector<float>().swap(power);

    // ---- 5. Per-feature normalize, unbiased ----
    // For each mel bin m: subtract the mean across time, divide by
    // (std + 1e-5). The variance uses (n_frames - 1) in the
    // denominator (Bessel's correction) to match NeMo's
    // normalize_batch. fp64 accumulators throughout to keep the row
    // sums numerically stable; cast back to fp32 at storage.
    //
    // resize() instead of assign(): the loop below writes every
    // element exactly once, so the zero-fill that assign() would do
    // is dead work. For a caller that reuses the same out_mel
    // buffer across calls of the same length (e.g. --repeat N on
    // one audio file), resize() is a no-op while assign() would
    // touch every byte. Real-world saving is small (the buffer is
    // ~565 KB for jfk.wav) but it's free.
    out_mel.resize(
        static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));
    for (int m = 0; m < n_mels; ++m) {
        const float * src =
            log_mel.data() + static_cast<size_t>(m) * n_frames;
        float * dst = out_mel.data() + static_cast<size_t>(m) * n_frames;

        double sum = 0.0;
        for (int t = 0; t < n_frames; ++t) {
            sum += static_cast<double>(src[t]);
        }
        const double mean = sum / static_cast<double>(n_frames);

        double sumsq = 0.0;
        for (int t = 0; t < n_frames; ++t) {
            const double d = static_cast<double>(src[t]) - mean;
            sumsq += d * d;
        }
        const double var    = sumsq / static_cast<double>(n_frames - 1);
        const double stddev = std::sqrt(var);
        const double inv    = 1.0 / (stddev + static_cast<double>(kNormEps));

        for (int t = 0; t < n_frames; ++t) {
            dst[t] = static_cast<float>(
                (static_cast<double>(src[t]) - mean) * inv);
        }
    }

    out_n_mels   = n_mels;
    out_n_frames = n_frames;
    return TRANSCRIBE_OK;
}

} // namespace transcribe
