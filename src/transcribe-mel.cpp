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
//     (features.py:316).
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
#include <limits>
#include <thread>
#include <vector>

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

// Build a Hann window of length win_length, zero-padded to n_fft on
// both sides. `periodic` selects between torch.hann_window(N, periodic=
// False) (symmetric, default — cos(2πk/(N-1))) and periodic=True
// (Whisper — cos(2πk/N)).
void build_hann_window_symmetric_padded(
    int win_length, int n_fft, bool periodic, std::vector<double> & out)
{
    out.assign(n_fft, 0.0);
    const int pad_each = (n_fft - win_length) / 2;
    const double denom = periodic
                       ? static_cast<double>(win_length)
                       : static_cast<double>(win_length - 1);
    for (int k = 0; k < win_length; ++k) {
        out[pad_each + k] =
            0.5 - 0.5 * std::cos(2.0 * M_PI * k / denom);
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
// Process-wide sin/cos cache, indexed by 2π*i / SIN_COS_N_COUNT. The
// leaf DFT at size N uses LUT-stride = SIN_COS_N_COUNT / N. A single
// 65536-entry table handles any N that divides 65536 evenly. For
// Whisper's 400 = 2^4 * 5^2 the leaf size is 25, and 65536 / 25 ≈ 2621.
// To keep the table index exact for all leaves we encounter we pad to
// the LCM of common leaf sizes. 2520 covers 1..10 and up to ~30
// comfortably — sufficient for n_fft ∈ {400, 512, 1024, …}. The FFT
// butterflies also use the LUT via 2π*k/N with stride LCM/N.
constexpr int kSinCosLut = 5040;  // 7! covers divisors 1..10 exactly

struct SinCosLut {
    double cos_vals[kSinCosLut];
    double sin_vals[kSinCosLut];

    SinCosLut() {
        for (int i = 0; i < kSinCosLut; ++i) {
            const double theta = 2.0 * M_PI * i / kSinCosLut;
            cos_vals[i] = std::cos(theta);
            sin_vals[i] = std::sin(theta);
        }
    }
};

const SinCosLut & sin_cos_lut() {
    static SinCosLut g_lut;
    return g_lut;
}

// Naive O(N^2) DFT leaf for the mixed-radix path. Used when N hits an
// odd factor during recursive halving (e.g. N=400 → 200 → 100 → 50 → 25
// odd → leaf). Uses the shared sin_cos_lut() so trig calls collapse to
// table lookups.
void dft_naive(const double * in, int N, double * out) {
    const auto & lut = sin_cos_lut();
    const int    stride = kSinCosLut / N;  // assumes N divides kSinCosLut

    if (stride * N == kSinCosLut) {
        for (int k = 0; k < N; ++k) {
            double re = 0.0;
            double im = 0.0;
            for (int n = 0; n < N; ++n) {
                const int idx = (k * n * stride) % kSinCosLut;
                re += in[n] * lut.cos_vals[idx];
                im -= in[n] * lut.sin_vals[idx];
            }
            out[2 * k    ] = re;
            out[2 * k + 1] = im;
        }
        return;
    }

    // Fallback for N that doesn't evenly divide kSinCosLut.
    for (int k = 0; k < N; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (int n = 0; n < N; ++n) {
            const double ang = -2.0 * M_PI *
                static_cast<double>(k) * static_cast<double>(n) /
                static_cast<double>(N);
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[2 * k    ] = re;
        out[2 * k + 1] = im;
    }
}

// Cooley-Tukey mixed-radix FFT. Recursively halves while N is even and
// falls back to naive DFT on odd leaves. Lifted directly from
// whisper.cpp (src/whisper.cpp::fft); lines up with our need to STFT
// Whisper's n_fft=400 = 2^4 * 5^2 efficiently (25-point DFT leaves,
// four levels of radix-2 butterflies above).
//
// Buffer contract (lifted from whisper.cpp):
//   in    : 2*N doubles. First N hold the real input; second N are
//           used as scratch for the even/odd split at this level.
//   out   : 8*N doubles. The top-level result lands in the first 2*N;
//           recursion uses the remainder as intermediate output.
void mixed_radix_fft(double * in, int N, double * out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0;
        return;
    }
    if (N & 1) {
        dft_naive(in, N, out);
        return;
    }

    const int half_N = N / 2;

    // Even/odd split into the scratch half of `in`.
    double * even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i] = in[2 * i];
    }
    double * even_fft = out + 2 * N;
    mixed_radix_fft(even, half_N, even_fft);

    double * odd = even;  // safe to reuse — `even` is no longer needed
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2 * i + 1];
    }
    double * odd_fft = even_fft + N;
    mixed_radix_fft(odd, half_N, odd_fft);

    const auto & lut = sin_cos_lut();
    const int    step = kSinCosLut / N;
    const bool   use_lut = (step * N == kSinCosLut);

    for (int k = 0; k < half_N; ++k) {
        double w_re, w_im;
        if (use_lut) {
            const int idx = k * step;
            w_re =  lut.cos_vals[idx];
            w_im = -lut.sin_vals[idx];
        } else {
            const double ang = -2.0 * M_PI * k / static_cast<double>(N);
            w_re = std::cos(ang);
            w_im = std::sin(ang);
        }
        const double re_odd = odd_fft[2 * k    ];
        const double im_odd = odd_fft[2 * k + 1];
        out[2 * k    ]                 = even_fft[2 * k    ] + w_re * re_odd - w_im * im_odd;
        out[2 * k + 1]                 = even_fft[2 * k + 1] + w_re * im_odd + w_im * re_odd;
        out[2 * (k + half_N)    ]      = even_fft[2 * k    ] - w_re * re_odd + w_im * im_odd;
        out[2 * (k + half_N) + 1]      = even_fft[2 * k + 1] - w_re * im_odd - w_im * re_odd;
    }
}

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

    // Use checkpoint-provided window if available, else compute.
    if (!cfg.window.empty()) {
        // Convert f32 checkpoint window to f64 and zero-pad to n_fft.
        window_.resize(cfg.n_fft, 0.0);
        const int total_pad = cfg.n_fft - cfg.win_length;
        const int left_pad  = total_pad / 2;
        for (int i = 0; i < cfg.win_length && i < static_cast<int>(cfg.window.size()); ++i) {
            window_[left_pad + i] = static_cast<double>(cfg.window[i]);
        }
    } else {
        const bool periodic = (cfg.window_type == "hann_periodic");
        build_hann_window_symmetric_padded(cfg.win_length, cfg.n_fft,
                                           periodic, window_);
    }

    // Use checkpoint-provided mel filterbank if available, else compute.
    if (!cfg.filterbank.empty()) {
        mel_fb_ = cfg.filterbank;
    } else {
        build_mel_filterbank_slaney(
            cfg.sample_rate, cfg.n_fft, cfg.num_mels,
            static_cast<double>(cfg.f_min),
            static_cast<double>(cfg.f_max),
            mel_fb_);
    }

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
    int & out_n_frames,
    int n_threads) const
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
    const bool use_reflect = (cfg_.pad_mode != "constant");
    if (n_frames < 2 || (use_reflect && n_samples < static_cast<size_t>(pad + 1))) {
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

    // ---- 2. Pad by n_fft/2 on each side ----
    std::vector<double> padded(n_samples + 2 * static_cast<size_t>(pad));
    if (use_reflect) {
        // Reflect padding.
        // For input [a, b, c, d] padded by 2: [c, b, a, b, c, d, c, b].
        // Boundary samples are NOT repeated (reflect not symmetric).
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
    } else {
        // Constant (zero) padding.
        std::memset(padded.data(), 0,
                    static_cast<size_t>(pad) * sizeof(double));
        std::memcpy(
            padded.data() + pad,
            emph.data(),
            n_samples * sizeof(double));
        std::memset(padded.data() + pad + n_samples, 0,
                    static_cast<size_t>(pad) * sizeof(double));
    }
    // Free emph eagerly; it's not needed past this point.
    std::vector<double>().swap(emph);

    // ---- 3. STFT in fp64. Output power spectrum [n_frames * n_freq] f32 row-major. ----
    std::vector<float>  power(static_cast<size_t>(n_frames) * n_freq);

    // Thread count for the STFT loop. The filterbank matmul (step 4)
    // is already parallelized by Accelerate/OpenBLAS; threading only
    // the STFT is where the win is.
    int stft_threads = n_threads;
    if (stft_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        stft_threads = hw > 0 ? static_cast<int>(hw) : 1;
        if (stft_threads > 8) stft_threads = 8;
    }
    if (stft_threads > n_frames) {
        stft_threads = std::max(1, n_frames);
    }

    // Both the vDSP real-input FFT path and the hand-rolled radix-2
    // fallback require n_fft to be a power of 2. Qwen3-ASR uses
    // n_fft=400 (Whisper), which factors as 2^4 * 5^2 and is not
    // supported. For non-pow2 sizes we drop to a mixed-radix FFT with
    // an odd-factor DFT leaf.
    const bool n_fft_is_pow2 = ((n_fft > 0) && ((n_fft & (n_fft - 1)) == 0));

    // Per-thread worker lambdas. Each worker owns its FFT scratch; the
    // power[] buffer is written with disjoint frame indices so no
    // locking is needed. Stride is n_threads so work is interleaved —
    // keeps per-worker memory access patterns contiguous enough to hit
    // the L2 after the first frame.
    auto run_threaded = [&](auto && worker) {
        if (stft_threads <= 1) {
            worker(0);
            return;
        }
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(stft_threads - 1));
        for (int tid = 1; tid < stft_threads; ++tid) {
            pool.emplace_back(worker, tid);
        }
        worker(0);
        for (auto & th : pool) th.join();
    };

    if (!n_fft_is_pow2) {
        // Mixed-radix FFT: recursive radix-2 butterflies until N hits
        // an odd factor (25 for Whisper's 400=2^4*5^2), then naive DFT
        // on the odd leaf. At N=400 this is ~16 leaf DFTs of size 25
        // (~10k ops) plus butterfly work at four levels, versus 160k
        // ops for the old O(N²) direct-DFT path per frame.
        auto worker = [&](int tid) {
            std::vector<double> fft_in(2 * static_cast<size_t>(n_fft), 0.0);
            std::vector<double> fft_out(8 * static_cast<size_t>(n_fft), 0.0);
            for (int t = tid; t < n_frames; t += stft_threads) {
                const size_t start =
                    static_cast<size_t>(t) * static_cast<size_t>(hop);
                for (int n = 0; n < n_fft; ++n) {
                    fft_in[n] = padded[start + n] * window_[n];
                }
                mixed_radix_fft(fft_in.data(), n_fft, fft_out.data());
                float * pwr_row =
                    power.data() + static_cast<size_t>(t) * n_freq;
                for (int k = 0; k < n_freq; ++k) {
                    const double re = fft_out[2 * k    ];
                    const double im = fft_out[2 * k + 1];
                    pwr_row[k] = static_cast<float>(re * re + im * im);
                }
            }
        };
        run_threaded(worker);
    } else {
#ifdef __APPLE__
    // vDSP real-input FFT path (fp64, vectorized). Same precision as
    // the hand-rolled radix-2 — both are fp64 throughout — but ~5-10x
    // faster via Accelerate's SIMD kernels. Requires n_fft a power of 2.
    int log2n = 0;
    { int tmp = n_fft; while (tmp > 1) { tmp >>= 1; ++log2n; } }

    // A single FFTSetupD is shared across threads: it stores
    // precomputed twiddle factors, not transient state, and vDSP's
    // forward FFTs are thread-safe for concurrent use given per-thread
    // split-complex buffers.
    FFTSetupD fft_setup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);
    const size_t half_n = static_cast<size_t>(n_fft / 2);

    auto worker = [&](int tid) {
        std::vector<double> fft_real(half_n);
        std::vector<double> fft_imag(half_n);
        DSPDoubleSplitComplex split = { fft_real.data(), fft_imag.data() };

        for (int t = tid; t < n_frames; t += stft_threads) {
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
    };
    run_threaded(worker);

    vDSP_destroy_fftsetupD(fft_setup);
#else
    // Hand-rolled radix-2 Cooley-Tukey FFT fallback for non-Apple.
    auto worker = [&](int tid) {
        std::vector<double> frame(2 * static_cast<size_t>(n_fft));   // interleaved complex
        for (int t = tid; t < n_frames; t += stft_threads) {
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
    };
    run_threaded(worker);
#endif
    }  // end else (pow2 path)
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

    // Whisper clips the pre-log mel energy to 1e-10 (hard floor), while
    // NeMo adds a softer 2^-24 offset. The difference matters on silent
    // frames / unused mel bins where the raw energy is O(1e-9) or below.
    const bool   whisper_mode = (cfg_.normalize == "per_utterance");
    const double log_floor    = whisper_mode ? 1.0e-10 : kLogEps;

#if defined(__APPLE__) || TRANSCRIBE_HAS_BLAS
    // C = A @ B^T  where A=[n_mels, n_freq], B=[n_frames, n_freq].
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_mels, n_frames, n_freq,
                1.0f, mel_fb_.data(), n_freq,
                power.data(), n_freq,
                0.0f, log_mel.data(), n_frames);
    {
        const size_t total = static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames);
        if (whisper_mode) {
            for (size_t i = 0; i < total; ++i) {
                double v = static_cast<double>(log_mel[i]);
                if (v < log_floor) v = log_floor;
                log_mel[i] = static_cast<float>(std::log(v));
            }
        } else {
            for (size_t i = 0; i < total; ++i) {
                log_mel[i] = std::log(log_mel[i] + kLogEps);
            }
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
            const double v = whisper_mode
                ? std::max(static_cast<double>(acc), log_floor)
                : static_cast<double>(acc) + static_cast<double>(kLogEps);
            log_mel[static_cast<size_t>(m) * n_frames + t] =
                static_cast<float>(std::log(v));
        }
    }
#endif
    std::vector<float>().swap(power);

    // ---- 5a. Whisper-style per-utterance normalization ----
    // log_mel currently holds ln(x + eps) values (natural log). Whisper
    // uses log10 with a max(x, 1e-10) floor, then global clamp to
    // max - 8.0, then (x + 4) / 4. Dividing ln(x+eps) by ln(10) gives
    // log10(x+eps) which is numerically equivalent to log10(max(x, eps))
    // within the clamp region. Whisper also drops the trailing center-
    // pad STFT frame so the output has n_samples / hop_length frames.
    if (cfg_.normalize == "per_utterance") {
        const int n_out = n_frames - 1;
        if (n_out <= 0) return TRANSCRIBE_ERR_INVALID_ARG;

        const double inv_ln10 = 1.0 / std::log(10.0);

        double global_max = -std::numeric_limits<double>::infinity();
        for (int m = 0; m < n_mels; ++m) {
            const float * src =
                log_mel.data() + static_cast<size_t>(m) * n_frames;
            for (int t = 0; t < n_out; ++t) {
                const double v = static_cast<double>(src[t]) * inv_ln10;
                if (v > global_max) global_max = v;
            }
        }
        const double floor_val = global_max - 8.0;

        out_mel.resize(
            static_cast<size_t>(n_mels) * static_cast<size_t>(n_out));
        for (int m = 0; m < n_mels; ++m) {
            const float * src =
                log_mel.data() + static_cast<size_t>(m) * n_frames;
            float * dst = out_mel.data() + static_cast<size_t>(m) * n_out;
            for (int t = 0; t < n_out; ++t) {
                double v = static_cast<double>(src[t]) * inv_ln10;
                if (v < floor_val) v = floor_val;
                dst[t] = static_cast<float>((v + 4.0) / 4.0);
            }
        }

        out_n_mels   = n_mels;
        out_n_frames = n_out;
        return TRANSCRIBE_OK;
    }

    // ---- 5. Per-feature normalize, unbiased ----
    // For each mel bin m: subtract the mean across time, divide by
    // (std + 1e-5). The variance uses (n_norm - 1) in the
    // denominator (Bessel's correction) to match NeMo's
    // normalize_batch. fp64 accumulators throughout to keep the row
    // sums numerically stable; cast back to fp32 at storage.
    //
    // When pad_mode is "constant" (Cohere), the last STFT frame is
    // a padding artifact. The normalization statistics are computed
    // over the first seq_len = n_samples / hop frames (not n_frames
    // = seq_len + 1), and the last frame is zeroed after
    // normalization. This matches the Cohere reference
    // (processing_cohere_asr.py get_seq_len / normalize_batch).
    //
    // When pad_mode is "reflect" (Parakeet), all n_frames frames
    // are valid and used for normalization.
    //
    // resize() instead of assign(): the loop below writes every
    // element exactly once, so the zero-fill that assign() would do
    // is dead work.
    const bool mask_last = (cfg_.pad_mode == "constant");
    const int  n_norm    = mask_last ? (n_frames - 1) : n_frames;

    out_mel.resize(
        static_cast<size_t>(n_mels) * static_cast<size_t>(n_frames));
    for (int m = 0; m < n_mels; ++m) {
        const float * src =
            log_mel.data() + static_cast<size_t>(m) * n_frames;
        float * dst = out_mel.data() + static_cast<size_t>(m) * n_frames;

        double sum = 0.0;
        for (int t = 0; t < n_norm; ++t) {
            sum += static_cast<double>(src[t]);
        }
        const double mean = sum / static_cast<double>(n_norm);

        double sumsq = 0.0;
        for (int t = 0; t < n_norm; ++t) {
            const double d = static_cast<double>(src[t]) - mean;
            sumsq += d * d;
        }
        const double var    = sumsq / static_cast<double>(n_norm - 1);
        const double stddev = std::sqrt(var);
        const double inv    = 1.0 / (stddev + static_cast<double>(kNormEps));

        for (int t = 0; t < n_norm; ++t) {
            dst[t] = static_cast<float>(
                (static_cast<double>(src[t]) - mean) * inv);
        }

        // Zero the last frame when it's a padding artifact.
        if (mask_last) {
            dst[n_norm] = 0.0f;
        }
    }

    out_n_mels   = n_mels;
    out_n_frames = n_frames;
    return TRANSCRIBE_OK;
}

} // namespace transcribe
