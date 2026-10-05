// ecapa_tdnn_mel_unit.cpp - hermetic unit tests for the ecapa_tdnn
// (SpeechBrain) log-mel front end, transcribe::ecapa_tdnn::MelFrontend.
//
// Every expected value in this file is either a closed-form constant of
// the periodic-Hamming / STFT math or is computed in the test itself at
// fp64. Nothing is read from disk and no model is required, so this test
// runs everywhere ctest does. The end-to-end comparison against
// SpeechBrain lives in tests/mel_parity_tool.cpp + scripts/validate.py.
//
// What each test pins down (i.e. the bug it would catch):
//
//   test_window            periodic vs symmetric Hamming — the classic
//                          off-by-one that biases every frame slightly.
//   test_n_frames_for      torch center=True framing: floor(n/hop) + 1,
//                          no trailing-frame drop.
//   test_fft               the lifted mixed-radix FFT against a naive
//                          O(N^2) fp64 DFT: wrong twiddle stride, a bad
//                          odd-leaf fallback, or a scratch-buffer alias.
//   test_pipeline_dc       window -> FFT -> power -> filterbank -> log
//                          wired together with the right scale factor.
//   test_top_db            the dynamic-range floor, and that it is taken
//                          over time AND frequency.
//   test_sentence_mean     per-bin mean subtraction over all frames.
//   test_thread_invariance bit-identical output for any thread count.
//   test_errors            config and argument rejection.

#include "arch/ecapa_tdnn/mel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(actual, expected, tol)                                                                          \
    do {                                                                                                           \
        const double _a = static_cast<double>(actual);                                                             \
        const double _e = static_cast<double>(expected);                                                           \
        const double _d = std::fabs(_a - _e);                                                                      \
        if (!(_d <= (tol))) {                                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s = %.17g, expected %.17g, diff %.6g > %.6g\n", __FILE__, __LINE__, \
                         #actual, _a, _e, _d, (tol));                                                              \
            ++g_failures;                                                                                          \
        }                                                                                                          \
    } while (0)

constexpr int kNFft  = 400;
constexpr int kNFreq = 201;
constexpr int kNMels = 60;

// A synthetic "one-hot" filterbank: mel bin m reads FFT bin 3*m with
// weight 1 and nothing else (3*59 = 177 < 201). This turns the mel matmul
// into an identity on a subset of the power spectrum, so the pipeline
// tests can assert exact closed-form dB values instead of a filterbank
// convolution. The real SpeechBrain filterbank is never rebuilt in C++;
// it comes from the GGUF and is validated by the parity tool.
std::vector<float> one_hot_filterbank() {
    std::vector<float> fb(static_cast<size_t>(kNMels) * kNFreq, 0.0f);
    for (int m = 0; m < kNMels; ++m) {
        fb[static_cast<size_t>(m) * kNFreq + static_cast<size_t>(m) * 3] = 1.0f;
    }
    return fb;
}

// The real VoxLingua107 front-end config, with a caller-supplied
// filterbank (the tests use the one-hot bank above).
transcribe::ecapa_tdnn::MelConfig base_config() {
    transcribe::ecapa_tdnn::MelConfig cfg;
    cfg.sample_rate = 16000;
    cfg.n_mels      = kNMels;
    cfg.n_fft       = kNFft;
    cfg.win_length  = kNFft;
    cfg.hop_length  = 160;
    cfg.filterbank  = one_hot_filterbank();
    return cfg;
}

void test_window() {
    transcribe::ecapa_tdnn::MelFrontend mf(base_config());
    CHECK(mf.status() == TRANSCRIBE_OK);

    const auto & w = mf.window();
    CHECK(w.size() == static_cast<size_t>(kNFft));
    if (w.size() != static_cast<size_t>(kNFft)) {
        return;
    }

    // Periodic Hamming, w[n] = 0.54 - 0.46*cos(2*pi*n/400):
    //   n = 0   -> 0.54 - 0.46          = 0.08
    //   n = 100 -> 0.54 - 0.46*cos(pi/2)= 0.54
    //   n = 200 -> 0.54 - 0.46*cos(pi)  = 1.00   (the peak is ON a sample;
    //                                             a SYMMETRIC Hamming of
    //                                             length 400 never reaches
    //                                             1.0 — that is the bug
    //                                             this line catches)
    //   n = 300 -> 0.54
    CHECK_NEAR(w[0], 0.08, 1e-6);
    CHECK_NEAR(w[100], 0.54, 1e-6);
    CHECK_NEAR(w[200], 1.0, 1e-6);
    CHECK_NEAR(w[300], 0.54, 1e-6);

    // w[1] has no closed form; compute the literal at fp64 here so the
    // test pins the exact formula (2*pi*n/N, not 2*pi*n/(N-1)).
    const double w1 = 0.54 - 0.46 * std::cos(2.0 * M_PI * 1.0 / 400.0);
    CHECK_NEAR(w[1], w1, 1e-6);
    // Cross-check the fp64 expression against its own decimal expansion,
    // so a bad M_PI or a mistyped denominator cannot make both sides
    // wrong in the same way.
    CHECK_NEAR(w1, 0.0800567490584361, 1e-15);

    // Sum over a full period: the cosine term cancels exactly, leaving
    // 0.54 * N = 216. The measured value is 216.00000069 (400 fp32
    // roundings of the fp64 window), comfortably inside 1e-6.
    double sum = 0.0;
    for (float v : w) {
        sum += static_cast<double>(v);
    }
    CHECK_NEAR(sum, 216.0, 1e-6);
}

void test_n_frames_for() {
    transcribe::ecapa_tdnn::MelFrontend mf(base_config());

    // torch center=True: floor(n / hop) + 1, trailing frames kept.
    CHECK(mf.n_frames_for(16000) == 101);  // 1 s
    CHECK(mf.n_frames_for(8000) == 51);    // 0.5 s
    CHECK(mf.n_frames_for(16159) == 101);  // one sample short of the next hop
    CHECK(mf.n_frames_for(16160) == 102);  // exactly on the next hop
    CHECK(mf.n_frames_for(1) == 1);        // a single sample still frames
}

// Straightforward O(N^2) DFT at fp64. Independent of the production LUT
// and recursion: this is the oracle the mixed-radix FFT is checked
// against.
void naive_dft(const float * in, int N, std::vector<double> & re, std::vector<double> & im) {
    re.assign(static_cast<size_t>(N / 2 + 1), 0.0);
    im.assign(static_cast<size_t>(N / 2 + 1), 0.0);
    for (int k = 0; k <= N / 2; ++k) {
        double sr = 0.0;
        double si = 0.0;
        for (int n = 0; n < N; ++n) {
            const double theta = 2.0 * M_PI * static_cast<double>(k) * static_cast<double>(n) / static_cast<double>(N);
            sr += static_cast<double>(in[n]) * std::cos(theta);
            si -= static_cast<double>(in[n]) * std::sin(theta);
        }
        re[static_cast<size_t>(k)] = sr;
        im[static_cast<size_t>(k)] = si;
    }
}

void test_fft() {
    std::mt19937                          rng(20240902u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int trial = 0; trial < 3; ++trial) {
        std::vector<float> in(static_cast<size_t>(kNFft));
        for (int i = 0; i < kNFft; ++i) {
            in[static_cast<size_t>(i)] = dist(rng);
        }

        std::vector<float> got_re(static_cast<size_t>(kNFreq), 0.0f);
        std::vector<float> got_im(static_cast<size_t>(kNFreq), 0.0f);
        transcribe::ecapa_tdnn::mel_test_fft(in.data(), got_re.data(), got_im.data());

        std::vector<double> ref_re;
        std::vector<double> ref_im;
        naive_dft(in.data(), kNFft, ref_re, ref_im);

        double max_mag = 0.0;
        double max_err = 0.0;
        for (int k = 0; k < kNFreq; ++k) {
            const double mag = std::hypot(ref_re[static_cast<size_t>(k)], ref_im[static_cast<size_t>(k)]);
            max_mag          = std::max(max_mag, mag);
            max_err          = std::max(max_err, std::fabs(static_cast<double>(got_re[static_cast<size_t>(k)]) -
                                                           ref_re[static_cast<size_t>(k)]));
            max_err          = std::max(max_err, std::fabs(static_cast<double>(got_im[static_cast<size_t>(k)]) -
                                                           ref_im[static_cast<size_t>(k)]));
        }
        // fp32 FFT vs fp64 DFT: relative to the peak magnitude the
        // measured error is ~1e-7. 1e-6 keeps 10x headroom for a
        // different libm while still catching a transform that has gone
        // meaningfully wrong (a mistyped twiddle stride, a bad odd-leaf
        // fallback); a looser bound here would guard nothing.
        CHECK(max_mag > 1.0);
        CHECK_NEAR(max_err / max_mag, 0.0, 1e-6);
    }
}

void test_pipeline_dc() {
    // Constant 1.0 for 1 s. For any interior frame the windowed signal is
    // just the window, so the DC bin of the DFT is sum(w) = 216 and the
    // DC power is 216^2 = 46656 -> 10*log10(46656) = 46.68907502 dB.
    //
    // A periodic Hamming has an exactly 3-tap DFT (bins 0, 1 and N-1), so
    // every other bin is exactly zero in exact arithmetic. The one-hot
    // filterbank reads bins 3, 6, ..., 177 for mel bins 1..59, all of
    // which are true zeros: they must land on the log floor,
    // 10*log10(1e-10) = -100. (Measured fp32 leakage tops out at
    // -105.4 dB, i.e. 4.6 dB of margin under the floor.)
    transcribe::ecapa_tdnn::MelConfig cfg = base_config();
    cfg.normalize                         = "none";
    cfg.top_db                            = 0.0f;  // disabled

    transcribe::ecapa_tdnn::MelFrontend mf(cfg);
    CHECK(mf.status() == TRANSCRIBE_OK);

    std::vector<float> pcm(16000, 1.0f);
    std::vector<float> out;
    int                n_frames = 0;
    CHECK(mf.compute(pcm.data(), 16000, out, n_frames, 1) == TRANSCRIBE_OK);
    CHECK(n_frames == 101);
    CHECK(out.size() == static_cast<size_t>(101) * kNMels);
    if (out.size() != static_cast<size_t>(101) * kNMels) {
        return;
    }

    const float * frame50 = out.data() + static_cast<size_t>(50) * kNMels;
    CHECK_NEAR(frame50[0], 46.68907502, 1e-3);
    for (int m = 1; m < kNMels; ++m) {
        CHECK_NEAR(frame50[m], -100.0, 1e-3);
    }
}

void test_top_db() {
    // Same input, top_db enabled. The global maximum over time and
    // frequency is the interior-frame DC value 46.68907502, so the floor
    // is 46.68907502 - 80 = -33.31092498 and every -100 dB bin rises to
    // it.
    transcribe::ecapa_tdnn::MelConfig cfg = base_config();
    cfg.normalize                         = "none";
    cfg.top_db                            = 80.0f;

    transcribe::ecapa_tdnn::MelFrontend mf(cfg);
    CHECK(mf.status() == TRANSCRIBE_OK);

    std::vector<float> pcm(16000, 1.0f);
    std::vector<float> out;
    int                n_frames = 0;
    CHECK(mf.compute(pcm.data(), 16000, out, n_frames, 1) == TRANSCRIBE_OK);
    CHECK(n_frames == 101);
    if (out.size() != static_cast<size_t>(101) * kNMels) {
        return;
    }

    const float * frame50 = out.data() + static_cast<size_t>(50) * kNMels;
    CHECK_NEAR(frame50[0], 46.68907502, 1e-3);
    for (int m = 1; m < kNMels; ++m) {
        CHECK_NEAR(frame50[m], -33.31092498, 1e-3);
    }

    // Nothing anywhere may sit below the floor.
    for (float v : out) {
        CHECK(static_cast<double>(v) >= -33.31092498 - 1e-3);
    }
}

void test_sentence_mean() {
    // 1 kHz sine, 1 s. After sentence-mean normalisation every mel bin's
    // mean over frames is zero by construction — this checks the pass runs
    // over the right axis (per bin over frames, not per frame over bins).
    transcribe::ecapa_tdnn::MelConfig cfg = base_config();
    cfg.normalize                         = "sentence_mean";

    transcribe::ecapa_tdnn::MelFrontend mf(cfg);
    CHECK(mf.status() == TRANSCRIBE_OK);

    std::vector<float> pcm(16000);
    for (int i = 0; i < 16000; ++i) {
        pcm[static_cast<size_t>(i)] = static_cast<float>(std::sin(2.0 * M_PI * 1000.0 * i / 16000.0));
    }

    std::vector<float> out;
    int                n_frames = 0;
    CHECK(mf.compute(pcm.data(), 16000, out, n_frames, 1) == TRANSCRIBE_OK);
    CHECK(n_frames == 101);
    if (out.size() != static_cast<size_t>(n_frames) * kNMels) {
        return;
    }

    for (int m = 0; m < kNMels; ++m) {
        double sum = 0.0;
        for (int t = 0; t < n_frames; ++t) {
            sum += static_cast<double>(out[static_cast<size_t>(t) * kNMels + static_cast<size_t>(m)]);
        }
        CHECK_NEAR(sum / n_frames, 0.0, 1e-4);
    }

    // A 1 kHz tone must not be flat: if the mean pass had subtracted the
    // wrong axis the per-frame spread would collapse.
    double spread = 0.0;
    for (float v : out) {
        spread = std::max(spread, std::fabs(static_cast<double>(v)));
    }
    CHECK(spread > 1.0);

    // Degenerate T=1 (fewer samples than one hop): the single frame is
    // its own mean, so the output is all zeros. No divide-by-zero, unlike
    // a variance-normalising front end.
    std::vector<float> tiny_out;
    int                tiny_frames = 0;
    CHECK(mf.compute(pcm.data(), 100, tiny_out, tiny_frames, 1) == TRANSCRIBE_OK);
    CHECK(tiny_frames == 1);
    CHECK(tiny_out.size() == static_cast<size_t>(kNMels));
    for (float v : tiny_out) {
        CHECK(v == 0.0f);
    }
}

void test_thread_invariance() {
    // 3 s of fixed-seed noise, computed single-threaded and with 4
    // threads. Every frame is written by exactly one thread into its own
    // output slot and the log / top-dB / mean passes are single-threaded,
    // so the results must be bit-identical, not merely close.
    transcribe::ecapa_tdnn::MelFrontend mf(base_config());
    CHECK(mf.status() == TRANSCRIBE_OK);

    std::mt19937                          rng(987654321u);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float>                    pcm(48000);
    for (auto & v : pcm) {
        v = dist(rng);
    }

    std::vector<float> out1;
    std::vector<float> out4;
    int                frames1 = 0;
    int                frames4 = 0;
    CHECK(mf.compute(pcm.data(), 48000, out1, frames1, 1) == TRANSCRIBE_OK);
    CHECK(mf.compute(pcm.data(), 48000, out4, frames4, 4) == TRANSCRIBE_OK);
    CHECK(frames1 == 301);
    CHECK(frames1 == frames4);
    CHECK(out1.size() == out4.size());
    if (out1.size() == out4.size() && !out1.empty()) {
        CHECK(std::memcmp(out1.data(), out4.data(), out1.size() * sizeof(float)) == 0);
    }
}

void test_errors() {
    transcribe::ecapa_tdnn::MelFrontend mf(base_config());
    std::vector<float>                  out;
    int                                 n_frames = 0;
    std::vector<float>                  pcm(16000, 0.1f);

    CHECK(mf.compute(nullptr, 16000, out, n_frames, 1) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(mf.compute(pcm.data(), 0, out, n_frames, 1) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(mf.compute(pcm.data(), -5, out, n_frames, 1) == TRANSCRIBE_ERR_INVALID_ARG);
    // One sample is legal: the 500 ms minimum is an API-layer rule, not a
    // front-end one.
    CHECK(mf.compute(pcm.data(), 1, out, n_frames, 1) == TRANSCRIBE_OK);
    CHECK(n_frames == 1);

    // Rejected configs. Each must set status() AND make compute() fail
    // without touching the audio.
    auto expect_rejected = [&](transcribe::ecapa_tdnn::MelConfig cfg, const char * what) {
        transcribe::ecapa_tdnn::MelFrontend bad(cfg);
        if (bad.status() != TRANSCRIBE_ERR_GGUF) {
            std::fprintf(stderr, "FAIL %s:%d: config '%s' was accepted\n", __FILE__, __LINE__, what);
            ++g_failures;
            return;
        }
        std::vector<float> bad_out;
        int                bad_frames = 0;
        CHECK(bad.compute(pcm.data(), 16000, bad_out, bad_frames, 1) == TRANSCRIBE_ERR_GGUF);
    };

    {
        transcribe::ecapa_tdnn::MelConfig cfg = base_config();
        cfg.window_type                       = "hann_periodic";
        expect_rejected(cfg, "window_type");
    }
    {
        transcribe::ecapa_tdnn::MelConfig cfg = base_config();
        cfg.pad_mode                          = "reflect";
        expect_rejected(cfg, "pad_mode");
    }
    {
        transcribe::ecapa_tdnn::MelConfig cfg = base_config();
        cfg.normalize                         = "per_feature";
        expect_rejected(cfg, "normalize");
    }
    {
        transcribe::ecapa_tdnn::MelConfig cfg = base_config();
        cfg.win_length                        = 320;
        expect_rejected(cfg, "win_length != n_fft");
    }
    {
        transcribe::ecapa_tdnn::MelConfig cfg = base_config();
        cfg.filterbank.clear();
        expect_rejected(cfg, "empty filterbank");
    }
    {
        transcribe::ecapa_tdnn::MelConfig cfg = base_config();
        cfg.filterbank.resize(cfg.filterbank.size() - 1);
        expect_rejected(cfg, "short filterbank");
    }
    {
        transcribe::ecapa_tdnn::MelConfig cfg = base_config();
        cfg.n_fft                             = 401;
        cfg.win_length                        = 401;
        expect_rejected(cfg, "odd n_fft");
    }
    {
        transcribe::ecapa_tdnn::MelConfig cfg = base_config();
        cfg.hop_length                        = 0;
        expect_rejected(cfg, "hop_length");
    }
    {
        transcribe::ecapa_tdnn::MelConfig cfg = base_config();
        cfg.n_mels                            = 0;
        expect_rejected(cfg, "n_mels");
    }
}

}  // namespace

int main() {
    test_window();
    test_n_frames_for();
    test_fft();
    test_pipeline_dc();
    test_top_db();
    test_sentence_mean();
    test_thread_invariance();
    test_errors();

    if (g_failures > 0) {
        std::fprintf(stderr, "mel_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "mel_unit: ok\n");
    return EXIT_SUCCESS;
}
