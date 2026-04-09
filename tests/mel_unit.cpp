// mel_unit.cpp - unit tests for the precomputed buffers in
// transcribe::MelFrontend.
//
// Validates the constructor outputs (the symmetric-Hann window
// zero-padded to n_fft, and the librosa Slaney mel filterbank)
// in isolation, before the full pipeline runs. Catches the kinds of
// off-by-one bugs that would silently bias the entire frontend:
//
//   - periodic vs symmetric Hann (the most common subtle window bug)
//   - off-by-one in the win_length / n_fft zero-pad placement
//   - wrong mel scale formula (HTK vs Slaney)
//   - missing Slaney area normalization (would scale by ~50x)
//
// All reference values are bit-precise constants captured from
// librosa 0.11 and the symmetric-hann formula. Regenerate via the
// preflight script if librosa updates and the tolerances drift.

#include "transcribe-mel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

#define CHECK_NEAR(actual, expected, tol)                                   \
    do {                                                                    \
        const double _a = static_cast<double>(actual);                      \
        const double _e = static_cast<double>(expected);                    \
        const double _d = std::fabs(_a - _e);                               \
        if (_d > (tol)) {                                                   \
            std::fprintf(stderr,                                            \
                "FAIL %s:%d: %s = %.17g, expected %.17g, diff %.6g > %.6g\n", \
                __FILE__, __LINE__, #actual, _a, _e, _d, (tol));            \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

// Real Parakeet 0.6B frontend config (v2 + v3 share these values).
transcribe::MelConfig parakeet_config() {
    transcribe::MelConfig cfg;
    cfg.sample_rate  = 16000;
    cfg.num_mels     = 128;
    cfg.n_fft        = 512;
    cfg.win_length   = 400;
    cfg.hop_length   = 160;
    cfg.pre_emphasis = 0.97f;
    cfg.f_min        = 0.0f;
    cfg.f_max        = 8000.0f;
    return cfg;
}

void test_window() {
    transcribe::MelFrontend mf(parakeet_config());
    const auto & w = mf.window();

    CHECK(w.size() == 512);

    // First and last 56 entries are the zero pad: (n_fft - win_length) / 2
    // on each side. The 57th entry from each end is the first non-zero
    // sample of the symmetric Hann.
    for (int i = 0; i < 56; ++i) CHECK(w[i] == 0.0);
    for (int i = 0; i < 56; ++i) CHECK(w[511 - i] == 0.0);

    // Symmetric Hann formula 0.5 - 0.5*cos(2*pi*k/(N-1)) with N=400.
    // The boundary samples at the edge of the Hann are tiny but non-zero.
    // Reference values captured from numpy and librosa, machine-precision.
    CHECK_NEAR(w[56], 0.0,                                  1e-15);
    CHECK_NEAR(w[57], 6.199333200590518e-05,                1e-15);
    CHECK_NEAR(w[58], 0.0002479579553307798,                1e-15);
    CHECK_NEAR(w[59], 0.0005578477557081074,                1e-15);
    CHECK_NEAR(w[60], 0.0009915858887327156,                1e-15);

    // Peak region. For a symmetric N=400 Hann the peak straddles indices
    // 199 and 200 (in window space) = 255 and 256 in the padded buffer,
    // both equal to ~0.99998 (NOT 1.0 - that would be the periodic form).
    CHECK_NEAR(w[254], 0.9998605186060137,                  1e-15);
    CHECK_NEAR(w[255], 0.9999845014267927,                  1e-15);
    CHECK_NEAR(w[256], 0.9999845014267927,                  1e-15);
    CHECK_NEAR(w[257], 0.9998605186060137,                  1e-15);
    CHECK_NEAR(w[258], 0.9996125837088883,                  1e-15);

    // Sum of a symmetric Hann window of length 400 is exactly
    // (N - 1) / 2 = 199.5. Off by anything > a few ULPs means the
    // formula is wrong.
    double sum = 0.0;
    for (double v : w) sum += v;
    CHECK_NEAR(sum, 199.5, 1e-12);

    // Sum of squares: closed form for a symmetric Hann length N is
    // (3N - 4) / 8. For N=400: (1200 - 4) / 8 = 149.5. But the
    // librosa-style symmetric form (k / (N-1)) gives a slightly
    // different value; reference captured from numpy directly:
    double sumsq = 0.0;
    for (double v : w) sumsq += v * v;
    CHECK_NEAR(sumsq, 149.625, 1e-12);
}

void test_mel_filterbank() {
    transcribe::MelFrontend mf(parakeet_config());
    const auto & fb = mf.filterbank();
    const int n_freq = mf.n_freq();
    const int n_mels = mf.num_mels();

    CHECK(static_cast<int>(fb.size()) == n_mels * n_freq);
    CHECK(n_freq == 257);
    CHECK(n_mels == 128);

    auto get = [&](int m, int k) -> float {
        return fb[static_cast<size_t>(m) * n_freq + k];
    };

    // Total filterbank sum. Captured from librosa 0.11 with the
    // exact (sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000,
    // norm='slaney') call.
    double total = 0.0;
    for (float v : fb) total += static_cast<double>(v);
    CHECK_NEAR(total, 4.090487480163574, 5e-6);

    // Per-row spot checks. Slaney filters are sparse at low mel
    // bins (some span only 1-3 freq bins) and progressively wider
    // at the top.

    // Mel bin 0: only fft bin 1 has a nonzero weight.
    for (int k = 0; k < n_freq; ++k) {
        if (k == 1) {
            CHECK_NEAR(get(0, k), 0.02837754227221012, 5e-7);
        } else {
            CHECK(get(0, k) == 0.0f);
        }
    }

    // Mel bin 5: triangular filter spanning fft bins 4 and 5.
    CHECK_NEAR(get(5, 4),  0.014789481647312641, 5e-7);
    CHECK_NEAR(get(5, 5),  0.013588061556220055, 5e-7);
    CHECK(get(5, 3) == 0.0f);
    CHECK(get(5, 6) == 0.0f);

    // Mel bin 64: middle of the bank.
    CHECK_NEAR(get(64, 55), 0.018818283453584, 5e-7);

    // Mel bin 100: 6 nonzero bins centered at fft bin 130.
    CHECK_NEAR(get(100, 130), 0.009136910550296, 5e-7);

    // Mel bin 127 (top): widest filter, peak at fft bin 250.
    CHECK_NEAR(get(127, 250), 0.005223188549280, 5e-7);

    // Per-row sums (Slaney area-normalized triangles, so each row's
    // sum is 2/(width) * triangle area = roughly constant per
    // octave; reference values from librosa).
    double row_0_sum = 0.0;
    double row_64_sum = 0.0;
    double row_127_sum = 0.0;
    for (int k = 0; k < n_freq; ++k) {
        row_0_sum   += get(0, k);
        row_64_sum  += get(64, k);
        row_127_sum += get(127, k);
    }
    CHECK_NEAR(row_0_sum,   0.028377542272210, 5e-7);
    CHECK_NEAR(row_64_sum,  0.030684133991599, 5e-6);
    CHECK_NEAR(row_127_sum, 0.031943686306477, 5e-6);

    // Total nonzero count: librosa Slaney with these params gives
    // exactly 504 non-zero entries out of 32896. A wrong formula
    // would either widen the filters (more nonzeros) or break the
    // mel scale (different distribution).
    int nonzero = 0;
    for (float v : fb) if (v != 0.0f) ++nonzero;
    CHECK(nonzero == 504);
}

void test_n_frames_for() {
    transcribe::MelFrontend mf(parakeet_config());
    // jfk.wav is 176000 samples = 11 s @ 16 kHz.
    // n_frames = 176000 / 160 + 1 = 1101.
    CHECK(mf.n_frames_for(176000) == 1101);
    // Another spot check.
    CHECK(mf.n_frames_for(16000) == 101);
    // Empty audio still returns the +1.
    CHECK(mf.n_frames_for(0) == 1);
}

} // namespace

int main() {
    test_window();
    test_mel_filterbank();
    test_n_frames_for();

    if (g_failures > 0) {
        std::fprintf(stderr, "mel_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "mel_unit: ok\n");
    return EXIT_SUCCESS;
}
