// parakeet_chunked_limited_with_rc_mask_unit.cpp - unit test for
// transcribe::parakeet::compute_chunked_limited_with_rc_mask.
//
// The mask drives the chunked-attention used by buffered streaming on
// parakeet-unified-en-0.6b. It MUST match NeMo's
// ConformerEncoder._create_masks output for the chunked_limited_with_rc
// branch (refs/NVIDIA-NeMo/.../conformer_encoder.py:843-869) so cpp
// streaming numerics match the reference framework byte-for-byte.
//
// Strategy: implement the NeMo formula inline as a reference, run both
// for representative (T, L, C, R) tuples, and compare element-by-element.
// The reference path uses the same integer arithmetic NeMo uses (clamp
// to [0, T-1]); the only freedom is which sentinel value represents
// "masked" (NeMo uses bool; we emit float -INF added onto matrix_bd).

#include "arch/parakeet/parakeet.h"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Reference implementation mirroring NeMo's _create_masks branch for
// chunked_limited_with_rc. Returns a flat row-major [T*T] bool buffer
// where true = "allowed to attend".
std::vector<bool> reference_mask(int T, int L, int C, int R, int P) {
    std::vector<bool> mask(static_cast<size_t>(T) * static_cast<size_t>(T), false);
    const int         pad = P > T ? T : P;
    for (int q = 0; q < T; ++q) {
        const int  c_q          = q / C;
        const int  ws_unclamped = c_q * C - L;
        const int  we_unclamped = c_q * C + C - 1 + R;
        const int  ws           = ws_unclamped > 0 ? ws_unclamped : 0;
        const int  we           = we_unclamped < (T - 1) ? we_unclamped : (T - 1);
        const bool q_padded     = q >= pad;
        for (int k = 0; k < T; ++k) {
            const bool k_padded                  = k >= pad;
            mask[static_cast<size_t>(q) * T + k] = (k >= ws && k <= we) && !q_padded && !k_padded;
        }
    }
    return mask;
}

void run_case(const char * name, int T, int L, int C, int R, int P = INT_MAX) {
    std::vector<float> got(static_cast<size_t>(T) * static_cast<size_t>(T));
    transcribe::parakeet::compute_chunked_limited_with_rc_mask(got.data(), T, L, C, R, P);

    const std::vector<bool> ref = reference_mask(T, L, C, R, P);

    int mismatches = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const bool  allowed_ref = ref[i];
        const float cell        = got[i];
        const bool  allowed_got = (cell == 0.0f);
        const bool  masked_got  = std::isinf(cell) && cell < 0.0f;
        const bool  well_formed = allowed_got || masked_got;
        if (!well_formed) {
            ++mismatches;
            if (mismatches <= 4) {
                const int q = static_cast<int>(i / T);
                const int k = static_cast<int>(i % T);
                std::fprintf(stderr, "  [%s] mask[%d, %d] = %f (expected 0.0 or -INF)\n", name, q, k, cell);
            }
            continue;
        }
        if (allowed_ref != allowed_got) {
            ++mismatches;
            if (mismatches <= 4) {
                const int q = static_cast<int>(i / T);
                const int k = static_cast<int>(i % T);
                std::fprintf(stderr, "  [%s] mask[%d, %d] = %s but ref says %s\n", name, q, k,
                             allowed_got ? "0.0 (allowed)" : "-INF (masked)", allowed_ref ? "allowed" : "masked");
            }
        }
    }
    if (mismatches == 0) {
        std::fprintf(stderr, "  [%s] OK T=%d L=%d C=%d R=%d\n", name, T, L, C, R);
    } else {
        std::fprintf(stderr, "  [%s] FAIL T=%d L=%d C=%d R=%d — %d mismatching cells\n", name, T, L, C, R, mismatches);
        ++g_failures;
    }
}

}  // namespace

int main() {
    // Parakeet-unified-en-0.6b's full training menu (frames at 80ms).
    // L ∈ {70}, C ∈ {1, 2, 7, 13}, R ∈ {0, 1, 2, 3, 4, 7, 13}. Test a
    // representative cross-section: the best-accuracy default
    // (70, 13, 13), the fastest (70, 1, 0), and a mid-tier
    // (70, 7, 7). T values cover both "smaller than a chunk" and
    // "many chunks" regimes.
    run_case("default-1.04s", 256, /*L=*/70, /*C=*/13, /*R=*/13);
    run_case("default-1.04s-small", 10, /*L=*/70, /*C=*/13, /*R=*/13);  // T < C: single chunk, hits clamps
    run_case("fastest-80ms", 256, 70, 1, 0);
    // R=1 specifically: the WER sweep showed (70,1,1) is a divergence
    // outlier vs the rest of the menu, so cover it explicitly.
    run_case("low-lat-160ms", 256, 70, 1, 1);
    run_case("low-lat-320ms", 256, 70, 2, 2);
    run_case("midtier-560ms", 256, 70, 7, 7);

    // L=0, R=0, C=1 — strictly diagonal (each frame sees only itself).
    run_case("strictly-diag", 16, 0, 1, 0);

    // L=T-1, R=T-1 — full attention (every (q, k) allowed).
    run_case("full-attention", 32, 31, 4, 31);

    // Edge: chunk straddles the end of T. Make sure clamping works.
    run_case("ragged-tail", 11, 70, 4, 4);

    // Pad-mask: conv-overhang case. T_enc = 35 frames produced, but
    // effective_T = 34 (one trailing frame is conv padding overhang).
    // The trailing frame must be masked out — both as a query (whole
    // row -INF) and as a key (whole column -INF). Mirrors NeMo's
    // pad_mask AND chunked_mask. Critical at low-C/low-R configs
    // where the leak tips greedy decisions.
    run_case("pad-overhang-low-lat", 35, 70, 1, 1, /*P=*/34);
    run_case("pad-overhang-default", 72, 70, 13, 13, /*P=*/72);  // no pad
    run_case("pad-zero", 5, 70, 1, 0, /*P=*/0);                  // all masked

    if (g_failures > 0) {
        std::fprintf(stderr, "FAILED: %d cases\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "OK\n");
    return EXIT_SUCCESS;
}
