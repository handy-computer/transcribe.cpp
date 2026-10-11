// parakeet_chunked_limited_window_mask_unit.cpp - equivalence test for the
// compact offline ChunkedLimited attention mask.

#include "arch/parakeet/parakeet.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

bool is_masked(float value) {
    return std::isinf(value) && value < 0.0f;
}

void fail_case(const char * reason, int T, int C, int left_chunks, int q, int k) {
    if (g_failures < 12) {
        std::fprintf(stderr, "FAIL %s: T=%d C=%d left_chunks=%d q=%d k=%d\n", reason, T, C, left_chunks, q, k);
    }
    ++g_failures;
}

void run_case(int T, int C, int left_chunks) {
    const int W = (left_chunks + 1) * C;
    const int N = (T + C - 1) / C;

    // The graph pads T to N*C before im2col. With symmetric left padding,
    // its first N windows must exist and begin at exactly the same absolute
    // key positions as the compact host mask.
    const int T_pad          = N * C;
    const int left_pad       = left_chunks * C;
    const int im2col_windows = (T_pad + 2 * left_pad - W) / C + 1;
    if (im2col_windows < N) {
        fail_case("im2col window count", T, C, left_chunks, -1, im2col_windows);
    }

    std::vector<float> compact(static_cast<size_t>(W) * C * N);
    transcribe::parakeet::compute_chunked_limited_window_mask(compact.data(), T, T, C, left_chunks);

    for (int n = 0; n < N; ++n) {
        const int key_start = (n - left_chunks) * C;
        if (n * C - left_pad != key_start) {
            fail_case("im2col window origin", T, C, left_chunks, n, key_start);
        }
        for (int q_local = 0; q_local < C; ++q_local) {
            const int     q_abs = n * C + q_local;
            const float * row   = compact.data() + (static_cast<size_t>(n) * C + q_local) * W;

            if (q_abs >= T) {
                // Padded queries are discarded, but must contain exactly one
                // finite score so softmax never sees an all-masked row.
                for (int k_local = 0; k_local < W; ++k_local) {
                    const bool expected_finite = k_local == 0;
                    const bool got_finite      = row[k_local] == 0.0f;
                    if (expected_finite != got_finite || (!got_finite && !is_masked(row[k_local]))) {
                        fail_case("padded query sentinel", T, C, left_chunks, q_abs, k_local);
                    }
                }
                continue;
            }

            // Reconstruct every dense [q,k] cell from the compact row and
            // compare it with the original chunk-index formula.
            for (int k_abs = 0; k_abs < T; ++k_abs) {
                const int  k_local         = k_abs - key_start;
                const bool compact_allowed = k_local >= 0 && k_local < W && row[k_local] == 0.0f;
                const int  k_chunk         = k_abs / C;
                const bool dense_allowed   = k_chunk >= (n - left_chunks > 0 ? n - left_chunks : 0) && k_chunk <= n;
                if (compact_allowed != dense_allowed) {
                    fail_case("dense equivalence", T, C, left_chunks, q_abs, k_abs);
                }
            }

            for (int k_local = 0; k_local < W; ++k_local) {
                const int  k_abs           = key_start + k_local;
                const bool expected_finite = k_abs >= 0 && k_abs < T;
                const bool got_finite      = row[k_local] == 0.0f;
                if (expected_finite != got_finite || (!got_finite && !is_masked(row[k_local]))) {
                    fail_case("window padding", T, C, left_chunks, q_abs, k_abs);
                }

                // The shortened positional table must address the same
                // q_abs-k_abs offset as the dense 2*T-1 table. rel_shift
                // selects source row k_local-q_local+C-1.
                const int source_row    = k_local - q_local + C - 1;
                const int zero_row      = W - 1;
                const int compact_delta = zero_row - source_row;
                if (compact_delta != q_abs - k_abs) {
                    fail_case("relative position", T, C, left_chunks, q_abs, k_abs);
                }
            }
        }
    }
}

// Regular-local band: the compact [W, C, N] mask must reproduce the dense
// "q - left <= k <= q + right" rule and address the same q-k offsets.
void run_local_case(int T, int C, int left, int right) {
    const int W = left + C + right;
    const int N = (T + C - 1) / C;

    std::vector<float> compact(static_cast<size_t>(W) * C * N);
    transcribe::parakeet::compute_local_window_mask(compact.data(), T, T, C, left, right);

    for (int n = 0; n < N; ++n) {
        const int key_start = n * C - left;
        for (int q_local = 0; q_local < C; ++q_local) {
            const int     q_abs = n * C + q_local;
            const float * row   = compact.data() + (static_cast<size_t>(n) * C + q_local) * W;
            if (q_abs >= T) {
                for (int k_local = 0; k_local < W; ++k_local) {
                    const bool got_finite = row[k_local] == 0.0f;
                    if (got_finite != (k_local == 0) || (!got_finite && !is_masked(row[k_local]))) {
                        fail_case("local padded query sentinel", T, C, left * 1000 + right, q_abs, k_local);
                    }
                }
                continue;
            }
            for (int k_abs = 0; k_abs < T; ++k_abs) {
                const int  k_local         = k_abs - key_start;
                const bool compact_allowed = k_local >= 0 && k_local < W && row[k_local] == 0.0f;
                const bool dense_allowed   = (k_abs >= q_abs - left) && (k_abs <= q_abs + right);
                if (compact_allowed != dense_allowed) {
                    fail_case("local dense equivalence", T, C, left * 1000 + right, q_abs, k_abs);
                }
            }
            for (int k_local = 0; k_local < W; ++k_local) {
                const int k_abs = key_start + k_local;
                if (row[k_local] == 0.0f && (k_abs < 0 || k_abs >= T)) {
                    fail_case("local edge padding", T, C, left * 1000 + right, q_abs, k_abs);
                }
                // pos table: rel_shift source row k_local - q_local + C - 1,
                // zero row left + C - 1, must address offset q_abs - k_abs.
                const int source_row = k_local - q_local + C - 1;
                const int zero_row   = left + C - 1;
                if (zero_row - source_row != q_abs - k_abs) {
                    fail_case("local relative position", T, C, left * 1000 + right, q_abs, k_abs);
                }
            }
        }
    }
}

// Batched geometry: an utterance of T valid frames inside a graph padded to
// T_alloc. Blocks covering its valid frames must match the unpadded mask of
// the same utterance exactly; everything at or past T is a padded key
// (-INF) or a padded query (sentinel row), never a valid cell.
void run_padded_case(int T, int T_alloc, int C, int left, int right, bool chunked) {
    const int W     = chunked ? (left + 1) * C : left + C + right;
    const int N     = (T_alloc + C - 1) / C;
    const int N_own = (T + C - 1) / C;

    std::vector<float> padded(static_cast<size_t>(W) * C * N);
    std::vector<float> own(static_cast<size_t>(W) * C * N_own);
    if (chunked) {
        transcribe::parakeet::compute_chunked_limited_window_mask(padded.data(), T, T_alloc, C, left);
        transcribe::parakeet::compute_chunked_limited_window_mask(own.data(), T, T, C, left);
    } else {
        transcribe::parakeet::compute_local_window_mask(padded.data(), T, T_alloc, C, left, right);
        transcribe::parakeet::compute_local_window_mask(own.data(), T, T, C, left, right);
    }
    const int tag = chunked ? -left : left * 1000 + right;
    for (int n = 0; n < N; ++n) {
        for (int q_local = 0; q_local < C; ++q_local) {
            const int     q_abs = n * C + q_local;
            const float * row   = padded.data() + (static_cast<size_t>(n) * C + q_local) * W;
            if (n < N_own) {
                const float * ref = own.data() + (static_cast<size_t>(n) * C + q_local) * W;
                for (int k = 0; k < W; ++k) {
                    if ((row[k] == 0.0f) != (ref[k] == 0.0f)) {
                        fail_case("padded batch equivalence", T, C, tag, q_abs, k);
                    }
                }
                continue;
            }
            for (int k = 0; k < W; ++k) {
                if ((row[k] == 0.0f) != (k == 0) || (row[k] != 0.0f && !is_masked(row[k]))) {
                    fail_case("padded batch sentinel", T, C, tag, q_abs, k);
                }
            }
        }
    }
}

}  // namespace

int main() {
    for (int T = 1; T <= 40; T += 3) {
        for (int T_alloc = T; T_alloc <= T + 37; T_alloc += 7) {
            for (int C = 1; C <= 16; C += 5) {
                run_padded_case(T, T_alloc, C, /*left=*/8, /*right=*/5, /*chunked=*/false);
                run_padded_case(T, T_alloc, C, /*left_chunks=*/2, /*right=*/0, /*chunked=*/true);
            }
        }
    }

    for (int T = 1; T <= 97; ++T) {
        for (int C = 1; C <= 16; C += 3) {
            for (int left = 0; left <= 20; left += 4) {
                for (int right = 0; right <= 20; right += 5) {
                    run_local_case(T, C, left, right);
                }
            }
        }
    }

    // Exhaust the awkward geometries around every chunk boundary and ragged
    // tail. This includes the Nemotron 3.5 geometry (C=14, left_chunks=4)
    // as well as much smaller windows that make off-by-one
    // errors easier to expose.
    for (int T = 1; T <= 97; ++T) {
        for (int C = 1; C <= 16; ++C) {
            for (int left_chunks = 0; left_chunks <= 6; ++left_chunks) {
                run_case(T, C, left_chunks);
            }
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d mismatches\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "OK\n");
    return EXIT_SUCCESS;
}
