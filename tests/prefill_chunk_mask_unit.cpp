// Geometry of the chunked-prefill causal mask.
//
// A single-shot prefill masks a square: query r sees keys [0, r]. Chunked
// prefill masks a trapezoid instead — query q of a chunk starting at n_past
// sits at absolute position n_past + q and sees keys [0, n_past + q], so the
// rows below n_past are visible to everything and only the tail is
// triangular. Getting that boundary wrong by one leaks exactly one future
// token per row, which does not crash and barely moves a short transcript, so
// it needs a direct test rather than an end-to-end one.

#include "causal_lm/causal_lm.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
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

bool is_keep(ggml_fp16_t v) {
    return ggml_fp16_to_fp32(v) == 0.0f;
}

bool is_drop(ggml_fp16_t v) {
    const float f = ggml_fp16_to_fp32(v);
    return std::isinf(f) && f < 0.0f;
}

void check_kv_context(int needed, int model_max, int expected) {
    const int got = transcribe::causal_lm::pick_kv_cache_context(needed, model_max);
    if (got != expected) {
        std::fprintf(stderr, "FAIL kv context (needed=%d model_max=%d): got %d, expected %d\n", needed, model_max, got,
                     expected);
        ++g_failures;
    }
}

// Every entry of a [max_n_kv, T_chunk] mask matches the causal rule exactly.
void check_chunk(int max_n_kv, int T_chunk, int n_past) {
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(max_n_kv) * T_chunk, ggml_fp32_to_fp16(123.0f));
    transcribe::causal_lm::fill_prefill_chunk_mask(mask.data(), max_n_kv, T_chunk, n_past);

    int bad = 0;
    for (int q = 0; q < T_chunk; ++q) {
        for (int k = 0; k < max_n_kv; ++k) {
            const ggml_fp16_t v    = mask[static_cast<size_t>(q) * max_n_kv + k];
            const bool        want = (k <= n_past + q);  // visible iff at or before this query
            if (want ? !is_keep(v) : !is_drop(v)) {
                ++bad;
            }
        }
    }
    if (bad != 0) {
        std::fprintf(stderr, "FAIL mask(max_n_kv=%d T_chunk=%d n_past=%d): %d wrong entries\n", max_n_kv, T_chunk,
                     n_past, bad);
        ++g_failures;
    }
}

}  // namespace

int main() {
    using transcribe::causal_lm::fill_prefill_chunk_mask;

    // Keep short-audio allocations at their historical sizes, but switch to
    // fixed 4K growth above that so a one-token boundary crossing does not
    // double a large cache.
    check_kv_context(1, 131072, 1024);
    check_kv_context(1024, 131072, 1024);
    check_kv_context(1025, 131072, 2048);
    check_kv_context(2048, 131072, 2048);
    check_kv_context(2049, 131072, 4096);
    check_kv_context(4096, 131072, 4096);
    check_kv_context(4097, 131072, 8192);
    check_kv_context(8192, 131072, 8192);
    check_kv_context(8193, 131072, 12288);
    check_kv_context(32768, 131072, 32768);
    check_kv_context(32769, 131072, 36864);
    check_kv_context(65537, 131072, 69632);
    check_kv_context(131071, 131072, 131072);
    check_kv_context(4097, 6000, 6000);  // trained/user ceiling wins
    check_kv_context(1, 512, 512);
    check_kv_context(1, 0, 0);

    // n_past == 0 with a full-width window is the plain causal triangle, i.e.
    // exactly what the single-shot prefill path uploads.
    check_chunk(/*max_n_kv=*/8, /*T_chunk=*/8, /*n_past=*/0);
    check_chunk(1, 1, 0);

    // Interior chunks: a rectangle of history plus a triangle of self.
    check_chunk(/*max_n_kv=*/16, /*T_chunk=*/8, /*n_past=*/8);
    check_chunk(/*max_n_kv=*/24, /*T_chunk=*/8, /*n_past=*/16);

    // Ragged final chunk (T_chunk smaller than the others), and the
    // single-token final chunk that a prompt of chunk*n+1 tokens produces.
    check_chunk(/*max_n_kv=*/20, /*T_chunk=*/4, /*n_past=*/16);
    check_chunk(/*max_n_kv=*/17, /*T_chunk=*/1, /*n_past=*/16);

    // Odd sizes, so nothing depends on power-of-two alignment.
    check_chunk(/*max_n_kv=*/23, /*T_chunk=*/7, /*n_past=*/16);
    check_chunk(/*max_n_kv=*/227, /*T_chunk=*/35, /*n_past=*/192);

    // Walking the whole prompt one chunk at a time must tile the square
    // triangle a single-shot prefill would have built — no gaps, no overlap,
    // no leaked future. Reassemble it and compare against the reference.
    {
        const int                T = 37, chunk = 8;
        // Reference: full [T, T] causal mask.
        std::vector<ggml_fp16_t> full(static_cast<size_t>(T) * T);
        fill_prefill_chunk_mask(full.data(), T, T, 0);

        std::vector<ggml_fp16_t> scratch(static_cast<size_t>(T) * chunk);
        for (int n_past = 0; n_past < T; n_past += chunk) {
            const int T_chunk  = std::min(chunk, T - n_past);
            const int max_n_kv = n_past + T_chunk;
            fill_prefill_chunk_mask(scratch.data(), max_n_kv, T_chunk, n_past);
            for (int q = 0; q < T_chunk; ++q) {
                const int row = n_past + q;
                for (int k = 0; k < T; ++k) {
                    // Beyond this chunk's window the key is simply not read.
                    const ggml_fp16_t got =
                        (k < max_n_kv) ? scratch[static_cast<size_t>(q) * max_n_kv + k] : ggml_fp32_to_fp16(-INFINITY);
                    const ggml_fp16_t want = full[static_cast<size_t>(row) * T + k];
                    CHECK(is_keep(got) == is_keep(want));
                }
            }
        }
    }

    // Bad geometry is rejected, not written past the end of the buffer. The
    // canary stays untouched because the helper bails before filling.
    {
        std::vector<ggml_fp16_t> mask(16, ggml_fp32_to_fp16(7.0f));
        fill_prefill_chunk_mask(mask.data(), /*max_n_kv=*/4, /*T_chunk=*/4, /*n_past=*/4);  // n_past+T > max_n_kv
        CHECK(ggml_fp16_to_fp32(mask[0]) == 7.0f);
        fill_prefill_chunk_mask(mask.data(), /*max_n_kv=*/4, /*T_chunk=*/4, /*n_past=*/-1);
        CHECK(ggml_fp16_to_fp32(mask[0]) == 7.0f);
        fill_prefill_chunk_mask(nullptr, 4, 4, 0);  // must not crash
    }

    if (g_failures == 0) {
        std::printf("OK\n");
    }
    return g_failures == 0 ? 0 : 1;
}
