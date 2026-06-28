// parakeet_stream_ext_reject_unit.cpp - regression test for the
// "< -1 sentinel returns INVALID_ARG" rule on Parakeet's streaming
// extension fields.
//
// The public per-field sentinel semantics for the streaming extension
// knobs are documented in include/transcribe/parakeet.h:
//
//   -1     "use the model default for this field"
//   < -1   caller bug; transcribe_stream_begin returns INVALID_ARG
//    0     a real requested value (when 0 is in the model's menu)
//   > 0    must be exact (frame multiple) / in-menu; otherwise INVALID_ARG
//
// Two paths use this rule:
//
//   - Cache-aware (ChunkedLimited, nemotron-streaming-style):
//     transcribe_parakeet_stream_ext::att_context_right.
//   - Buffered (ChunkedLimitedWithRc, parakeet-unified-style):
//     transcribe_parakeet_buffered_stream_ext::{left,chunk,right}_ms.
//
// This test reaches both paths against the synthetic
// tokenizer_minimal_streaming_{cache_aware,buffered}.gguf fixtures.
// The rejection fires inside parakeet's stream_validate hook BEFORE
// the dispatcher clears the previous result snapshot or transitions
// the stream lifecycle out of IDLE, so the toy fixtures without real
// weights are sufficient. Pre-flight rejection also means the session
// stays IDLE (not FAILED) and last_status is untouched — that is the
// public contract for caller-side option mistakes.

#include "transcribe.h"
#include "transcribe/parakeet.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#    error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

const std::string g_fixtures_dir = TRANSCRIBE_TEST_FIXTURES_DIR;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

bool load_and_init(const char *                 fixture_name,
                   struct transcribe_model **   out_model,
                   struct transcribe_session ** out_ctx) {
    *out_model = nullptr;
    *out_ctx   = nullptr;

    const std::string p = g_fixtures_dir + "/" + fixture_name;
    if (!file_exists(p)) {
        std::fprintf(stderr,
                     "parakeet_stream_ext_reject_unit: fixture not found: %s\n"
                     "  regenerate with: cmake --build build --target fixtures\n",
                     p.c_str());
        ++g_failures;
        return false;
    }

    const transcribe_status load_st = transcribe_model_load_file(p.c_str(), nullptr, out_model);
    if (load_st != TRANSCRIBE_OK || *out_model == nullptr) {
        std::fprintf(stderr, "FAIL load %s: %s\n", fixture_name, transcribe_status_string(load_st));
        ++g_failures;
        return false;
    }

    const transcribe_status init_st = transcribe_session_init(*out_model, nullptr, out_ctx);
    if (init_st != TRANSCRIBE_OK || *out_ctx == nullptr) {
        std::fprintf(stderr, "FAIL session_init %s: %s\n", fixture_name, transcribe_status_string(init_st));
        transcribe_model_free(*out_model);
        *out_model = nullptr;
        ++g_failures;
        return false;
    }
    return true;
}

// Cache-aware path. transcribe_parakeet_stream_ext::att_context_right
// < -1 must return INVALID_ARG. att_context_right == -1 is the
// model-default sentinel and is accepted (we don't fully exercise the
// happy path here because the toy fixture has no compute weights;
// stream_capability_unit and the real-model parity test cover that).
void test_cache_aware_rejects_sub_sentinel() {
    struct transcribe_model *   model = nullptr;
    struct transcribe_session * ctx   = nullptr;
    if (!load_and_init("tokenizer_minimal_streaming_cache_aware.gguf", &model, &ctx)) {
        return;
    }

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);

    transcribe_parakeet_stream_ext ext;
    transcribe_parakeet_stream_ext_init(&ext);
    ext.att_context_right = -2;

    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    sp.family = &ext.ext;

    const transcribe_status st = transcribe_stream_begin(ctx, &rp, &sp);
    CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);
    // Pre-flight rejection: dispatcher returns the error before clear
    // or any state transition, so the session stays IDLE with
    // last_status untouched.
    CHECK(transcribe_stream_get_state(ctx) == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_stream_last_status(ctx) == TRANSCRIBE_OK);

    // Also try a strongly negative value to confirm the boundary.
    ext.att_context_right       = -42;
    sp.family                   = &ext.ext;
    const transcribe_status st2 = transcribe_stream_begin(ctx, &rp, &sp);
    CHECK(st2 == TRANSCRIBE_ERR_INVALID_ARG);

    transcribe_session_free(ctx);
    transcribe_model_free(model);
}

// Buffered path. Each of {left,chunk,right}_ms < -1 must return
// INVALID_ARG. Tests each field independently so a bug that misses one
// of the three rejects is caught.
void test_buffered_rejects_sub_sentinel() {
    struct transcribe_model *   model = nullptr;
    struct transcribe_session * ctx   = nullptr;
    if (!load_and_init("tokenizer_minimal_streaming_buffered.gguf", &model, &ctx)) {
        return;
    }

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);

    auto try_reject = [&](int32_t L, int32_t C, int32_t R) {
        transcribe_parakeet_buffered_stream_ext ext;
        transcribe_parakeet_buffered_stream_ext_init(&ext);
        ext.left_ms  = L;
        ext.chunk_ms = C;
        ext.right_ms = R;
        transcribe_stream_params sp;
        transcribe_stream_params_init(&sp);
        sp.family                  = &ext.ext;
        const transcribe_status st = transcribe_stream_begin(ctx, &rp, &sp);
        CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);
        // Pre-flight rejection leaves lifecycle / last_status untouched.
        CHECK(transcribe_stream_get_state(ctx) == TRANSCRIBE_STREAM_IDLE);
        CHECK(transcribe_stream_last_status(ctx) == TRANSCRIBE_OK);
    };

    // Each field independently triggers the reject. The other two are
    // -1 (model default), which is the legitimate "use default" form.
    try_reject(-2, -1, -1);
    try_reject(-1, -2, -1);
    try_reject(-1, -1, -2);

    // All three together, with a strongly negative value.
    try_reject(-99, -99, -99);

    transcribe_session_free(ctx);
    transcribe_model_free(model);
}

// Confirms the lazy boundary: 0 is a legitimate request value for the
// buffered right_ms field (the toy menu includes right=0). With a tuple
// that resolves cleanly inside the menu, transcribe_stream_begin must
// return OK — i.e. sentinel-shape validation accepted right_ms=0.
//
// The toy fixture's frame_ms collapses to 1 ms because its
// (subsampling_factor=2, hop_length=4, sample_rate=16000) integer-
// divides to zero and the runtime clamps to 1. We pick ms values that
// resolve to frame counts in the menu (L=70, C=1, R=0) so the menu
// check also passes and the only thing under test is the sentinel
// rule.
void test_buffered_zero_is_real_value() {
    struct transcribe_model *   model = nullptr;
    struct transcribe_session * ctx   = nullptr;
    if (!load_and_init("tokenizer_minimal_streaming_buffered.gguf", &model, &ctx)) {
        return;
    }

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);

    transcribe_parakeet_buffered_stream_ext ext;
    transcribe_parakeet_buffered_stream_ext_init(&ext);
    // At frame_ms=1 for this fixture, ms == frame count. Menu allows
    // L ∈ {70}, C ∈ {1,2,7,13}, R ∈ {0,1,2,4,7,13}. Pick the corners.
    ext.left_ms  = 70;
    ext.chunk_ms = 1;
    ext.right_ms = 0;
    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    sp.family = &ext.ext;

    const transcribe_status st = transcribe_stream_begin(ctx, &rp, &sp);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr,
                     "FAIL %s:%d: right_ms=0 with in-menu tuple was "
                     "rejected (%s); 0 is a real value and must pass the "
                     "sentinel + menu checks\n",
                     __FILE__, __LINE__, transcribe_status_string(st));
        ++g_failures;
    }

    transcribe_session_free(ctx);
    transcribe_model_free(model);
}

}  // namespace

int main() {
    test_cache_aware_rejects_sub_sentinel();
    test_buffered_rejects_sub_sentinel();
    test_buffered_zero_is_real_value();

    if (g_failures > 0) {
        std::fprintf(stderr, "parakeet_stream_ext_reject_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "parakeet_stream_ext_reject_unit: ok\n");
    return EXIT_SUCCESS;
}
