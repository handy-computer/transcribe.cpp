// cohere_smoke.cpp - real-model gated end-to-end test for the Cohere ASR
// family through the public C ABI.
//
// Loads a real Cohere ASR GGUF, runs transcription on a WAV file, and
// verifies:
//
//   1. Model loads OK and reports arch "cohere_asr".
//   2. Backend is non-empty after load (one of "metal", "cpu", etc.).
//   3. transcribe_run completes OK on the test WAV.
//   4. transcribe_full_text is non-empty and matches the reference
//      greedy decode on samples/jfk.wav to within Levenshtein edit
//      distance 3 (same WER budget as decoder_smoke.cpp's Parakeet
//      gate). Catches regressions like BN fuse bugs or attention
//      mask flips that a loose substring check silently tolerates.
//   5. Timing information is present and reasonable (non-zero, not
//      hung -- load + run under 120 seconds).
//   6. Resources are cleaned up without crash.
//
// Gating:
//
//   - The CMake option TRANSCRIBE_BUILD_REAL_MODEL_TESTS (default OFF)
//     controls whether this binary is built.
//   - At runtime, the GGUF path comes from TRANSCRIBE_COHERE_MODEL
//     env var. If unset, the test falls back to
//     models/cohere/cohere.f16.gguf (relative to CMAKE_SOURCE_DIR).
//     If neither path exists, the test exits 77 (CTest "skipped").
//   - The WAV path comes from TRANSCRIBE_TEST_AUDIO env var, falling
//     back to samples/jfk.wav.
//   - transcribe_model_params::use_gpu is set to false before model
//     load for deterministic results across platforms.
//
// CI never builds this -- it is a developer-local manual gate. The
// synthetic fixture-based test (when written) covers structural
// correctness without needing a real model file.

#include "transcribe.h"

#include "wav.h"

#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_SAMPLES_DIR
#  define TRANSCRIBE_TEST_SAMPLES_DIR "samples"
#endif

#ifndef TRANSCRIBE_TEST_MODELS_DIR
#  define TRANSCRIBE_TEST_MODELS_DIR "models"
#endif

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

#define CHECK_EQ_INT(actual, expected)                                      \
    do {                                                                    \
        const long long _a = static_cast<long long>(actual);                \
        const long long _e = static_cast<long long>(expected);              \
        if (_a != _e) {                                                     \
            std::fprintf(stderr,                                            \
                         "FAIL %s:%d: %s = %lld, expected %lld\n",          \
                         __FILE__, __LINE__, #actual, _a, _e);              \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

#define CHECK_STR_EQ(a, b)                                                  \
    do {                                                                    \
        const std::string _av = (a);                                        \
        const std::string _bv = (b);                                        \
        if (_av != _bv) {                                                   \
            std::fprintf(stderr,                                            \
                         "FAIL %s:%d: \"%s\" != \"%s\"\n",                  \
                         __FILE__, __LINE__,                                \
                         _av.c_str(), _bv.c_str());                         \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

// Levenshtein edit distance between two strings. Used for the WER-ish
// gate ("edit distance <= 3"). Implementation is the standard rolling-
// row DP; quadratic in time, linear in memory. jfk.wav's reference
// text is ~110 characters so this is trivially fast at the test gate.
//
// Duplicated verbatim from decoder_smoke.cpp rather than hoisted into
// a shared header: the porting policy is no-refactor-during-port, and
// a one-function helper in two tests is cheaper than a new header.
int edit_distance(const std::string & a, const std::string & b) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());
    std::vector<int> prev(static_cast<size_t>(m + 1));
    std::vector<int> curr(static_cast<size_t>(m + 1));
    for (int j = 0; j <= m; ++j) prev[static_cast<size_t>(j)] = j;
    for (int i = 1; i <= n; ++i) {
        curr[0] = i;
        for (int j = 1; j <= m; ++j) {
            const int cost = (a[static_cast<size_t>(i - 1)] ==
                              b[static_cast<size_t>(j - 1)]) ? 0 : 1;
            const int del = prev[static_cast<size_t>(j)]     + 1;
            const int ins = curr[static_cast<size_t>(j - 1)] + 1;
            const int sub = prev[static_cast<size_t>(j - 1)] + cost;
            int best = del < ins ? del : ins;
            if (sub < best) best = sub;
            curr[static_cast<size_t>(j)] = best;
        }
        prev.swap(curr);
    }
    return prev[static_cast<size_t>(m)];
}

// Reference text from the Cohere ASR greedy decode against the
// cohere-transcribe-03-2026 checkpoint on samples/jfk.wav. This
// matches parakeet-mlx's output character-for-character (same JFK
// quote, same punctuation), which is the point -- the text is
// unambiguous so any reasonable STT system should produce it. The
// Levenshtein tolerance gives us 3 character-edits of slack for
// minor tokenizer / detokenizer changes.
const char * const k_jfk_reference_text =
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country.";

constexpr int k_max_edit_distance = 3;

} // namespace

int main() {
    // ---- Resolve model path ------------------------------------------
    std::string model_path;
    {
        const char * env = std::getenv("TRANSCRIBE_COHERE_MODEL");
        if (env != nullptr && env[0] != '\0') {
            model_path = env;
        } else {
            model_path = std::string(TRANSCRIBE_TEST_MODELS_DIR)
                         + "/cohere/cohere.f16.gguf";
        }
    }

    if (!file_exists(model_path)) {
        std::fprintf(stderr,
                     "cohere_smoke: model not found: %s\n"
                     "Set TRANSCRIBE_COHERE_MODEL or place the GGUF at the "
                     "default path. Skipping.\n",
                     model_path.c_str());
        return 77; // CTest "skipped"
    }

    // ---- Resolve audio path ------------------------------------------
    std::string wav_path;
    {
        const char * env = std::getenv("TRANSCRIBE_TEST_AUDIO");
        if (env != nullptr && env[0] != '\0') {
            wav_path = env;
        } else {
            wav_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
        }
    }

    if (!file_exists(wav_path)) {
        std::fprintf(stderr,
                     "cohere_smoke: wav not found: %s\n", wav_path.c_str());
        return EXIT_FAILURE;
    }

    std::fprintf(stderr, "cohere_smoke: model=%s\n", model_path.c_str());
    std::fprintf(stderr, "cohere_smoke: audio=%s\n", wav_path.c_str());

    // ---- Load model --------------------------------------------------
    const auto t_start = std::chrono::steady_clock::now();

    transcribe_model_params mp = transcribe_model_default_params();
    mp.use_gpu = false;  // CPU for cross-platform determinism
    struct transcribe_model * model = nullptr;
    {
        const transcribe_status st =
            transcribe_model_load_file(model_path.c_str(), &mp, &model);
        if (st != TRANSCRIBE_OK || model == nullptr) {
            std::fprintf(stderr, "FAIL load: %s\n",
                         transcribe_status_string(st));
            return EXIT_FAILURE;
        }
    }

    // ---- Public ABI sanity -------------------------------------------
    CHECK_STR_EQ(transcribe_model_arch_string(model), "cohere_asr");
    {
        const char * backend = transcribe_model_backend(model);
        if (backend == nullptr || backend[0] == '\0') {
            std::fprintf(stderr,
                         "FAIL: backend = \"\" after load, expected non-empty\n");
            ++g_failures;
        } else {
            std::fprintf(stderr, "cohere_smoke: backend=%s\n", backend);
        }
    }

    // ---- Load wav ----------------------------------------------------
    std::vector<float> pcm;
    std::string load_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, load_err)) {
        std::fprintf(stderr, "cohere_smoke: wav load: %s\n",
                     load_err.c_str());
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "cohere_smoke: loaded %zu samples (%.2f s)\n",
                 pcm.size(), static_cast<double>(pcm.size()) / 16000.0);

    // ---- Init context + run ------------------------------------------
    transcribe_context_params cp = transcribe_context_default_params();
    struct transcribe_context * ctx = nullptr;
    {
        const transcribe_status st =
            transcribe_context_init(model, &cp, &ctx);
        if (st != TRANSCRIBE_OK || ctx == nullptr) {
            std::fprintf(stderr, "cohere_smoke: ctx init: %s\n",
                         transcribe_status_string(st));
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    transcribe_params rp = transcribe_default_params();
    {
        const transcribe_status st =
            transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL run: %s\n",
                         transcribe_status_string(st));
            transcribe_context_free(ctx);
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    // ---- Verify output text ------------------------------------------
    const char * full = transcribe_full_text(ctx);
    CHECK(full != nullptr);
    const std::string actual = full ? full : "";
    std::fprintf(stderr, "cohere_smoke: text=\"%s\"\n", actual.c_str());

    // Text must be non-empty.
    CHECK(!actual.empty());

    // WER-ish gate: Levenshtein edit distance against the reference
    // greedy decode. Tolerance 3 matches decoder_smoke.cpp's budget
    // for Parakeet; it's small enough to catch real regressions
    // (BN fuse flip, attention-mask off-by-one, token boundary
    // rewrite) but not so tight that a one-character detokenizer
    // tweak trips the gate.
    const int dist = edit_distance(actual, k_jfk_reference_text);
    std::fprintf(stderr,
                 "cohere_smoke: edit_distance=%d (tolerance=%d)\n",
                 dist, k_max_edit_distance);
    if (dist > k_max_edit_distance) {
        std::fprintf(stderr,
                     "FAIL: text edit distance %d exceeds %d\n",
                     dist, k_max_edit_distance);
        std::fprintf(stderr, "  reference: %s\n", k_jfk_reference_text);
        std::fprintf(stderr, "  actual:    %s\n", actual.c_str());
        ++g_failures;
    }

    // ---- Verify timing -----------------------------------------------
    const transcribe_timings t = transcribe_get_timings(ctx);
    std::fprintf(stderr,
                 "cohere_smoke: timings load=%.2f mel=%.2f "
                 "encode=%.2f decode=%.2f ms\n",
                 t.load_ms, t.mel_ms, t.encode_ms, t.decode_ms);

    // Load time must be non-zero (model was loaded from disk).
    CHECK(t.load_ms > 0.0f);

    // At least some compute must have happened.
    CHECK(t.mel_ms > 0.0f || t.encode_ms > 0.0f || t.decode_ms > 0.0f);

    // Wall clock sanity: the entire test (load + inference) should
    // complete well under 120 seconds on CPU for an 11-second clip.
    // If it takes longer, something is hung or degenerate.
    std::fprintf(stderr, "cohere_smoke: wall=%.2f ms\n", wall_ms);
    if (wall_ms > 120000.0) {
        std::fprintf(stderr,
                     "FAIL: wall time %.0f ms exceeds 120 s budget\n",
                     wall_ms);
        ++g_failures;
    }

    // ---- Verify result accessors are populated -----------------------
    const int n_segments = transcribe_n_segments(ctx);
    CHECK(n_segments > 0);

    // At least one segment with non-empty text.
    if (n_segments > 0) {
        const char * seg_text = transcribe_segment_text(ctx, 0);
        CHECK(seg_text != nullptr && seg_text[0] != '\0');
    }

    // ---- Teardown ----------------------------------------------------
    transcribe_context_free(ctx);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "cohere_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "cohere_smoke: ok\n");
    return EXIT_SUCCESS;
}
