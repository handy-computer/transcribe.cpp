// qwen3_asr_e2e_smoke.cpp - real-model gated end-to-end transcription
// test through the public C ABI.
//
// Loads a real Qwen3-ASR GGUF, runs transcription on samples/jfk.wav,
// and verifies:
//   1. Load returns OK; arch string is "qwen3_asr".
//   2. transcribe_run completes OK.
//   3. transcribe_full_text is non-empty and matches the reference
//      greedy decode on jfk.wav to within Levenshtein edit distance 5.
//      (Slightly looser than Cohere's 3 because we use bf16 LM on CPU
//      without the reference's flash attention — identical argmaxes at
//      prefill, but accumulated step-decoder drift at the 10–20 token
//      mark can change a single word occasionally.)
//   4. Load + run completes under 180 seconds on CPU.
//
// Gating:
//   - TRANSCRIBE_BUILD_REAL_MODEL_TESTS (CMake) controls whether this
//     binary is built.
//   - TRANSCRIBE_QWEN3_ASR_GGUF picks the GGUF at runtime. Unset → 77.
//   - TRANSCRIBE_TEST_AUDIO overrides the audio path; default jfk.wav.

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

// Reference transcript for samples/jfk.wav (Qwen3-ASR 0.6B greedy
// decode stripped of the "language English<asr_text>" prefix).
const char * const k_jfk_reference_text =
    "And so, my fellow Americans, ask not what your country can do for "
    "you; ask what you can do for your country.";

constexpr int k_max_edit_distance = 5;

} // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_QWEN3_ASR_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "qwen3_asr_e2e_smoke: TRANSCRIBE_QWEN3_ASR_GGUF not set; "
                     "skipping.\n");
        return 77;
    }
    const std::string model_path = env;
    if (!file_exists(model_path)) {
        std::fprintf(stderr,
                     "qwen3_asr_e2e_smoke: model not found: %s\n",
                     model_path.c_str());
        return 77;
    }

    std::string wav_path;
    if (const char * w = std::getenv("TRANSCRIBE_TEST_AUDIO"); w && w[0]) {
        wav_path = w;
    } else {
        wav_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    }
    if (!file_exists(wav_path)) {
        std::fprintf(stderr,
                     "qwen3_asr_e2e_smoke: wav not found: %s\n",
                     wav_path.c_str());
        return EXIT_FAILURE;
    }

    const auto t_start = std::chrono::steady_clock::now();

    transcribe_model_params mp = transcribe_model_default_params();
    mp.backend = TRANSCRIBE_BACKEND_CPU;
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

    CHECK_STR_EQ(transcribe_model_arch_string(model), "qwen3_asr");

    std::vector<float> pcm;
    std::string load_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, load_err)) {
        std::fprintf(stderr, "qwen3_asr_e2e_smoke: wav load: %s\n",
                     load_err.c_str());
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_context_params cp = transcribe_context_default_params();
    struct transcribe_context * ctx = nullptr;
    {
        const transcribe_status st =
            transcribe_context_init(model, &cp, &ctx);
        if (st != TRANSCRIBE_OK || ctx == nullptr) {
            std::fprintf(stderr, "FAIL ctx_init: %s\n",
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
    const double wall_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    const char * full = transcribe_full_text(ctx);
    CHECK(full != nullptr);
    const std::string actual = full ? full : "";
    std::fprintf(stderr, "qwen3_asr_e2e_smoke: text=\"%s\"\n", actual.c_str());
    CHECK(!actual.empty());

    const int dist = edit_distance(actual, k_jfk_reference_text);
    std::fprintf(stderr,
                 "qwen3_asr_e2e_smoke: edit_distance=%d (tolerance=%d)\n",
                 dist, k_max_edit_distance);
    if (dist > k_max_edit_distance) {
        std::fprintf(stderr,
                     "FAIL: text edit distance %d exceeds %d\n",
                     dist, k_max_edit_distance);
        std::fprintf(stderr, "  reference: %s\n", k_jfk_reference_text);
        std::fprintf(stderr, "  actual:    %s\n", actual.c_str());
        ++g_failures;
    }

    std::fprintf(stderr, "qwen3_asr_e2e_smoke: wall=%.2f ms\n", wall_ms);
    if (wall_ms > 180000.0) {
        std::fprintf(stderr,
                     "FAIL: wall time %.2f ms exceeds 180000 ms\n", wall_ms);
        ++g_failures;
    }

    transcribe_context_free(ctx);
    transcribe_model_free(model);

    if (g_failures > 0) return EXIT_FAILURE;
    std::fprintf(stdout, "qwen3_asr_e2e_smoke: ok\n");
    return EXIT_SUCCESS;
}
