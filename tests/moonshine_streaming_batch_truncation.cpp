// moonshine_streaming_batch_truncation.cpp - real-model gated test that
// transcribe_run_batch reports output truncation PER UTTERANCE.
//
// Moonshine's cap is on output (max_length decode tokens), not input, so a
// long clip runs the decoder into the cap before end-of-stream. In a batch,
// that must surface as a per-utterance TRANSCRIBE_ERR_OUTPUT_TRUNCATED on the
// affected row (with its partial text retained), while a short row that
// finishes normally stays TRANSCRIBE_OK and the whole-batch call still returns
// OK. transcribe_was_truncated() is also set. See docs/input-limits.md.
//
// Batch makeup:
//   row 0 = jfk.wav (~11 s)  -> completes under the cap   -> OK
//   row 1 = love-loss.wav (~197 s) -> exceeds the cap     -> OUTPUT_TRUNCATED
//
// Gating:
//   - TRANSCRIBE_BUILD_REAL_MODEL_TESTS (CMake, default OFF) builds it.
//   - At runtime, TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF points at a tiny
//     GGUF. If unset/missing (or a sample is missing), exits 77 ("skipped").

#include "transcribe.h"
#include "wav.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <string>
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

#define CHECK_EQ_INT(actual, expected)                                                                           \
    do {                                                                                                         \
        const long long _a = static_cast<long long>(actual);                                                     \
        const long long _e = static_cast<long long>(expected);                                                   \
        if (_a != _e) {                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #actual, _a, _e); \
            ++g_failures;                                                                                        \
        }                                                                                                        \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

bool load_sample(const std::string & name, std::vector<float> & pcm) {
    const std::string path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/" + name;
    if (!file_exists(path)) {
        std::fprintf(stderr, "skipping: %s missing\n", path.c_str());
        return false;
    }
    std::string err;
    if (!transcribe_cli::load_wav_mono_16k(path, pcm, err) || pcm.empty()) {
        std::fprintf(stderr, "failed to load %s: %s\n", path.c_str(), err.c_str());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const char * model_path = std::getenv("TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF");
    if (model_path == nullptr || *model_path == '\0' || !file_exists(model_path)) {
        std::fprintf(stderr,
                     "skipping: TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF unset "
                     "or missing\n");
        return 77;
    }

    std::vector<float> pcm_short, pcm_long;
    if (!load_sample("jfk.wav", pcm_short)) {
        return 77;
    }
    if (!load_sample("love-loss.wav", pcm_long)) {
        return 77;
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(model_path, &mp, &model) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "model load failed: %s\n", model_path);
        return 1;
    }

    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    struct transcribe_session * s = nullptr;
    if (transcribe_session_init(model, &sp, &s) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "session init failed\n");
        transcribe_model_free(model);
        return 1;
    }

    const float * pcms[2] = { pcm_short.data(), pcm_long.data() };
    const int     lens[2] = { (int) pcm_short.size(), (int) pcm_long.size() };

    // The whole-batch call succeeds even though a row truncates.
    CHECK(transcribe_run_batch(s, pcms, lens, 2, nullptr) == TRANSCRIBE_OK);
    CHECK_EQ_INT(transcribe_batch_n_results(s), 2);

    // Row 0 (short) completes; row 1 (long) hits the output cap.
    CHECK(transcribe_batch_status(s, 0) == TRANSCRIBE_OK);
    CHECK(transcribe_batch_status(s, 1) == TRANSCRIBE_ERR_OUTPUT_TRUNCATED);

    // Both rows keep their (partial, for row 1) transcript.
    for (int i = 0; i < 2; ++i) {
        const char * text = transcribe_batch_full_text(s, i);
        CHECK(text != nullptr && text[0] != '\0');
    }

    // The supplemental flag is set whenever any row truncated.
    CHECK(transcribe_was_truncated(s) == true);

    transcribe_session_free(s);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "moonshine_streaming_batch_truncation: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "moonshine_streaming_batch_truncation: ok\n");
    return EXIT_SUCCESS;
}
