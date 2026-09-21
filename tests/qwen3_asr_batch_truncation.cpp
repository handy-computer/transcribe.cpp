// Real-model regression test for decode-budget scaling and single/batch
// OUTPUT_TRUNCATED parity. The long clip completes at the default context and
// truncates under a lowered n_ctx; the short clip completes in both cases.
// Requires TRANSCRIBE_QWEN3_ASR_0_6B_GGUF; missing inputs return 77.

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
    const char * model_path = std::getenv("TRANSCRIBE_QWEN3_ASR_0_6B_GGUF");
    if (model_path == nullptr || *model_path == '\0' || !file_exists(model_path)) {
        std::fprintf(stderr, "skipping: TRANSCRIBE_QWEN3_ASR_0_6B_GGUF unset or missing\n");
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

    // The long clip must complete with the default context.
    {
        transcribe_session_params full_sp;
        transcribe_session_params_init(&full_sp);
        struct transcribe_session * full_s = nullptr;
        if (transcribe_session_init(model, &full_sp, &full_s) != TRANSCRIBE_OK) {
            std::fprintf(stderr, "session init failed\n");
            transcribe_model_free(model);
            return 1;
        }
        const transcribe_status rl = transcribe_run(full_s, pcm_long.data(), (int) pcm_long.size(), nullptr);
        CHECK(rl == TRANSCRIBE_OK);
        CHECK(transcribe_was_truncated(full_s) == false);
        transcribe_session_free(full_s);
    }

    // Lower n_ctx enough to force truncation without rejecting the input.
    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    sp.n_ctx                      = 3072;
    struct transcribe_session * s = nullptr;
    if (transcribe_session_init(model, &sp, &s) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "session init failed\n");
        transcribe_model_free(model);
        return 1;
    }

    // Single-shot truncation and reset behavior.
    {
        const transcribe_status rl = transcribe_run(s, pcm_long.data(), (int) pcm_long.size(), nullptr);
        CHECK(rl == TRANSCRIBE_ERR_OUTPUT_TRUNCATED);
        CHECK(transcribe_was_truncated(s) == true);
        const char * t = transcribe_full_text(s);
        CHECK(t != nullptr && t[0] != '\0');  // partial transcript retained
    }
    {
        const transcribe_status rs = transcribe_run(s, pcm_short.data(), (int) pcm_short.size(), nullptr);
        CHECK(rs == TRANSCRIBE_OK);
        CHECK(transcribe_was_truncated(s) == false);  // reset + completed
    }

    // Batch results must match the single-shot results.
    {
        const float * pcms[2] = { pcm_short.data(), pcm_long.data() };
        const int     lens[2] = { (int) pcm_short.size(), (int) pcm_long.size() };

        CHECK(transcribe_run_batch(s, pcms, lens, 2, nullptr) == TRANSCRIBE_OK);
        CHECK_EQ_INT(transcribe_batch_n_results(s), 2);

        CHECK(transcribe_batch_status(s, 0) == TRANSCRIBE_OK);
        CHECK(transcribe_batch_status(s, 1) == TRANSCRIBE_ERR_OUTPUT_TRUNCATED);

        // Both rows keep their (partial, for row 1) transcript.
        for (int i = 0; i < 2; ++i) {
            const char * text = transcribe_batch_full_text(s, i);
            CHECK(text != nullptr && text[0] != '\0');
        }

        // The supplemental flag is set whenever any row truncated.
        CHECK(transcribe_was_truncated(s) == true);
    }

    transcribe_session_free(s);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "qwen3_asr_batch_truncation: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "qwen3_asr_batch_truncation: ok\n");
    return EXIT_SUCCESS;
}
