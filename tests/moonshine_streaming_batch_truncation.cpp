// moonshine_streaming_batch_truncation.cpp - real-model gated test for the
// learned adapter-position limit in one-shot and batch inference.
//
// The adapter has 4096 position rows and receives one position per 20 ms
// encoder frame, so tiny accepts at most 81.92 seconds. Longer audio must be
// rejected before encoder compute rather than reaching ggml_get_rows with an
// out-of-range index.
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
    transcribe_model * model = nullptr;
    if (transcribe_model_load_file(model_path, &mp, &model) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "model load failed: %s\n", model_path);
        return 1;
    }

    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
    CHECK_EQ_INT(caps.max_audio_ms, 81920);

    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    transcribe_session * s = nullptr;
    if (transcribe_session_init(model, &sp, &s) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "context init failed\n");
        transcribe_model_free(model);
        return 1;
    }

    transcribe_session_limits limits;
    transcribe_session_limits_init(&limits);
    CHECK(transcribe_session_get_limits(s, &limits) == TRANSCRIBE_OK);
    CHECK_EQ_INT(limits.effective_n_ctx, 4096);
    CHECK_EQ_INT(limits.effective_max_audio_ms, 81920);
    CHECK_EQ_INT(limits.max_kv_bytes, 62914560);

    // One-shot rejects before encoder compute and clears a prior transcript.
    CHECK(transcribe_run(s, pcm_short.data(), static_cast<int>(pcm_short.size()), nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_full_text(s) != nullptr && transcribe_full_text(s)[0] != '\0');
    CHECK(transcribe_run(s, pcm_long.data(), static_cast<int>(pcm_long.size()), nullptr) ==
          TRANSCRIBE_ERR_INPUT_TOO_LONG);
    CHECK(transcribe_full_text(s) == nullptr || transcribe_full_text(s)[0] == '\0');

    const float * pcms[2] = { pcm_short.data(), pcm_long.data() };
    const int     lens[2] = { static_cast<int>(pcm_short.size()), static_cast<int>(pcm_long.size()) };

    // A mixed batch reports the hard input limit per utterance.
    CHECK(transcribe_run_batch(s, pcms, lens, 2, nullptr) == TRANSCRIBE_OK);
    CHECK_EQ_INT(transcribe_batch_n_results(s), 2);
    CHECK(transcribe_batch_status(s, 0) == TRANSCRIBE_OK);
    CHECK(transcribe_batch_status(s, 1) == TRANSCRIBE_ERR_INPUT_TOO_LONG);
    const char * short_text = transcribe_batch_full_text(s, 0);
    const char * long_text  = transcribe_batch_full_text(s, 1);
    CHECK(short_text != nullptr && short_text[0] != '\0');
    CHECK(long_text == nullptr || long_text[0] == '\0');
    CHECK(transcribe_was_truncated(s) == false);

    // Streaming rejects a feed that would cross the same table bound.
    transcribe_run_params run_params;
    transcribe_run_params_init(&run_params);
    transcribe_stream_params stream_params;
    transcribe_stream_params_init(&stream_params);
    CHECK(transcribe_stream_begin(s, &run_params, &stream_params) == TRANSCRIBE_OK);
    transcribe_stream_update update;
    transcribe_stream_update_init(&update);
    CHECK(transcribe_stream_feed(s, pcm_long.data(), static_cast<int>(pcm_long.size()), &update) ==
          TRANSCRIBE_ERR_INPUT_TOO_LONG);

    transcribe_session_free(s);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "moonshine_streaming_input_limit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "moonshine_streaming_input_limit: ok\n");
    return EXIT_SUCCESS;
}
