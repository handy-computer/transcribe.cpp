// granite5_ctc_e2e_smoke.cpp - real-model, public-C-ABI transcript smoke.
//
// Set TRANSCRIBE_GRANITE5_CTC_GGUF to any published Granite Speech 5.0
// TurboCTC tier. The test runs CPU inference on samples/jfk.wav, checks the
// transcript and no-timestamps result contract, then verifies that finer
// timestamp requests are rejected while preserving the previous result.

#include "transcribe.h"
#include "wav.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_SAMPLES_DIR
#    define TRANSCRIBE_TEST_SAMPLES_DIR "samples"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_EQ(actual, expected)                                                                               \
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

int edit_distance(const std::string & a, const std::string & b) {
    std::vector<int> prev(b.size() + 1);
    std::vector<int> curr(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) {
        prev[j] = static_cast<int>(j);
    }
    for (size_t i = 1; i <= a.size(); ++i) {
        curr[0] = static_cast<int>(i);
        for (size_t j = 1; j <= b.size(); ++j) {
            const int cost = a[i - 1] == b[j - 1] ? 0 : 1;
            curr[j]        = std::min({ prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost });
        }
        prev.swap(curr);
    }
    return prev[b.size()];
}

const char * const kReference =
    "and so my fellow americans ask not what your country can do for you ask what you can do for your country";

}  // namespace

int main() {
    const char * model_env = std::getenv("TRANSCRIBE_GRANITE5_CTC_GGUF");
    if (model_env == nullptr || model_env[0] == '\0' || !file_exists(model_env)) {
        std::fprintf(stderr, "granite5_ctc_e2e_smoke: TRANSCRIBE_GRANITE5_CTC_GGUF is unset or missing; skipping\n");
        return 77;
    }

    const char *      audio_env  = std::getenv("TRANSCRIBE_TEST_AUDIO");
    const std::string audio_path = audio_env != nullptr && audio_env[0] != '\0' ?
                                       audio_env :
                                       std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    if (!file_exists(audio_path)) {
        std::fprintf(stderr, "granite5_ctc_e2e_smoke: audio not found: %s\n", audio_path.c_str());
        return EXIT_FAILURE;
    }

    const auto                   start = std::chrono::steady_clock::now();
    transcribe_model_load_params load_params;
    transcribe_model_load_params_init(&load_params);
    load_params.backend      = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * model = nullptr;
    transcribe_status  st    = transcribe_model_load_file(model_env, &load_params, &model);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }

    CHECK(std::strcmp(transcribe_model_arch_string(model), "granite_speech5_ctc") == 0);
    CHECK(transcribe_model_backend(model) != nullptr && transcribe_model_backend(model)[0] != '\0');
    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
    CHECK_EQ(caps.max_timestamp_kind, TRANSCRIBE_TIMESTAMPS_NONE);

    std::vector<float> pcm;
    std::string        wav_error;
    if (!transcribe_cli::load_wav_mono_16k(audio_path, pcm, wav_error)) {
        std::fprintf(stderr, "FAIL audio load: %s\n", wav_error.c_str());
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_session_params session_params;
    transcribe_session_params_init(&session_params);
    transcribe_session * session = nullptr;
    st                           = transcribe_session_init(model, &session_params, &session);
    if (st != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "FAIL session init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_run_params run_params;
    transcribe_run_params_init(&run_params);
    st = transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &run_params);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL run: %s\n", transcribe_status_string(st));
        transcribe_session_free(session);
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    const std::string actual   = transcribe_full_text(session);
    const int         distance = edit_distance(actual, kReference);
    std::fprintf(stderr, "granite5_ctc_e2e_smoke: text=\"%s\" edit_distance=%d\n", actual.c_str(), distance);
    CHECK(!actual.empty());
    CHECK(distance <= 2);
    CHECK_EQ(transcribe_n_segments(session), 1);
    CHECK_EQ(transcribe_n_words(session), 0);
    const int n_tokens = transcribe_n_tokens(session);
    CHECK(n_tokens > 0);  // token ids/text remain useful without timing data
    CHECK_EQ(transcribe_returned_timestamp_kind(session), TRANSCRIBE_TIMESTAMPS_NONE);

    transcribe_segment segment;
    transcribe_segment_init(&segment);
    CHECK(transcribe_get_segment(session, 0, &segment) == TRANSCRIBE_OK);
    CHECK(segment.text != nullptr && segment.text[0] != '\0');
    CHECK_EQ(segment.n_words, 0);
    CHECK_EQ(segment.n_tokens, n_tokens);

    transcribe_timings timings;
    transcribe_timings_init(&timings);
    CHECK(transcribe_get_timings(session, &timings) == TRANSCRIBE_OK);
    CHECK(timings.load_ms > 0.0f);
    CHECK(timings.mel_ms > 0.0f);
    CHECK(timings.encode_ms > 0.0f);
    CHECK(timings.decode_ms > 0.0f);
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    CHECK(wall_ms < 120000.0);

    transcribe_run_params rejected_params;
    transcribe_run_params_init(&rejected_params);
    rejected_params.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
    CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rejected_params) ==
          TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS);
    CHECK(std::strcmp(transcribe_full_text(session), actual.c_str()) == 0);
    CHECK_EQ(transcribe_n_segments(session), 1);
    CHECK_EQ(transcribe_n_words(session), 0);
    CHECK_EQ(transcribe_n_tokens(session), n_tokens);

    rejected_params.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
    CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rejected_params) ==
          TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS);
    CHECK(std::strcmp(transcribe_full_text(session), actual.c_str()) == 0);
    CHECK_EQ(transcribe_n_segments(session), 1);

    transcribe_session_free(session);
    transcribe_model_free(model);
    if (g_failures != 0) {
        std::fprintf(stderr, "granite5_ctc_e2e_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "granite5_ctc_e2e_smoke: ok\n");
    return EXIT_SUCCESS;
}
