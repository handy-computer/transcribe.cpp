// canary_timestamp_real_smoke.cpp - real-model gated public-API smoke for
// Canary 1B v2 timestamp behavior.

#include "transcribe.h"
#include "wav.h"

#include <sys/stat.h>

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

transcribe_model * load_model(const std::string & path) {
    transcribe_model_load_params params;
    transcribe_model_load_params_init(&params);

    transcribe_model *      model = nullptr;
    const transcribe_status st    = transcribe_model_load_file(path.c_str(), &params, &model);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load %s: %s\n", path.c_str(), transcribe_status_string(st));
        ++g_failures;
        return nullptr;
    }
    return model;
}

transcribe_session * init_session(transcribe_model * model, int n_ctx = 0) {
    transcribe_session_params params;
    transcribe_session_params_init(&params);
    params.n_ctx = n_ctx;

    transcribe_session *    session = nullptr;
    const transcribe_status st      = transcribe_session_init(model, &params, &session);
    if (st != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "FAIL session init: %s\n", transcribe_status_string(st));
        ++g_failures;
        return nullptr;
    }
    return session;
}

void check_empty_result(transcribe_session * session) {
    CHECK(std::strcmp(transcribe_full_text(session), "") == 0);
    CHECK_EQ_INT(transcribe_returned_timestamp_kind(session), TRANSCRIBE_TIMESTAMPS_NONE);
    CHECK_EQ_INT(transcribe_n_segments(session), 0);
    CHECK_EQ_INT(transcribe_n_words(session), 0);
    CHECK_EQ_INT(transcribe_n_tokens(session), 0);
}

void check_none_result(transcribe_session * session) {
    CHECK(transcribe_full_text(session)[0] != '\0');
    CHECK_EQ_INT(transcribe_returned_timestamp_kind(session), TRANSCRIBE_TIMESTAMPS_NONE);
    CHECK_EQ_INT(transcribe_n_segments(session), 1);
    CHECK_EQ_INT(transcribe_n_words(session), 0);
    CHECK_EQ_INT(transcribe_n_tokens(session), 0);

    transcribe_segment segment;
    transcribe_segment_init(&segment);
    CHECK_EQ_INT(transcribe_get_segment(session, 0, &segment), TRANSCRIBE_OK);
    CHECK(segment.text != nullptr && segment.text[0] != '\0');
    CHECK_EQ_INT(segment.t0_ms, 0);
    CHECK_EQ_INT(segment.t1_ms, 0);
    CHECK_EQ_INT(segment.first_word, 0);
    CHECK_EQ_INT(segment.n_words, 0);
    CHECK_EQ_INT(segment.first_token, 0);
    CHECK_EQ_INT(segment.n_tokens, 0);
}

void check_word_result(transcribe_session * session) {
    CHECK(transcribe_full_text(session)[0] != '\0');
    CHECK_EQ_INT(transcribe_returned_timestamp_kind(session), TRANSCRIBE_TIMESTAMPS_WORD);

    const int n_segments = transcribe_n_segments(session);
    const int n_words    = transcribe_n_words(session);
    CHECK(n_segments > 0);
    CHECK(n_words > 0);
    CHECK_EQ_INT(transcribe_n_tokens(session), 0);

    int     expected_first_word  = 0;
    int64_t previous_segment_end = -1;
    for (int i = 0; i < n_segments; ++i) {
        transcribe_segment segment;
        transcribe_segment_init(&segment);
        CHECK_EQ_INT(transcribe_get_segment(session, i, &segment), TRANSCRIBE_OK);
        CHECK(segment.text != nullptr && segment.text[0] != '\0');
        CHECK(segment.t0_ms >= previous_segment_end);
        CHECK(segment.t1_ms > segment.t0_ms);
        CHECK_EQ_INT(segment.first_word, expected_first_word);
        CHECK(segment.n_words > 0);
        CHECK_EQ_INT(segment.first_token, 0);
        CHECK_EQ_INT(segment.n_tokens, 0);
        expected_first_word += segment.n_words;
        previous_segment_end = segment.t1_ms;
    }
    CHECK_EQ_INT(expected_first_word, n_words);

    int64_t previous_word_end = -1;
    for (int i = 0; i < n_words; ++i) {
        transcribe_word word;
        transcribe_word_init(&word);
        CHECK_EQ_INT(transcribe_get_word(session, i, &word), TRANSCRIBE_OK);
        CHECK(word.text != nullptr && word.text[0] != '\0');
        CHECK(word.t0_ms >= previous_word_end);
        CHECK(word.t1_ms > word.t0_ms);
        CHECK(word.seg_index >= 0 && word.seg_index < n_segments);
        CHECK_EQ_INT(word.first_token, 0);
        CHECK_EQ_INT(word.n_tokens, 0);
        previous_word_end = word.t1_ms;
    }
}

void check_segment_result(transcribe_session * session) {
    CHECK(transcribe_full_text(session)[0] != '\0');
    CHECK_EQ_INT(transcribe_returned_timestamp_kind(session), TRANSCRIBE_TIMESTAMPS_SEGMENT);

    const int n_segments = transcribe_n_segments(session);
    CHECK(n_segments > 0);
    CHECK_EQ_INT(transcribe_n_words(session), 0);
    CHECK_EQ_INT(transcribe_n_tokens(session), 0);

    int64_t previous_end = -1;
    for (int i = 0; i < n_segments; ++i) {
        transcribe_segment segment;
        transcribe_segment_init(&segment);
        CHECK_EQ_INT(transcribe_get_segment(session, i, &segment), TRANSCRIBE_OK);
        CHECK(segment.text != nullptr && segment.text[0] != '\0');
        CHECK(segment.t0_ms >= previous_end);
        CHECK(segment.t1_ms > segment.t0_ms);
        CHECK_EQ_INT(segment.first_word, 0);
        CHECK_EQ_INT(segment.n_words, 0);
        CHECK_EQ_INT(segment.first_token, 0);
        CHECK_EQ_INT(segment.n_tokens, 0);
        previous_end = segment.t1_ms;
    }
}

void check_batch_truncation_reset(transcribe_model * model, const std::vector<float> & pcm) {
    // Force JFK to hit a tiny decoder budget, then verify a following silent
    // row does not inherit its per-utterance truncation state.
    transcribe_session * session = init_session(model, 18);
    if (session == nullptr) {
        return;
    }

    const std::vector<float> silence(16000, 0.0f);
    const float *            rows[]   = { pcm.data(), silence.data() };
    const int                lengths[] = { static_cast<int>(pcm.size()), static_cast<int>(silence.size()) };

    transcribe_run_params params;
    transcribe_run_params_init(&params);
    params.language   = "en";
    params.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;

    CHECK_EQ_INT(transcribe_run_batch(session, rows, lengths, 2, &params), TRANSCRIBE_OK);
    CHECK_EQ_INT(transcribe_batch_n_results(session), 2);
    CHECK_EQ_INT(transcribe_batch_status(session, 0), TRANSCRIBE_ERR_OUTPUT_TRUNCATED);
    CHECK(transcribe_batch_full_text(session, 0)[0] != '\0');
    CHECK_EQ_INT(transcribe_batch_status(session, 1), TRANSCRIBE_OK);
    CHECK(transcribe_was_truncated(session));

    transcribe_session_free(session);
}

void check_timestamp_model(const std::string & path, const std::vector<float> & pcm) {
    transcribe_model * model = load_model(path);
    if (model == nullptr) {
        return;
    }

    CHECK(std::strcmp(transcribe_model_arch_string(model), "canary") == 0);
    CHECK(std::strcmp(transcribe_model_variant_string(model), "canary-1b-v2") == 0);

    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    CHECK_EQ_INT(transcribe_model_get_capabilities(model, &caps), TRANSCRIBE_OK);
    CHECK_EQ_INT(caps.native_sample_rate, 16000);
    CHECK_EQ_INT(caps.max_timestamp_kind, TRANSCRIBE_TIMESTAMPS_WORD);

    transcribe_session * session = init_session(model);
    if (session == nullptr) {
        transcribe_model_free(model);
        return;
    }

    transcribe_run_params params;
    transcribe_run_params_init(&params);
    params.language   = "en";
    params.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
    CHECK_EQ_INT(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &params), TRANSCRIBE_OK);
    check_none_result(session);

    params.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
    CHECK_EQ_INT(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &params), TRANSCRIBE_OK);
    check_word_result(session);

    params.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
    CHECK_EQ_INT(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &params), TRANSCRIBE_OK);
    check_segment_result(session);

    params.task            = TRANSCRIBE_TASK_TRANSLATE;
    params.target_language = "de";
    params.timestamps      = TRANSCRIBE_TIMESTAMPS_AUTO;
    CHECK_EQ_INT(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &params), TRANSCRIBE_OK);
    check_none_result(session);

    // Explicit timestamps are invalid for translated text because the CTC
    // aligner emits source-language acoustic labels. The preflight rejection
    // preserves the preceding AUTO translation result.
    params.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
    CHECK_EQ_INT(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &params),
                 TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS);
    check_none_result(session);

    transcribe_session_free(session);
    check_batch_truncation_reset(model, pcm);
    transcribe_model_free(model);
}

void check_legacy_model(const std::string & path, const std::vector<float> & pcm) {
    transcribe_model * model = load_model(path);
    if (model == nullptr) {
        return;
    }

    CHECK(std::strcmp(transcribe_model_arch_string(model), "canary") == 0);
    CHECK(std::strcmp(transcribe_model_variant_string(model), "canary-1b-v2") == 0);

    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    CHECK_EQ_INT(transcribe_model_get_capabilities(model, &caps), TRANSCRIBE_OK);
    CHECK_EQ_INT(caps.max_timestamp_kind, TRANSCRIBE_TIMESTAMPS_NONE);

    transcribe_session * session = init_session(model);
    if (session == nullptr) {
        transcribe_model_free(model);
        return;
    }

    transcribe_run_params params;
    transcribe_run_params_init(&params);
    params.language   = "en";
    params.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
    CHECK_EQ_INT(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &params),
                 TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS);
    check_empty_result(session);

    params.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
    CHECK_EQ_INT(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &params),
                 TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS);
    check_empty_result(session);

    transcribe_session_free(session);
    transcribe_model_free(model);
}

}  // namespace

int main() {
    const char * timestamp_env = std::getenv("TRANSCRIBE_CANARY_TIMESTAMP_GGUF");
    if (timestamp_env == nullptr || timestamp_env[0] == '\0' || !file_exists(timestamp_env)) {
        std::fprintf(stderr,
                     "canary_timestamp_real_smoke: TRANSCRIBE_CANARY_TIMESTAMP_GGUF "
                     "unset or missing; skipping.\n");
        return 77;
    }

    std::string wav_path;
    if (const char * audio_env = std::getenv("TRANSCRIBE_TEST_AUDIO"); audio_env != nullptr && audio_env[0] != '\0') {
        wav_path = audio_env;
    } else {
        wav_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    }
    if (!file_exists(wav_path)) {
        std::fprintf(stderr, "canary_timestamp_real_smoke: audio not found: %s\n", wav_path.c_str());
        return EXIT_FAILURE;
    }

    std::vector<float> pcm;
    std::string        wav_error;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, wav_error) || pcm.empty()) {
        std::fprintf(stderr, "canary_timestamp_real_smoke: wav load: %s\n", wav_error.c_str());
        return EXIT_FAILURE;
    }

    check_timestamp_model(timestamp_env, pcm);

    const char * legacy_env = std::getenv("TRANSCRIBE_CANARY_LEGACY_GGUF");
    if (legacy_env == nullptr || legacy_env[0] == '\0') {
        std::fprintf(stderr,
                     "canary_timestamp_real_smoke: TRANSCRIBE_CANARY_LEGACY_GGUF "
                     "not set; legacy checks skipped.\n");
    } else if (!file_exists(legacy_env)) {
        std::fprintf(stderr, "FAIL legacy model not found: %s\n", legacy_env);
        ++g_failures;
    } else {
        check_legacy_model(legacy_env, pcm);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "canary_timestamp_real_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "canary_timestamp_real_smoke: ok\n");
    return EXIT_SUCCESS;
}
