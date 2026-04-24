// whisper_e2e_smoke.cpp - real-model gated public-ABI test for Whisper.
//
// Covers the runtime capability contract that Stage 4 must not fake:
// language detection is advertised and exercised by running without a
// language hint, segment timestamps are advertised and returned for AUTO,
// and an explicit NONE request keeps the tensor-validation prompt available.

#include "transcribe.h"

#include "wav.h"

#include <sys/stat.h>

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

bool file_exists(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

} // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_WHISPER_MODEL");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "whisper_e2e_smoke: TRANSCRIBE_WHISPER_MODEL not set; skipping.\n");
        return 77;
    }
    const std::string model_path = env;
    if (!file_exists(model_path)) {
        std::fprintf(stderr,
                     "whisper_e2e_smoke: model not found: %s\n",
                     model_path.c_str());
        return 77;
    }

    const std::string wav_path =
        std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    std::vector<float> pcm;
    std::string wav_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, wav_err)) {
        std::fprintf(stderr, "whisper_e2e_smoke: wav load: %s\n", wav_err.c_str());
        return EXIT_FAILURE;
    }

    transcribe_model_params mp = transcribe_model_default_params();
    mp.backend = TRANSCRIBE_BACKEND_CPU;
    struct transcribe_model * model = nullptr;
    transcribe_status st = transcribe_model_load_file(model_path.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }

    CHECK(std::strcmp(transcribe_model_arch_string(model), "whisper") == 0);
    const transcribe_capabilities * caps = transcribe_model_capabilities(model);
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK_EQ_INT(caps->native_sample_rate, 16000);
        CHECK(caps->supports_language_detect);
        CHECK(caps->supports_translate);
        CHECK_EQ_INT(caps->max_timestamp_kind, TRANSCRIBE_TIMESTAMPS_SEGMENT);
        CHECK(caps->n_languages > 0);
        CHECK(caps->languages != nullptr);
    }

    transcribe_context_params cp = transcribe_context_default_params();
    cp.kv_type = TRANSCRIBE_KV_TYPE_F32;
    struct transcribe_context * ctx = nullptr;
    st = transcribe_context_init(model, &cp, &ctx);
    if (st != TRANSCRIBE_OK || ctx == nullptr) {
        std::fprintf(stderr, "FAIL context init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    {
        transcribe_params rp = transcribe_default_params();
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
        CHECK_EQ_INT(transcribe_returned_timestamp_kind(ctx),
                     TRANSCRIBE_TIMESTAMPS_SEGMENT);
        CHECK_EQ_INT(transcribe_n_segments(ctx), 1);
        if (transcribe_n_segments(ctx) == 1) {
            CHECK(transcribe_segment_t1_ms(ctx, 0) > transcribe_segment_t0_ms(ctx, 0));
        }
    }

    {
        transcribe_params rp = transcribe_default_params();
        rp.language = "en";
        rp.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
        CHECK_EQ_INT(transcribe_returned_timestamp_kind(ctx),
                     TRANSCRIBE_TIMESTAMPS_NONE);
    }

    {
        transcribe_params rp = transcribe_default_params();
        rp.task = TRANSCRIBE_TASK_TRANSLATE;
        rp.language = "en";
        rp.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
        CHECK_EQ_INT(transcribe_returned_timestamp_kind(ctx),
                     TRANSCRIBE_TIMESTAMPS_NONE);
    }

    {
        transcribe_params rp = transcribe_default_params();
        rp.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK(st == TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS);
    }

    transcribe_context_free(ctx);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "whisper_e2e_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "whisper_e2e_smoke: ok\n");
    return EXIT_SUCCESS;
}
