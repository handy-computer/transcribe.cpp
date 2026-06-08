// voxtral_realtime_real_smoke.cpp - real-model gated input-limit contract
// test for the Voxtral Realtime family.
//
// voxtral_realtime is a chunked/unbounded (bucket-1) family: its decoder KV is
// a constant-memory sliding ring and the only architectural wall is the
// absolute RoPE position cap (dec_max_position, ~2.9 h of audio). Like whisper
// and parakeet, it deliberately IGNORES the session n_ctx knob. This test
// pins that contract down — it is the regression guard for the "n_ctx is a
// no-op here" decision (see docs/input-limits.md):
//
//   1. capabilities.max_audio_ms == 0 (unbounded).
//   2. transcribe_session_get_limits() reports 0/0/0 at the default context AND
//      under a tiny n_ctx — the knob does not narrow anything.
//   3. One-shot transcribe_run with a tiny n_ctx transcribes normally (n_ctx is
//      a no-op), is not flagged truncated, and yields non-empty text.
//   4. transcribe_run_batch with a tiny n_ctx returns whole-batch OK with every
//      per-utterance status OK and no crash — i.e. the realtime batch decode is
//      NOT cut below its audio horizon (the earlier --n-ctx clamp crash).
//
// Gating:
//   - TRANSCRIBE_BUILD_REAL_MODEL_TESTS (CMake, default OFF) controls whether
//     this binary is built.
//   - At runtime, TRANSCRIBE_VOXTRAL_REALTIME_GGUF points at a Voxtral Realtime
//     GGUF. If unset/missing, exits 77 (CTest "skipped").

#include "transcribe.h"

#include "wav.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
    const char * model_path = std::getenv("TRANSCRIBE_VOXTRAL_REALTIME_GGUF");
    if (model_path == nullptr || *model_path == '\0' || !file_exists(model_path)) {
        std::fprintf(stderr,
                     "skipping: TRANSCRIBE_VOXTRAL_REALTIME_GGUF unset or "
                     "missing\n");
        return 77;
    }

    const std::string sample_path =
        std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    if (!file_exists(sample_path)) {
        std::fprintf(stderr, "skipping: %s missing\n", sample_path.c_str());
        return 77;
    }

    std::vector<float> pcm;
    {
        std::string err;
        if (!transcribe_cli::load_wav_mono_16k(sample_path, pcm, err) ||
            pcm.empty())
        {
            std::fprintf(stderr, "failed to load %s: %s\n",
                         sample_path.c_str(), err.c_str());
            return 1;
        }
    }

    transcribe_model_load_params mp; transcribe_model_load_params_init(&mp);
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(model_path, &mp, &model) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "model load failed: %s\n", model_path);
        return 1;
    }

    // (1) Unbounded capability.
    {
        transcribe_capabilities caps; transcribe_capabilities_init(&caps);
        CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
        CHECK_EQ_INT(caps.max_audio_ms, 0);   // 0 == unbounded
    }

    // (2) Session limits report unbounded at the default context AND under a
    //     tiny n_ctx — the knob is a no-op for this family.
    auto check_unbounded_limits = [&](int32_t n_ctx) {
        transcribe_session_params sp; transcribe_session_params_init(&sp);
        sp.n_ctx = n_ctx;
        struct transcribe_session * s = nullptr;
        CHECK(transcribe_session_init(model, &sp, &s) == TRANSCRIBE_OK);
        transcribe_session_limits lim; transcribe_session_limits_init(&lim);
        CHECK(transcribe_session_get_limits(s, &lim) == TRANSCRIBE_OK);
        CHECK_EQ_INT(lim.effective_n_ctx, 0);          // unbounded
        CHECK_EQ_INT(lim.effective_max_audio_ms, 0);   // unbounded
        CHECK_EQ_INT(lim.max_kv_bytes, 0);             // unbounded
        transcribe_session_free(s);
    };
    check_unbounded_limits(0);    // default
    check_unbounded_limits(64);   // tiny n_ctx — still unbounded (no-op)

    // (3) One-shot with a tiny n_ctx transcribes normally (n_ctx no-op).
    {
        transcribe_session_params sp; transcribe_session_params_init(&sp);
        sp.n_ctx = 64;
        struct transcribe_session * s = nullptr;
        CHECK(transcribe_session_init(model, &sp, &s) == TRANSCRIBE_OK);
        const transcribe_status rst =
            transcribe_run(s, pcm.data(), (int) pcm.size(), nullptr);
        CHECK(rst == TRANSCRIBE_OK);                   // NOT INPUT_TOO_LONG
        CHECK(transcribe_was_truncated(s) == false);   // not truncated
        const char * text = transcribe_full_text(s);
        CHECK(text != nullptr && text[0] != '\0');     // real transcript
        transcribe_session_free(s);
    }

    // (4) Batch with a tiny n_ctx: whole-batch OK, every row OK, no crash.
    //     (Regression guard for the --n-ctx clamp that corrupted the realtime
    //     batch decode below its audio horizon.)
    {
        transcribe_session_params sp; transcribe_session_params_init(&sp);
        sp.n_ctx = 64;
        struct transcribe_session * s = nullptr;
        CHECK(transcribe_session_init(model, &sp, &s) == TRANSCRIBE_OK);
        std::vector<float> pcm2 = pcm;
        const float * pcms[2] = { pcm.data(), pcm2.data() };
        const int     lens[2] = { (int) pcm.size(), (int) pcm2.size() };
        CHECK(transcribe_run_batch(s, pcms, lens, 2, nullptr) == TRANSCRIBE_OK);
        CHECK_EQ_INT(transcribe_batch_n_results(s), 2);
        for (int i = 0; i < 2; ++i) {
            CHECK(transcribe_batch_status(s, i) == TRANSCRIBE_OK);
            const char * text = transcribe_batch_full_text(s, i);
            CHECK(text != nullptr && text[0] != '\0');
        }
        CHECK(transcribe_was_truncated(s) == false);
        transcribe_session_free(s);
    }

    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "voxtral_realtime_real_smoke: %d failures\n",
                     g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "voxtral_realtime_real_smoke: ok\n");
    return EXIT_SUCCESS;
}
