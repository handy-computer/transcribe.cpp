// qwen3_asr_batch_truncation.cpp - real-model gated test that the single-shot
// and batch decode paths both report mid-decode OUTPUT_TRUNCATED for a
// causal_lm (LLM-decoder) family.
//
// qwen3_asr caps generation at max_new = 256 tokens. A long speech clip passes
// the up-front input-length gate (its audio tokens fit the 65536-token decoder
// context with room to spare) but its natural transcript exceeds 256 tokens, so
// greedy decode hits the generation budget before EOS — the transcript is
// truncated. Per docs/input-limits.md that must surface as the hard
// TRANSCRIBE_ERR_OUTPUT_TRUNCATED status (partial transcript retained,
// transcribe_was_truncated() set) in BOTH paths, while a short clip that
// finishes under the budget stays OK and the whole-batch call still returns OK.
//
// This is the causal_lm counterpart to moonshine_streaming_batch_truncation
// (which exercises the encoder-decoder batch loop in transcribe-batch-util.cpp).
// It specifically guards the shared src/causal_lm batched step loop's per-row
// truncation detection: that loop marks every stopped row `finished`
// regardless of WHY it stopped, so truncation must be inferred from the last
// sampled token (!= eos), not from `!finished`. A regression there makes a
// truncated batch row silently report TRANSCRIBE_OK with an incomplete
// transcript — the exact failure this test catches.
//
// Batch makeup:
//   row 0 = jfk.wav (~11 s)        -> completes under the budget -> OK
//   row 1 = love-loss.wav (~197 s) -> exceeds the budget         -> OUTPUT_TRUNCATED
//
// Gating:
//   - TRANSCRIBE_BUILD_REAL_MODEL_TESTS (CMake, default OFF) builds it.
//   - At runtime, TRANSCRIBE_QWEN3_ASR_0_6B_GGUF points at the GGUF. If
//     unset/missing (or a sample is missing), exits 77 ("skipped").

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

} // namespace

int main() {
    const char * model_path = std::getenv("TRANSCRIBE_QWEN3_ASR_0_6B_GGUF");
    if (model_path == nullptr || *model_path == '\0' || !file_exists(model_path)) {
        std::fprintf(stderr,
                     "skipping: TRANSCRIBE_QWEN3_ASR_0_6B_GGUF unset or missing\n");
        return 77;
    }

    std::vector<float> pcm_short, pcm_long;
    if (!load_sample("jfk.wav", pcm_short)) return 77;
    if (!load_sample("love-loss.wav", pcm_long)) return 77;

    transcribe_model_load_params mp; transcribe_model_load_params_init(&mp);
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(model_path, &mp, &model) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "model load failed: %s\n", model_path);
        return 1;
    }

    transcribe_session_params sp; transcribe_session_params_init(&sp);
    struct transcribe_session * s = nullptr;
    if (transcribe_session_init(model, &sp, &s) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "session init failed\n");
        transcribe_model_free(model);
        return 1;
    }

    // ---- Single-shot baseline: the long clip truncates, the short one does not.
    // Both pass the input-length gate at the default (full) context; the long
    // clip simply runs the decoder into the 256-token generation budget.
    {
        const transcribe_status rl = transcribe_run(
            s, pcm_long.data(), (int) pcm_long.size(), nullptr);
        CHECK(rl == TRANSCRIBE_ERR_OUTPUT_TRUNCATED);
        CHECK(transcribe_was_truncated(s) == true);
        const char * t = transcribe_full_text(s);
        CHECK(t != nullptr && t[0] != '\0');   // partial transcript retained
    }
    {
        const transcribe_status rs = transcribe_run(
            s, pcm_short.data(), (int) pcm_short.size(), nullptr);
        CHECK(rs == TRANSCRIBE_OK);
        CHECK(transcribe_was_truncated(s) == false);  // reset + completed
    }

    // ---- Batch parity: the shared causal_lm batched step loop must report the
    // SAME per-utterance verdict. row 0 (short) finishes -> OK; row 1 (long)
    // hits the budget -> OUTPUT_TRUNCATED; whole-batch call still returns OK.
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
        std::fprintf(stderr, "qwen3_asr_batch_truncation: %d failures\n",
                     g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "qwen3_asr_batch_truncation: ok\n");
    return EXIT_SUCCESS;
}
