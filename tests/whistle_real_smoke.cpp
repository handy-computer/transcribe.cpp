// whistle_real_smoke.cpp - real-model gated behavior + input/memory contract
// test for the Whistle family.
//
// Gating: built only with TRANSCRIBE_BUILD_REAL_MODEL_TESTS; at runtime the
// GGUF comes from TRANSCRIBE_WHISTLE_GGUF (exit 77 = skipped when unset).
// Runs on the CPU backend (the Stage 4 correctness regime).
//
// Asserts (expected values are the closed Needle engine's outputs, see
// docs/porting/families/whistle.md):
//   1. arch "whistle", 7 languages, word timestamps advertised, 30 s cap.
//   2. jfk with language "en": exact engine transcript.
//   3. jfk with no language: auto-detects "en", same transcript.
//   4. noise.wav: no-speech gate -> OK, empty transcript.
//   5. 30 s + 1 sample: TRANSCRIBE_ERR_INPUT_TOO_LONG before any decode.
//   6. n_ctx = 8: TRANSCRIBE_ERR_OUTPUT_TRUNCATED with a non-empty prefix of
//      the full transcript (n_ctx lowers the decoder context, never raised).
//   7. word timestamps on jfk: 22 words, monotone, within the clip.
//   8. run_batch over {jfk, noise, jfk}: per-utterance results equal the
//      single-shot ones.

#include "transcribe.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

const char * k_jfk_text =
    "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

std::vector<float> load(const std::string & name) {
    std::vector<float> pcm;
    std::string        err;
    const std::string  path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/" + name;
    if (!transcribe_cli::load_wav_mono_16k(path, pcm, err)) {
        std::fprintf(stderr, "whistle_real_smoke: cannot read %s: %s\n", path.c_str(), err.c_str());
        std::exit(EXIT_FAILURE);
    }
    return pcm;
}

transcribe_session * open_session(transcribe_model * model, int n_ctx) {
    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    sp.n_ctx                  = n_ctx;
    transcribe_session * sess = nullptr;
    if (transcribe_session_init(model, &sp, &sess) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL session_init\n");
        std::exit(EXIT_FAILURE);
    }
    return sess;
}

}  // namespace

int main() {
    const char * gguf = std::getenv("TRANSCRIBE_WHISTLE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::fprintf(stderr,
                     "whistle_real_smoke: TRANSCRIBE_WHISTLE_GGUF not set; skipping. Convert with:\n"
                     "  uv run --project scripts/envs/whistle scripts/convert-whistle.py Cactus-Compute/whistle "
                     "--repo-id Cactus-Compute/whistle\n"
                     "and re-run with TRANSCRIBE_WHISTLE_GGUF=models/whistle/whistle-F32.gguf\n");
        return 77;
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend               = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * model = nullptr;
    if (transcribe_model_load_file(gguf, &mp, &model) != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load %s\n", gguf);
        return EXIT_FAILURE;
    }

    // 1. Identity + capabilities.
    CHECK(std::strcmp(transcribe_model_arch_string(model), "whistle") == 0);
    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
    CHECK(caps.native_sample_rate == 16000);
    CHECK(caps.n_languages == 7);
    CHECK(caps.supports_language_detect);
    CHECK(!caps.supports_translate);
    CHECK(caps.max_timestamp_kind == TRANSCRIBE_TIMESTAMPS_WORD);
    CHECK(caps.max_audio_ms == 30000);

    const std::vector<float> jfk   = load("jfk.wav");
    const std::vector<float> noise = load("noise.wav");

    transcribe_session *  sess = open_session(model, 0);
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);

    // 2. Explicit language.
    rp.language = "en";
    CHECK(transcribe_run(sess, jfk.data(), static_cast<int>(jfk.size()), &rp) == TRANSCRIBE_OK);
    const std::string jfk_en = transcribe_full_text(sess);
    CHECK(jfk_en == k_jfk_text);

    // 3. Auto language.
    rp.language = nullptr;
    CHECK(transcribe_run(sess, jfk.data(), static_cast<int>(jfk.size()), &rp) == TRANSCRIBE_OK);
    CHECK(std::string(transcribe_full_text(sess)) == k_jfk_text);
    CHECK(std::string(transcribe_detected_language(sess)) == "en");

    // 4. No speech.
    CHECK(transcribe_run(sess, noise.data(), static_cast<int>(noise.size()), &rp) == TRANSCRIBE_OK);
    CHECK(std::string(transcribe_full_text(sess)).empty());

    // 5. Hard input cap: 30 s + 1 sample.
    {
        std::vector<float> long_pcm(480001, 0.0f);
        for (size_t i = 0; i < long_pcm.size(); ++i) {
            long_pcm[i] = jfk[i % jfk.size()];
        }
        CHECK(transcribe_run(sess, long_pcm.data(), static_cast<int>(long_pcm.size()), &rp) ==
              TRANSCRIBE_ERR_INPUT_TOO_LONG);
    }

    // 6. n_ctx lowers the decoder context; truncation keeps the prefix.
    {
        transcribe_session * small = open_session(model, 8);
        rp.language                = "en";
        CHECK(transcribe_run(small, jfk.data(), static_cast<int>(jfk.size()), &rp) == TRANSCRIBE_ERR_OUTPUT_TRUNCATED);
        CHECK(transcribe_was_truncated(small));
        const std::string partial = transcribe_full_text(small);
        CHECK(!partial.empty() && jfk_en.compare(0, partial.size(), partial) == 0);
        transcribe_session_free(small);
    }

    // 7. Word timestamps.
    rp.language   = "en";
    rp.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
    CHECK(transcribe_run(sess, jfk.data(), static_cast<int>(jfk.size()), &rp) == TRANSCRIBE_OK);
    CHECK(transcribe_n_words(sess) == 22);
    int64_t prev = 0;
    for (int i = 0; i < transcribe_n_words(sess); ++i) {
        transcribe_word w;
        transcribe_word_init(&w);
        CHECK(transcribe_get_word(sess, i, &w) == TRANSCRIBE_OK);
        CHECK(w.t0_ms >= prev && w.t1_ms >= w.t0_ms && w.t1_ms <= 11000);
        prev = w.t0_ms;
    }

    // 8. Batch == single.
    rp.timestamps         = TRANSCRIBE_TIMESTAMPS_NONE;
    rp.language           = nullptr;
    const float * pcms[3] = { jfk.data(), noise.data(), jfk.data() };
    const int ns[3] = { static_cast<int>(jfk.size()), static_cast<int>(noise.size()), static_cast<int>(jfk.size()) };
    CHECK(transcribe_run_batch(sess, pcms, ns, 3, &rp) == TRANSCRIBE_OK);
    CHECK(transcribe_batch_n_results(sess) == 3);
    if (transcribe_batch_n_results(sess) == 3) {
        CHECK(std::string(transcribe_batch_full_text(sess, 0)) == k_jfk_text);
        CHECK(std::string(transcribe_batch_full_text(sess, 1)).empty());
        CHECK(std::string(transcribe_batch_full_text(sess, 2)) == k_jfk_text);
        CHECK(std::string(transcribe_batch_detected_language(sess, 0)) == "en");
    }

    transcribe_session_free(sess);
    transcribe_model_free(model);
    if (g_failures != 0) {
        std::fprintf(stderr, "whistle_real_smoke: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "whistle_real_smoke: ok\n");
    return 0;
}
