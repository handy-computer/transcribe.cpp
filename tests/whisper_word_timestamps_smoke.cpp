// Real-model public ABI check for Whisper's opt-in word timestamps.

#include "transcribe.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

// The bundled JFK clip contains 22 words and ends near 10.4 s. Also check
// linkage, coverage, and forward motion so flat but in-range DTW fails.
void check_words(const transcribe_session * ctx, int64_t audio_ms) {
    CHECK(transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_WORD);
    CHECK(transcribe_n_words(ctx) >= 20);
    CHECK(transcribe_n_tokens(ctx) == 0);
    int     seen            = 0;
    int     advancing_words = 0;
    int64_t previous_end    = 0;
    for (int i = 0; i < transcribe_n_segments(ctx); ++i) {
        transcribe_segment segment;
        transcribe_segment_init(&segment);
        CHECK(transcribe_get_segment(ctx, i, &segment) == TRANSCRIBE_OK);
        CHECK(segment.first_word == seen);
        CHECK(segment.n_words > 0);
        for (int j = 0; j < segment.n_words; ++j) {
            transcribe_word word;
            transcribe_word_init(&word);
            CHECK(transcribe_get_word(ctx, seen++, &word) == TRANSCRIBE_OK);
            CHECK(word.seg_index == i);
            CHECK(word.n_tokens == 0);
            CHECK(word.text != nullptr && word.text[0] != '\0');
            CHECK(word.t0_ms >= previous_end && word.t1_ms >= word.t0_ms);
            CHECK(word.t0_ms >= segment.t0_ms && word.t1_ms <= segment.t1_ms);
            CHECK(word.t1_ms <= audio_ms);
            advancing_words += word.t1_ms > word.t0_ms;
            previous_end = word.t1_ms;
        }
    }
    CHECK(seen == transcribe_n_words(ctx));
    CHECK(advancing_words + 2 >= seen);
    CHECK(previous_end > 9000 && previous_end < 10900);
}

}  // namespace

int main() {
    const char * path = std::getenv("TRANSCRIBE_WHISPER_WORD_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::fprintf(stderr, "whisper_word_timestamps_smoke: TRANSCRIBE_WHISPER_WORD_GGUF not set; skipping.\n");
        return 77;
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend              = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * raw  = nullptr;
    const auto         load = transcribe_model_load_file(path, &mp, &raw);
    if (load != TRANSCRIBE_OK || raw == nullptr) {
        std::fprintf(stderr, "FAIL load: %s\n", transcribe_status_string(load));
        return EXIT_FAILURE;
    }
    const std::unique_ptr<transcribe_model, decltype(&transcribe_model_free)> model(raw, transcribe_model_free);

    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    if (transcribe_model_get_capabilities(model.get(), &caps) != TRANSCRIBE_OK ||
        std::strcmp(transcribe_model_arch_string(model.get()), "whisper") != 0) {
        return EXIT_FAILURE;
    }
    // This fixture is intended for stock Whisper GGUFs with known heads.
    if (caps.max_timestamp_kind != TRANSCRIBE_TIMESTAMPS_WORD) {
        std::fprintf(stderr, "whisper_word_timestamps_smoke: model does not advertise word timestamps.\n");
        return EXIT_FAILURE;
    }

    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    sp.n_threads                     = 4;
    sp.kv_type                       = TRANSCRIBE_KV_TYPE_F32;
    transcribe_session * raw_session = nullptr;
    if (transcribe_session_init(model.get(), &sp, &raw_session) != TRANSCRIBE_OK || raw_session == nullptr) {
        return EXIT_FAILURE;
    }
    const std::unique_ptr<transcribe_session, decltype(&transcribe_session_free)> session(raw_session,
                                                                                          transcribe_session_free);
    auto *                                                                        ctx = session.get();

    const std::string  sample = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    std::vector<float> pcm;
    std::string        error;
    if (!transcribe_cli::load_wav_mono_16k(sample, pcm, error)) {
        std::fprintf(stderr, "FAIL wav load: %s\n", error.c_str());
        return EXIT_FAILURE;
    }

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    rp.language         = "en";
    const int n_samples = static_cast<int>(pcm.size());
    CHECK(transcribe_run(ctx, pcm.data(), n_samples, &rp) == TRANSCRIBE_OK);
    CHECK(transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_SEGMENT);
    CHECK(transcribe_n_words(ctx) == 0);
    const std::string text = transcribe_full_text(ctx);

    rp.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
    CHECK(transcribe_run(ctx, pcm.data(), n_samples, &rp) == TRANSCRIBE_OK);
    CHECK(text == transcribe_full_text(ctx));
    check_words(ctx, static_cast<int64_t>(n_samples) * 1000 / 16000);

    rp.prefix            = "And so";
    const int word_count = transcribe_n_words(ctx);
    CHECK(transcribe_run(ctx, pcm.data(), n_samples, &rp) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_n_words(ctx) == word_count);
    rp.prefix = nullptr;

    // The second utterance crosses the 30 s internal window. Its second
    // copy of the speech must have absolute word times in the later window.
    std::vector<float> delayed = pcm;
    delayed.resize(32 * 16000);
    delayed.insert(delayed.end(), pcm.begin(), pcm.end());
    const float * audio[]   = { pcm.data(), delayed.data() };
    const int     lengths[] = { n_samples, static_cast<int>(delayed.size()) };
    CHECK(transcribe_run_batch(ctx, audio, lengths, 2, &rp) == TRANSCRIBE_OK);
    CHECK(transcribe_batch_n_results(ctx) == 2);
    CHECK(transcribe_batch_n_words(ctx, 0) == word_count);
    CHECK(transcribe_batch_returned_timestamp_kind(ctx, 1) == TRANSCRIBE_TIMESTAMPS_WORD);
    const int delayed_words = transcribe_batch_n_words(ctx, 1);
    CHECK(delayed_words > word_count);
    if (delayed_words > 0) {
        transcribe_word last;
        transcribe_word_init(&last);
        CHECK(transcribe_batch_get_word(ctx, 1, delayed_words - 1, &last) == TRANSCRIBE_OK);
        CHECK(last.t1_ms > 40000 && last.t1_ms <= static_cast<int64_t>(lengths[1]) * 1000 / 16000);
    }

    rp.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
    CHECK(transcribe_run(ctx, pcm.data(), n_samples, &rp) == TRANSCRIBE_OK);
    CHECK(transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_SEGMENT);
    CHECK(transcribe_n_words(ctx) == 0);

    if (caps.supports_language_detect) {
        const std::string japanese = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/fleurs-ja.wav";
        if (!transcribe_cli::load_wav_mono_16k(japanese, pcm, error)) {
            std::fprintf(stderr, "FAIL Japanese wav load: %s\n", error.c_str());
            return EXIT_FAILURE;
        }
        rp.language   = "ja";
        rp.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
        CHECK(transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
        CHECK(transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_WORD);
        // A no-space language should expose multiple complete UTF-8 pieces.
        CHECK(transcribe_n_words(ctx) > 5);
    }

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
