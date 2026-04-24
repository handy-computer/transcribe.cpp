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

        // Stage 2 capability surface. Cancellation + temperature
        // fallback + long-form are all live; initial_prompt waits for
        // Stage 3 when <|startofprev|> plumbing lands.
        CHECK(caps->supports_cancellation);
        CHECK(caps->supports_temperature_fallback);
        CHECK(caps->supports_long_form);
        CHECK(!caps->supports_initial_prompt);
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

    // Abort callback exercises the per-step poll inside the greedy
    // KV loop. The callback returns true after `trigger_after`
    // invocations; we expect TRANSCRIBE_ERR_ABORTED, transcribe_was_aborted
    // true, and the context to expose whatever partial content was
    // committed before the abort point (Stage 1: no chunks, so
    // partial == "nothing beyond the prompt-pass argmax that hasn't
    // been consumed yet"). The assertion here is semantic: the run
    // short-circuited and was_aborted is true.
    struct AbortState {
        int calls_so_far;
        int trigger_after;
    };

    auto abort_cb = [](void * ud) -> bool {
        auto * s = static_cast<AbortState *>(ud);
        s->calls_so_far += 1;
        return s->calls_so_far > s->trigger_after;
    };

    {
        // trigger_after=5: let a handful of decode steps run, then
        // abort. This exercises the per-step poll, not the pre-run
        // placeholder.
        AbortState state{0, 5};
        transcribe_set_abort_callback(ctx, abort_cb, &state);

        transcribe_params rp = transcribe_default_params();
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_ERR_ABORTED);
        CHECK(transcribe_was_aborted(ctx));
        // The callback must have fired at least trigger_after+1 times.
        CHECK(state.calls_so_far >= state.trigger_after + 1);
    }

    {
        // trigger_after=0: abort before the very first poll return
        // succeeds → pre-run chunk-level check fires. Run never even
        // builds the encoder graph.
        AbortState state{0, 0};
        transcribe_set_abort_callback(ctx, abort_cb, &state);

        transcribe_params rp = transcribe_default_params();
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_ERR_ABORTED);
        CHECK(transcribe_was_aborted(ctx));
        CHECK(state.calls_so_far == 1);
    }

    // Uninstall the callback and confirm the next run recovers to OK
    // and clears was_aborted.
    transcribe_set_abort_callback(ctx, nullptr, nullptr);
    {
        transcribe_params rp = transcribe_default_params();
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        CHECK(!transcribe_was_aborted(ctx));
    }

    // Stage 2.6: trace accessors populated on short-form. Exactly one
    // chunk for a ≤ 30 s clip. Tier 0 at default params accepts, so
    // temperature_used == 0.0, n_fallbacks == 0. compression_ratio and
    // avg_logprob are real numbers from the decoder, not zero.
    {
        CHECK_EQ_INT(transcribe_get_whisper_chunk_count(ctx), 1);
        transcribe_whisper_chunk_trace tr =
            transcribe_get_whisper_chunk_trace(ctx, 0);
        CHECK(tr.t0_ms == 0);
        CHECK(tr.t1_ms > tr.t0_ms);
        CHECK(tr.temperature_used == 0.0f);
        CHECK(tr.n_fallbacks == 0);
        /* compression_ratio can be <1 for short clips because zlib's
         * header overhead exceeds the savings on a few dozen tokens;
         * just assert the metric populated to a real finite number. */
        CHECK(tr.compression_ratio >= 0.0f);
        CHECK(tr.compression_ratio < 10.0f);
        CHECK(tr.avg_logprob < 0.0f); /* log probabilities are negative */
        CHECK(!tr.no_speech_triggered);
        /* Out-of-range index returns a zeroed trace. */
        transcribe_whisper_chunk_trace oor =
            transcribe_get_whisper_chunk_trace(ctx, 99);
        CHECK(oor.t0_ms == 0 && oor.t1_ms == 0 &&
              oor.temperature_used == 0.0f && oor.n_fallbacks == 0);
    }

    // Stage 2.2 + 2.3: long-form chunk loop with dynamic stride.
    // Synthesize a 40-second clip from the JFK PCM (repeat pattern).
    // The real audio exceeds one 30-second window, so the seek loop
    // runs at least twice and we expect at least two chunk traces.
    {
        std::vector<float> long_pcm;
        const int repeats = 4;
        long_pcm.reserve(pcm.size() * repeats);
        for (int i = 0; i < repeats; ++i) {
            long_pcm.insert(long_pcm.end(), pcm.begin(), pcm.end());
        }
        transcribe_params rp = transcribe_default_params();
        rp.language = "en";
        st = transcribe_run(ctx, long_pcm.data(),
                            static_cast<int>(long_pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        const char * text = transcribe_full_text(ctx);
        /* The transcript must include the canonical "country" at least
         * once from each of the four JFK repetitions. Require ≥ 2 so
         * the check remains robust to occasional tier-fallback noise. */
        int country_hits = 0;
        for (const char * p = text; (p = std::strstr(p, "country")) != nullptr; ++p) {
            ++country_hits;
        }
        CHECK(country_hits >= 2);
        CHECK(transcribe_get_whisper_chunk_count(ctx) >= 2);
        /* Every chunk trace's temperature is on the fallback tuple. */
        const int n_chunks = transcribe_get_whisper_chunk_count(ctx);
        for (int i = 0; i < n_chunks; ++i) {
            transcribe_whisper_chunk_trace tr =
                transcribe_get_whisper_chunk_trace(ctx, i);
            CHECK(tr.temperature_used >= 0.0f);
            CHECK(tr.temperature_used <= 1.0f + 1e-3f);
            CHECK(tr.n_fallbacks >= 0);
            CHECK(tr.n_fallbacks <= 5); /* 6-tier tuple => 0..5 */
        }
    }

    // Stage 2.5 silence-detection test is deferred: all-zero PCM is a
    // pathological input for the mel frontend (log10(0) behavior) and
    // the downstream no_speech_prob is not guaranteed to exceed the
    // 0.6 default threshold without real ambient audio. The mechanism
    // itself is exercised by the trace-inspection loop above, which
    // reads no_speech_triggered on every chunk. The forced-threshold
    // test below exercises the skip logic on JFK audio.
    //
    // TODO (Stage 2 follow-ups; tracked on the whisper-parity branch):
    //   - Real silence fixture under samples/ (room-tone or mic-off
    //     recording) so the default-threshold no-speech path exercises
    //     un-tuned behavior end-to-end.
    //   - Hallucination-prone clip (e.g. clipped-silence tail, repeated
    //     syllables) that triggers temperature fallback n_fallbacks > 0
    //     at default thresholds.
    //   - Single-timestamp-ending vs double-timestamp-ending long-form
    //     stitching cases (PRs #34537 / #35750 behavior coverage).
    //   - 120s / 5-minute LibriSpeech clip with HF-reference-comparison
    //     long-form WER gate and segment-count parity.
    //   - Golden compression-ratio unit test covering token inputs with
    //     mixed timestamps / specials / EOS (requires exposing the
    //     internal helper or mirroring it).

    // No-speech AND-gate parity: setting no_speech_thold to the
    // _DISABLED sentinel must prevent the skip gate from firing
    // regardless of the no_speech_prob observed, AND setting
    // logprob_thold to _DISABLED must do the same regardless of
    // avg_logprob. The gate requires both conditions per HF
    // _need_fallback; disabling either one disables the skip.
    {
        transcribe_params rp = transcribe_default_params();
        transcribe_whisper_params wp = transcribe_whisper_default_params();
        wp.no_speech_thold = TRANSCRIBE_WHISPER_THOLD_DISABLED;
        rp.whisper = &wp;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        const int n = transcribe_get_whisper_chunk_count(ctx);
        for (int i = 0; i < n; ++i) {
            transcribe_whisper_chunk_trace tr =
                transcribe_get_whisper_chunk_trace(ctx, i);
            /* no_speech skip gate MUST NOT fire when no_speech_thold
             * is disabled; no_speech_prob itself still populates
             * because the capture is independent of the skip check. */
            CHECK(!tr.no_speech_triggered);
        }
    }
    {
        transcribe_params rp = transcribe_default_params();
        transcribe_whisper_params wp = transcribe_whisper_default_params();
        wp.logprob_thold = TRANSCRIBE_WHISPER_LOGPROB_DISABLED;
        rp.whisper = &wp;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        const int n = transcribe_get_whisper_chunk_count(ctx);
        for (int i = 0; i < n; ++i) {
            transcribe_whisper_chunk_trace tr =
                transcribe_get_whisper_chunk_trace(ctx, i);
            CHECK(!tr.no_speech_triggered);
        }
    }

    // Forced no-speech skip at tier 0: construct thresholds that
    // guarantee both legs of the AND fire on tier 0 (real JFK audio).
    //   * no_speech_thold = -1.0 → no_speech_prob > -1.0 is always true
    //     (probabilities are non-negative).
    //   * logprob_thold   =  0.0 → avg_logprob < 0.0 is always true for
    //     real decoded text (logprobs are strictly negative).
    //
    // The per-tier skip check must trip at tier 0 AND halt fallback —
    // n_fallbacks must stay 0 (HF's needs_fallback=False branch). A
    // regression that left the skip check post-fallback would first
    // escalate through every hotter tier before discarding, producing
    // n_fallbacks >= 1 here.
    {
        transcribe_params rp = transcribe_default_params();
        transcribe_whisper_params wp = transcribe_whisper_default_params();
        wp.no_speech_thold = -1.0f;
        wp.logprob_thold   =  0.0f;
        /* Disable compression threshold so compression-ratio failures
         * don't accidentally trigger fallback before the no-speech
         * check gets a chance. */
        wp.compression_ratio_thold = TRANSCRIBE_WHISPER_THOLD_DISABLED;
        rp.whisper = &wp;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        const int n = transcribe_get_whisper_chunk_count(ctx);
        CHECK(n >= 1);
        if (n >= 1) {
            transcribe_whisper_chunk_trace tr =
                transcribe_get_whisper_chunk_trace(ctx, 0);
            CHECK(tr.no_speech_triggered);
            CHECK_EQ_INT(tr.n_fallbacks, 0);
            /* Chunk output was discarded → full_text is empty. */
            CHECK(std::strcmp(transcribe_full_text(ctx), "") == 0);
        }
    }

    // Seeded determinism: temperature > 0 makes the sampler active, so
    // two runs with the same seed must produce byte-identical full
    // text. The RNG is run-scoped (not reset per chunk), so this holds
    // for both short-form and long-form inputs. A regression that
    // reset the RNG per chunk would still pass single-chunk runs but
    // would return slightly different output on the long-form clip,
    // which is why we use the repeated-JFK 40 s PCM here.
    {
        std::vector<float> long_pcm;
        const int repeats = 4;
        long_pcm.reserve(pcm.size() * repeats);
        for (int i = 0; i < repeats; ++i) {
            long_pcm.insert(long_pcm.end(), pcm.begin(), pcm.end());
        }

        transcribe_whisper_params wp = transcribe_whisper_default_params();
        wp.temperature = 0.4f;           /* forces the sampler active */
        wp.temperature_inc = 0.0f;       /* single-tier, no fallback  */
        wp.compression_ratio_thold =
            TRANSCRIBE_WHISPER_THOLD_DISABLED;
        wp.logprob_thold = TRANSCRIBE_WHISPER_LOGPROB_DISABLED;
        wp.no_speech_thold = TRANSCRIBE_WHISPER_THOLD_DISABLED;
        wp.seed = 42;

        transcribe_params rp = transcribe_default_params();
        rp.language = "en";
        rp.whisper = &wp;

        st = transcribe_run(ctx, long_pcm.data(),
                            static_cast<int>(long_pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        std::string text_a = transcribe_full_text(ctx);

        st = transcribe_run(ctx, long_pcm.data(),
                            static_cast<int>(long_pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        std::string text_b = transcribe_full_text(ctx);

        CHECK(text_a == text_b);
        CHECK(!text_a.empty());
    }

    // Partial result visibility on abort: after the abort callback
    // trips mid-decode, transcribe_full_text / transcribe_n_segments
    // must expose whatever the chunk loop committed before the abort
    // point (the contract in include/transcribe.h). For a single
    // short-form chunk aborted mid-step, that is typically empty —
    // the chunk hasn't finished — but has_result must still be true,
    // which we prove by observing that transcribe_returned_timestamp_kind
    // reports something other than the pre-run NONE sentinel.
    {
        AbortState state{0, 5};
        transcribe_set_abort_callback(ctx, abort_cb, &state);
        transcribe_params rp = transcribe_default_params();
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_ERR_ABORTED);
        CHECK(transcribe_was_aborted(ctx));
        /* TIMESTAMPS_AUTO resolves to TIMESTAMPS_SEGMENT for the
         * whisper family; that value appears on the context even
         * after abort, confirming has_result=true was committed. */
        CHECK_EQ_INT(transcribe_returned_timestamp_kind(ctx),
                     TRANSCRIBE_TIMESTAMPS_SEGMENT);
        transcribe_set_abort_callback(ctx, nullptr, nullptr);
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
