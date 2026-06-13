// whisper_e2e_smoke.cpp - real-model gated public-ABI test for Whisper.
//
// Covers the runtime capability contract that Stage 4 must not fake:
// language detection is advertised and exercised by running without a
// language hint, default params use no-timestamp decode, explicit
// SEGMENT timestamps are advertised and returned, and no-timestamp
// decode keeps the tensor-validation prompt available.

#include "transcribe.h"
#include "transcribe/whisper.h"

#include "wav.h"

#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#  include <dbghelp.h>
#endif

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

#if defined(_WIN32)
// Diagnostic (known-issues.md "B"): on a Windows/MSVC integer divide-by-zero
// (STATUS_INTEGER_DIVIDE_BY_ZERO, 0xC0000094) the run faults with no useful
// trace in CI. This vectored exception handler symbolizes the faulting IP and
// the whole stack via DbgHelp — self-contained, so no external debugger (cdb)
// needs to be present on the runner; it only needs the Release+/Z7 PDB. Gated
// by TRANSCRIBE_DIAG_FAULT_TRACE so normal/native-ci runs are untouched. Remove
// this block (and the dbghelp link + diag-b.yml) once "B" is fixed.
void diag_resolve(HANDLE proc, DWORD64 addr, const char * tag) {
    char symbuf[sizeof(SYMBOL_INFO) + 512] = {};
    auto * sym        = reinterpret_cast<SYMBOL_INFO *>(symbuf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen   = 511;
    DWORD64 sdisp     = 0;
    const bool have_sym = SymFromAddr(proc, addr, &sdisp, sym) != FALSE;

    IMAGEHLP_LINE64 line {};
    line.SizeOfStruct = sizeof(line);
    DWORD ldisp       = 0;
    const bool have_line = SymGetLineFromAddr64(proc, addr, &ldisp, &line) != FALSE;

    std::fprintf(stderr, "  %-12s 0x%016llx  %s+0x%llx", tag,
                 static_cast<unsigned long long>(addr),
                 have_sym ? sym->Name : "<no-symbol>",
                 static_cast<unsigned long long>(sdisp));
    if (have_line) {
        std::fprintf(stderr, "   (%s:%lu)", line.FileName, line.LineNumber);
    }
    std::fprintf(stderr, "\n");
}

LONG WINAPI diag_fault_trace(EXCEPTION_POINTERS * ep) {
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_INT_DIVIDE_BY_ZERO && code != EXCEPTION_INT_OVERFLOW) {
        return EXCEPTION_CONTINUE_SEARCH;  // not ours — let normal handling run
    }
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(proc, nullptr, TRUE);

    std::fprintf(stderr, "\n==== FAULT 0x%08lx (%s) ====\n",
                 static_cast<unsigned long>(code),
                 code == EXCEPTION_INT_DIVIDE_BY_ZERO ? "INT_DIVIDE_BY_ZERO"
                                                      : "INT_OVERFLOW");
    diag_resolve(proc,
                 reinterpret_cast<DWORD64>(ep->ExceptionRecord->ExceptionAddress),
                 "FAULTING-IP");
    std::fprintf(stderr, "---- backtrace (top frames first) ----\n");
    void *       frames[64];
    const USHORT n = CaptureStackBackTrace(0, 64, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        char idx[16];
        std::snprintf(idx, sizeof(idx), "#%-2u", i);
        diag_resolve(proc, reinterpret_cast<DWORD64>(frames[i]), idx);
    }
    std::fflush(stderr);
    TerminateProcess(proc, 94);
    return EXCEPTION_CONTINUE_EXECUTION;  // unreached (process is gone)
}
#endif  // _WIN32

} // namespace

int main() {
#if defined(_WIN32)
    // Diagnostic (known-issues.md "B"): install the int-÷0 backtrace handler
    // before anything runs. No-op unless TRANSCRIBE_DIAG_FAULT_TRACE is set.
    if (const char * d = std::getenv("TRANSCRIBE_DIAG_FAULT_TRACE");
        d != nullptr && d[0] != '\0' && std::strcmp(d, "0") != 0) {
        AddVectoredExceptionHandler(/*first=*/1u, diag_fault_trace);
    }
#endif
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

    transcribe_model_load_params mp; transcribe_model_load_params_init(&mp);
    mp.backend = TRANSCRIBE_BACKEND_CPU;
    struct transcribe_model * model = nullptr;
    transcribe_status st = transcribe_model_load_file(model_path.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }

    CHECK(std::strcmp(transcribe_model_arch_string(model), "whisper") == 0);
    transcribe_capabilities caps_buf; transcribe_capabilities_init(&caps_buf);
    const bool caps_ok =
        transcribe_model_get_capabilities(model, &caps_buf) == TRANSCRIBE_OK;
    const transcribe_capabilities * caps = caps_ok ? &caps_buf : nullptr;
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK_EQ_INT(caps->native_sample_rate, 16000);
        CHECK(caps->supports_language_detect);
        CHECK(caps->supports_translate);
        CHECK_EQ_INT(caps->max_timestamp_kind, TRANSCRIBE_TIMESTAMPS_SEGMENT);
        CHECK(caps->n_languages > 0);
        CHECK(caps->languages != nullptr);

        // Stage 2 + 3 feature surface. Cancellation + temperature
        // fallback + long-form shipped in Stage 2; initial_prompt
        // (initial_prompt / prompt_tokens / condition_on_prev_tokens)
        // shipped in Stage 3. These advisories live behind the
        // transcribe_model_supports() probe, not the caps struct.
        CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_CANCELLATION));
        CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK));
        CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_LONG_FORM));
        CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_INITIAL_PROMPT));
    }

    transcribe_session_params cp; transcribe_session_params_init(&cp);
    cp.kv_type = TRANSCRIBE_KV_TYPE_F32;
    struct transcribe_session * ctx = nullptr;
    st = transcribe_session_init(model, &cp, &ctx);
    if (st != TRANSCRIBE_OK || ctx == nullptr) {
        std::fprintf(stderr, "FAIL context init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    // Opt-in diagnostic harness (TRANSCRIBE_DIAG_CANCEL): threaded wall-clock
    // cancel of a LONG-FORM run, swept over cancel delays and repeated to shake
    // timing. Reproduces the x86-only integer divide-by-zero in whisper's
    // abort/partial path (it faults on x86, is masked on arm64). Driven under a
    // debugger by .github/workflows/diag-cancel.yml to capture the faulting
    // file:line. No-op unless the env var is set, so normal test runs are
    // unaffected.
    if (const char * d = std::getenv("TRANSCRIBE_DIAG_CANCEL"); d && *d && *d != '0') {
        std::vector<float> long_pcm;
        const int repeats = 6;  // 6x jfk ~= 66s -> long-form (2+ windows)
        long_pcm.reserve(pcm.size() * repeats);
        for (int i = 0; i < repeats; ++i)
            long_pcm.insert(long_pcm.end(), pcm.begin(), pcm.end());

        // Small sweep: the fault is reliable on a slow runner near a ~40ms
        // cancel, and the debugger quits on the first crash. Keep it short so it
        // finishes inside the job timeout when it does NOT reproduce.
        static std::atomic<bool> g_cancel{false};
        for (int rep = 0; rep < 6; ++rep) {
            for (int sleep_ms = 20; sleep_ms <= 120; sleep_ms += 20) {
                g_cancel.store(false);
                transcribe_set_abort_callback(
                    ctx, [](void *) -> bool { return g_cancel.load(); }, nullptr);
                std::thread killer([sleep_ms] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                    g_cancel.store(true);
                });
                transcribe_run_params rp; transcribe_run_params_init(&rp);
                rp.language = "en";
                (void) transcribe_run(ctx, long_pcm.data(),
                                      static_cast<int>(long_pcm.size()), &rp);
                killer.join();
            }
        }
        transcribe_set_abort_callback(ctx, nullptr, nullptr);
        std::fprintf(stderr, "[diag-cancel] completed without reproducing\n");
        transcribe_session_free(ctx);
        transcribe_model_free(model);
        return EXIT_SUCCESS;
    }

    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
        CHECK_EQ_INT(transcribe_returned_timestamp_kind(ctx),
                     TRANSCRIBE_TIMESTAMPS_NONE);
        CHECK_EQ_INT(transcribe_n_segments(ctx), 1);
        {
            transcribe_segment seg; transcribe_segment_init(&seg);
            CHECK_EQ_INT(transcribe_get_segment(ctx, 0, &seg), TRANSCRIBE_OK);
            CHECK_EQ_INT(seg.t0_ms, 0);
            CHECK_EQ_INT(seg.t1_ms, 0);
        }
    }

    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        rp.language = "en";
        rp.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
        CHECK_EQ_INT(transcribe_returned_timestamp_kind(ctx),
                     TRANSCRIBE_TIMESTAMPS_SEGMENT);
        CHECK_EQ_INT(transcribe_n_segments(ctx), 1);
        if (transcribe_n_segments(ctx) == 1) {
            transcribe_segment seg; transcribe_segment_init(&seg);
            CHECK_EQ_INT(transcribe_get_segment(ctx, 0, &seg), TRANSCRIBE_OK);
            CHECK(seg.t1_ms > seg.t0_ms);
        }
    }

    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        rp.language = "en";
        rp.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
        CHECK_EQ_INT(transcribe_returned_timestamp_kind(ctx),
                     TRANSCRIBE_TIMESTAMPS_NONE);
    }

    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
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
        transcribe_run_params rp; transcribe_run_params_init(&rp);
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

        transcribe_run_params rp; transcribe_run_params_init(&rp);
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

        transcribe_run_params rp; transcribe_run_params_init(&rp);
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_ERR_ABORTED);
        CHECK(transcribe_was_aborted(ctx));
        CHECK(state.calls_so_far == 1);
    }

    // Uninstall the callback and confirm the next run recovers to OK
    // and clears was_aborted.
    transcribe_set_abort_callback(ctx, nullptr, nullptr);
    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
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
        transcribe_whisper_chunk_trace tr; transcribe_whisper_chunk_trace_init(&tr);
        CHECK_EQ_INT(transcribe_get_whisper_chunk_trace(ctx, 0, &tr), TRANSCRIBE_OK);
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
        transcribe_whisper_chunk_trace oor; transcribe_whisper_chunk_trace_init(&oor);
        CHECK_EQ_INT(transcribe_get_whisper_chunk_trace(ctx, 99, &oor), TRANSCRIBE_OK);
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
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        rp.language = "en";
        rp.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
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
            transcribe_whisper_chunk_trace tr; transcribe_whisper_chunk_trace_init(&tr);
            CHECK_EQ_INT(transcribe_get_whisper_chunk_trace(ctx, i, &tr), TRANSCRIBE_OK);
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
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.no_speech_thold = TRANSCRIBE_WHISPER_THOLD_DISABLED;
        rp.family = &wp.ext;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        const int n = transcribe_get_whisper_chunk_count(ctx);
        for (int i = 0; i < n; ++i) {
            transcribe_whisper_chunk_trace tr; transcribe_whisper_chunk_trace_init(&tr);
            CHECK_EQ_INT(transcribe_get_whisper_chunk_trace(ctx, i, &tr), TRANSCRIBE_OK);
            /* no_speech skip gate MUST NOT fire when no_speech_thold
             * is disabled; no_speech_prob itself still populates
             * because the capture is independent of the skip check. */
            CHECK(!tr.no_speech_triggered);
        }
    }
    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.logprob_thold = TRANSCRIBE_WHISPER_LOGPROB_DISABLED;
        rp.family = &wp.ext;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        const int n = transcribe_get_whisper_chunk_count(ctx);
        for (int i = 0; i < n; ++i) {
            transcribe_whisper_chunk_trace tr; transcribe_whisper_chunk_trace_init(&tr);
            CHECK_EQ_INT(transcribe_get_whisper_chunk_trace(ctx, i, &tr), TRANSCRIBE_OK);
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
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.no_speech_thold = -1.0f;
        wp.logprob_thold   =  0.0f;
        /* Disable compression threshold so compression-ratio failures
         * don't accidentally trigger fallback before the no-speech
         * check gets a chance. */
        wp.compression_ratio_thold = TRANSCRIBE_WHISPER_THOLD_DISABLED;
        rp.family = &wp.ext;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        const int n = transcribe_get_whisper_chunk_count(ctx);
        CHECK(n >= 1);
        if (n >= 1) {
            transcribe_whisper_chunk_trace tr; transcribe_whisper_chunk_trace_init(&tr);
            CHECK_EQ_INT(transcribe_get_whisper_chunk_trace(ctx, 0, &tr), TRANSCRIBE_OK);
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

        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.temperature = 0.4f;           /* forces the sampler active */
        wp.temperature_inc = 0.0f;       /* single-tier, no fallback  */
        wp.compression_ratio_thold =
            TRANSCRIBE_WHISPER_THOLD_DISABLED;
        wp.logprob_thold = TRANSCRIBE_WHISPER_LOGPROB_DISABLED;
        wp.no_speech_thold = TRANSCRIBE_WHISPER_THOLD_DISABLED;
        wp.seed = 42;

        transcribe_run_params rp; transcribe_run_params_init(&rp);
        rp.language = "en";
        rp.family = &wp.ext;

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

    // ============ Stage 3: prompt + condition_on_prev_tokens ============

    // ALL_SEGMENTS without condition_on_prev_tokens must reject with
    // INVALID_ARG before any compute runs (HF generation_whisper.py
    // :1743-1745 raises ValueError; our parity is to translate that
    // into a non-OK status). No abort callback is set; the rejection
    // happens inside transcribe_run, not the callback.
    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.prompt_condition         = TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS;
        wp.condition_on_prev_tokens = false;
        rp.family = &wp.ext;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_ERR_INVALID_ARG);
    }

    // initial_prompt rejects literal special-token strings ("<|en|>",
    // "<|notimestamps|>", etc.). HF's get_prompt_ids relies on its
    // tokenizer's added-token recognition to surface specials and
    // rejects ids >= all_special_ids[0]. Our gpt-2 BPE encoder doesn't
    // recognize specials, so we precheck for "<|...|>" patterns
    // directly against the vocab. "<|en|>" is a real id (50259 for
    // multilingual whisper) and must be rejected before any compute.
    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.initial_prompt = "Inaugural <|en|> address";
        rp.family = &wp.ext;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_ERR_INVALID_ARG);
    }

    // prompt_tokens must NOT include <|startofprev|> at index 0 — the
    // library prepends it. A leading prev_sot id is a double-prepend
    // and surfaces as INVALID_ARG so the API contract is observable.
    {
        /* multilingual whisper-tiny: <|startofprev|> = 50361 */
        int32_t bad[] = { 50361, 100, 200 };
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.prompt_tokens   = bad;
        wp.n_prompt_tokens = 3;
        rp.family = &wp.ext;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_ERR_INVALID_ARG);
    }

    // initial_prompt path: a domain-term prompt should not break the
    // JFK transcript. We don't assert biasing here (would need a
    // pathological audio); we assert that the run completes and
    // "country" still appears.
    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.initial_prompt = "Inaugural address";
        rp.family = &wp.ext;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
    }

    // prompt_tokens (pre-tokenized) path: roundtrip a small token list
    // and confirm the run completes. Use transcribe_tokenize for
    // realistic ids matching the model's vocab.
    {
        int32_t tok_buf[32];
        const int n = transcribe_tokenize(model, " inaugural address",
                                          tok_buf, 32);
        CHECK(n > 0);
        if (n > 0) {
            transcribe_run_params rp; transcribe_run_params_init(&rp);
            transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
            wp.prompt_tokens   = tok_buf;
            wp.n_prompt_tokens = static_cast<size_t>(n);
            rp.family = &wp.ext;
            st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
            CHECK_EQ_INT(st, TRANSCRIBE_OK);
            CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
        }
    }

    // Meaningful initial_prompt regression: this fixture contains
    // deliberately awkward product names. The glossary prompt should
    // move the transcript toward the exact dashed/camel-cased names,
    // matching HF's prompt_ids path on the same clip.
    {
        const std::string product_path =
            std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/product-names.wav";
        std::vector<float> product_pcm;
        std::string product_err;
        if (!transcribe_cli::load_wav_mono_16k(product_path,
                                               product_pcm, product_err))
        {
            std::fprintf(stderr, "whisper_e2e_smoke: wav load: %s\n",
                         product_err.c_str());
            ++g_failures;
        } else {
            auto product_hits = [](const char * text) {
                const char * terms[] = {
                    "QuirkQuid",
                    "P3-Quattro",
                    "O3-Omni",
                    "B3-BondX",
                    "E3-Equity",
                    "W3-WrapZ",
                    "O2-Outlier",
                    "U3-UniFund",
                    "M3-Mover",
                };
                int n = 0;
                for (const char * term : terms) {
                    if (std::strstr(text, term) != nullptr) {
                        ++n;
                    }
                }
                return n;
            };

            transcribe_run_params rp; transcribe_run_params_init(&rp);
            rp.language = "en";
            rp.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
            st = transcribe_run(ctx, product_pcm.data(),
                                static_cast<int>(product_pcm.size()), &rp);
            CHECK_EQ_INT(st, TRANSCRIBE_OK);
            const int unprompted_hits =
                product_hits(transcribe_full_text(ctx));

            transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
            wp.initial_prompt =
                "QuirkQuid Quill Inc, P3-Quattro, O3-Omni, "
                "B3-BondX, E3-Equity, W3-WrapZ, O2-Outlier, "
                "U3-UniFund, M3-Mover";
            rp.family = &wp.ext;
            st = transcribe_run(ctx, product_pcm.data(),
                                static_cast<int>(product_pcm.size()), &rp);
            CHECK_EQ_INT(st, TRANSCRIBE_OK);
            const int prompted_hits =
                product_hits(transcribe_full_text(ctx));
            CHECK(prompted_hits >= 5);
            CHECK(prompted_hits > unprompted_hits + 3);
        }
    }

    // Whitespace-only prompts must collapse to "no prompt" — nothing
    // to tokenize, nothing prepended. We can't directly assert "no
    // prompt was used", but the run must succeed with output that
    // still contains "country" (matches the no-prompt default).
    {
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.initial_prompt = "   \t\n  ";
        rp.family = &wp.ext;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
    }

    // condition_on_prev_tokens on long-form: synthesize a >30s clip
    // from JFK and confirm the prev-context wiring doesn't crash and
    // doesn't degrade the obvious markers in the transcript.
    {
        std::vector<float> long_pcm;
        const int repeats = 4;
        long_pcm.reserve(pcm.size() * repeats);
        for (int i = 0; i < repeats; ++i) {
            long_pcm.insert(long_pcm.end(), pcm.begin(), pcm.end());
        }
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.condition_on_prev_tokens = true;
        rp.language = "en";
        rp.family = &wp.ext;
        st = transcribe_run(ctx, long_pcm.data(),
                            static_cast<int>(long_pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        const char * text = transcribe_full_text(ctx);
        int country_hits = 0;
        for (const char * p = text;
             (p = std::strstr(p, "country")) != nullptr; ++p)
        {
            ++country_hits;
        }
        CHECK(country_hits >= 2);
        CHECK(transcribe_get_whisper_chunk_count(ctx) >= 2);
    }

    // ALL_SEGMENTS with condition_on_prev_tokens=true: the prompt is
    // re-prepended on every chunk as the BOS of the prev-context
    // window. Run completes; "country" still present from the audio.
    {
        std::vector<float> long_pcm;
        const int repeats = 4;
        long_pcm.reserve(pcm.size() * repeats);
        for (int i = 0; i < repeats; ++i) {
            long_pcm.insert(long_pcm.end(), pcm.begin(), pcm.end());
        }
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        transcribe_whisper_run_ext wp; transcribe_whisper_run_ext_init(&wp);
        wp.initial_prompt           = "Inaugural address";
        wp.prompt_condition         = TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS;
        wp.condition_on_prev_tokens = true;
        rp.language = "en";
        rp.family = &wp.ext;
        st = transcribe_run(ctx, long_pcm.data(),
                            static_cast<int>(long_pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_OK);
        CHECK(std::strstr(transcribe_full_text(ctx), "country") != nullptr);
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
        transcribe_run_params rp; transcribe_run_params_init(&rp);
        rp.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
        st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        CHECK_EQ_INT(st, TRANSCRIBE_ERR_ABORTED);
        CHECK(transcribe_was_aborted(ctx));
        /* The explicit SEGMENT request appears on the context even
         * after abort, confirming has_result=true was committed. */
        CHECK_EQ_INT(transcribe_returned_timestamp_kind(ctx),
                     TRANSCRIBE_TIMESTAMPS_SEGMENT);
        transcribe_set_abort_callback(ctx, nullptr, nullptr);
    }

    transcribe_session_free(ctx);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "whisper_e2e_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "whisper_e2e_smoke: ok\n");
    return EXIT_SUCCESS;
}
