/*
 * api_smoke.c - pure C smoke test for the public transcribe ABI.
 *
 * Goals:
 *   1. Prove the public header compiles cleanly as C11 (catches any
 *      C++ leakage past extern "C" — references, default args, bool
 *      shenanigans, etc.).
 *   2. Exercise every accessor with a NULL context and confirm each one
 *      returns the safe sentinel documented in include/transcribe.h.
 *   3. Confirm the two real entry points return TRANSCRIBE_ERR_INVALID_ARG
 *      on bad input and TRANSCRIBE_ERR_NOT_IMPLEMENTED on otherwise-valid
 *      stub input.
 *   4. Confirm the factory functions populate sensible defaults.
 *
 * This test does NOT load a model. It runs in pass 1 against the stub
 * implementation and will continue to run after the real implementation
 * lands - the contracts it asserts must remain stable.
 */

#include "transcribe.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                   \
        }                                                                   \
    } while (0)

#define CHECK_STR_EMPTY(s)                                                  \
    do {                                                                    \
        const char * _s = (s);                                              \
        if (_s == NULL || _s[0] != '\0') {                                  \
            fprintf(stderr,                                                 \
                    "FAIL %s:%d: expected empty string, got %s\n",          \
                    __FILE__, __LINE__,                                     \
                    _s == NULL ? "(null)" : _s);                            \
            g_failures++;                                                   \
        }                                                                   \
    } while (0)

static void test_status_string(void) {
    /* Every defined enumerator returns a non-empty static string.
     * transcribe_status_string takes int, so the enumerators convert
     * implicitly at the call site. */
    const int all[] = {
        TRANSCRIBE_OK,
        TRANSCRIBE_ERR_INVALID_ARG,
        TRANSCRIBE_ERR_NOT_IMPLEMENTED,
        TRANSCRIBE_ERR_FILE_NOT_FOUND,
        TRANSCRIBE_ERR_GGUF,
        TRANSCRIBE_ERR_UNSUPPORTED_ARCH,
        TRANSCRIBE_ERR_UNSUPPORTED_VARIANT,
        TRANSCRIBE_ERR_OOM,
        TRANSCRIBE_ERR_BACKEND,
        TRANSCRIBE_ERR_SAMPLE_RATE,
        TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE,
        TRANSCRIBE_ERR_UNSUPPORTED_TASK,
        TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS,
        TRANSCRIBE_ERR_ABORTED,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i) {
        const char * s = transcribe_status_string(all[i]);
        CHECK(s != NULL);
        CHECK(s[0] != '\0');
    }

    /* Out-of-range still returns a non-NULL string. Pass via int so we
     * never form an out-of-range enum object (which would be UB in C++
     * and would trip UBSan in the implementation). */
    const char * unk = transcribe_status_string(9999);
    CHECK(unk != NULL);
    CHECK(unk[0] != '\0');

    /* Negative is also out-of-range and should still return a string. */
    const char * neg = transcribe_status_string(-1);
    CHECK(neg != NULL);
    CHECK(neg[0] != '\0');
}

static void test_log_level_values(void) {
    /* These numeric values must mirror GGML_LOG_LEVEL_* exactly. If this
     * test ever fails, the public contract documented in transcribe.h
     * has been broken. */
    CHECK(TRANSCRIBE_LOG_LEVEL_NONE  == 0);
    CHECK(TRANSCRIBE_LOG_LEVEL_INFO  == 1);
    CHECK(TRANSCRIBE_LOG_LEVEL_WARN  == 2);
    CHECK(TRANSCRIBE_LOG_LEVEL_ERROR == 3);
    CHECK(TRANSCRIBE_LOG_LEVEL_DEBUG == 4);
    CHECK(TRANSCRIBE_LOG_LEVEL_CONT  == 5);
}

static void test_log_set_null(void) {
    /* Disabling the log sink must not crash. */
    transcribe_log_set(NULL, NULL);
}

static void test_factories(void) {
    struct transcribe_model_params mp = transcribe_model_default_params();
    CHECK(mp.backend    == TRANSCRIBE_BACKEND_AUTO);
    CHECK(mp.gpu_device == -1);

    struct transcribe_context_params cp = transcribe_context_default_params();
    CHECK(cp.n_threads == 0);

    struct transcribe_params rp = transcribe_default_params();
    CHECK(rp.task               == TRANSCRIBE_TASK_TRANSCRIBE);
    CHECK(rp.timestamps         == TRANSCRIBE_TIMESTAMPS_NONE);
    CHECK(rp.language           == NULL);
    CHECK(rp.target_language    == NULL);
    CHECK(rp.strip_special_tags == true);
    CHECK(rp.whisper            == NULL);
    CHECK(rp.sensevoice         == NULL);
    CHECK(rp.funasr_nano        == NULL);

    struct transcribe_sensevoice_params svp = transcribe_sensevoice_default_params();
    CHECK(svp.use_itn == false);

    struct transcribe_funasr_nano_params fnp = transcribe_funasr_nano_default_params();
    CHECK(fnp.use_itn == false);

    struct transcribe_whisper_params wp = transcribe_whisper_default_params();
    CHECK(wp.initial_prompt           == NULL);
    CHECK(wp.prompt_tokens            == NULL);
    CHECK(wp.n_prompt_tokens          == 0);
    CHECK(wp.prompt_condition         == TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT);
    CHECK(wp.condition_on_prev_tokens == false);
    CHECK(wp.max_prev_context_tokens  == 223);
    CHECK(wp.temperature              == 0.0f);
    CHECK(wp.temperature_inc          == 0.2f);
    CHECK(wp.compression_ratio_thold  == 2.4f);
    CHECK(wp.logprob_thold            == -1.0f);
    CHECK(wp.no_speech_thold          == 0.6f);
    CHECK(wp.seed                     == 0);
    CHECK(wp.max_initial_timestamp    == 1.0f);

    /* The disabled-sentinel defines must be usable as compile-time
     * constants with the right signs. */
    CHECK(TRANSCRIBE_WHISPER_THOLD_DISABLED   > 0.0f);
    CHECK(TRANSCRIBE_WHISPER_LOGPROB_DISABLED < 0.0f);
    CHECK(isinf(TRANSCRIBE_WHISPER_THOLD_DISABLED));
    CHECK(isinf(TRANSCRIBE_WHISPER_LOGPROB_DISABLED));
}

/*
 * dummy_log: a callback we install just to exercise the publication path
 * end-to-end. It increments a counter so we can confirm the smoke test
 * itself is sane (the counter must remain 0 because nothing in pass 1
 * actually emits log entries yet).
 */
static int g_dummy_log_calls = 0;
static void dummy_log(transcribe_log_level level, const char * msg, void * userdata) {
    (void)level;
    (void)msg;
    (void)userdata;
    g_dummy_log_calls++;
}

static void test_log_set_publication(void) {
    int ud_a = 1;
    int ud_b = 2;

    /*
     * Single-threaded smoke of the install path. Install, replace, then
     * uninstall. None of these should crash. The dummy callback must not
     * be invoked because pass 1 has no emission path wired up to the
     * public sink.
     *
     * This test does NOT validate concurrent reconfiguration, mid-run
     * reconfiguration with live emitters, or old-sink lifetime safety.
     * Per the threading contract in transcribe.h, those scenarios are
     * unsupported in 0.x; the supported model is "install once at
     * startup before threads or models exist." This test exercises that
     * supported path and nothing more.
     */
    transcribe_log_set(dummy_log, &ud_a);
    transcribe_log_set(dummy_log, &ud_b);
    transcribe_log_set(NULL, NULL);

    CHECK(g_dummy_log_calls == 0);
}

static void test_load_invalid(void) {
    struct transcribe_model_params mp = transcribe_model_default_params();
    struct transcribe_model *      m  = (struct transcribe_model *)0xdeadbeef;

    /* NULL out_model -> INVALID_ARG. */
    CHECK(transcribe_model_load_file("ignored", &mp, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);

    /* NULL path -> INVALID_ARG and out_model is cleared. */
    CHECK(transcribe_model_load_file(NULL, &mp, &m)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(m == NULL);

    /* NULL params -> INVALID_ARG and out_model is cleared. */
    m = (struct transcribe_model *)0xdeadbeef;
    CHECK(transcribe_model_load_file("ignored", NULL, &m)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(m == NULL);

    /* Otherwise-valid call against a path that does not exist on disk
     * -> FILE_NOT_FOUND. The pre-existence-check inside the loader is
     * what makes this distinguishable from a generic ERR_GGUF, so this
     * assertion locks the loader's stat() pre-check in place. The
     * fixture-driven loader_smoke test exercises the GGUF / arch /
     * variant paths against real on-disk fixtures. */
    m = (struct transcribe_model *)0xdeadbeef;
    CHECK(transcribe_model_load_file("/__transcribe_smoke_does_not_exist__.gguf",
                                     &mp, &m)
          == TRANSCRIBE_ERR_FILE_NOT_FOUND);
    CHECK(m == NULL);

    /* Free on NULL is a no-op. */
    transcribe_model_free(NULL);
}

static void test_context_invalid(void) {
    struct transcribe_context_params cp = transcribe_context_default_params();
    struct transcribe_context *      c  = (struct transcribe_context *)0xdeadbeef;

    CHECK(transcribe_context_init(NULL, &cp, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_context_init(NULL, &cp, &c)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(c == NULL);

    transcribe_context_free(NULL);
}

static void test_run_invalid(void) {
    struct transcribe_params rp     = transcribe_default_params();
    float                    pcm[8] = { 0.0f };

    CHECK(transcribe_run(NULL, pcm, 8, &rp)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_run((struct transcribe_context *)0x1, NULL, 8, &rp)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_run((struct transcribe_context *)0x1, pcm, 8, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_run((struct transcribe_context *)0x1, pcm, -1, &rp)
          == TRANSCRIBE_ERR_INVALID_ARG);
}

static void test_model_introspection_null(void) {
    /* Per the contract, accessors are safe to call on NULL. They return
     * empty strings / NULL capability struct. */
    CHECK(transcribe_model_capabilities(NULL) == NULL);
    CHECK_STR_EMPTY(transcribe_model_arch_string(NULL));
    CHECK_STR_EMPTY(transcribe_model_variant_string(NULL));
    CHECK_STR_EMPTY(transcribe_model_backend(NULL));
}

static void test_tokenize_null(void) {
    int32_t tokens[8];
    /* NULL model -> INT_MIN (hard error). */
    CHECK(transcribe_tokenize(NULL, "hello", tokens, 8) == INT_MIN);
    /* NULL text -> INT_MIN. */
    CHECK(transcribe_tokenize((const struct transcribe_model *)0x1,
                              NULL, tokens, 8) == INT_MIN);
}

static int g_abort_calls = 0;
static bool abort_cb_false(void * u) {
    (void)u;
    g_abort_calls++;
    return false;
}

static void test_abort_callback_null(void) {
    /* NULL ctx: set is a no-op, was_aborted returns false. */
    transcribe_set_abort_callback(NULL, abort_cb_false, NULL);
    CHECK(transcribe_was_aborted(NULL) == false);
    CHECK(g_abort_calls == 0);
}

static void test_whisper_chunk_trace_null(void) {
    /* Whisper-specific trace accessors must also honor the safe-sentinel
     * contract: NULL ctx returns 0 / zeroed-struct, not a crash. A
     * non-Whisper context (anything whose arch-name does not match
     * "whisper") is likewise treated as a zero/empty read — the
     * accessor is defined over every context type for C-API
     * uniformity but produces no state for families that don't have a
     * per-chunk fallback loop to trace. NULL exercises both legs here;
     * the arch-mismatch leg is covered by the family-specific smoke
     * tests for Parakeet / Qwen3-ASR / Cohere which never call the
     * Whisper trace accessors in their happy path. */
    CHECK(transcribe_get_whisper_chunk_count(NULL) == 0);

    struct transcribe_whisper_chunk_trace tr =
        transcribe_get_whisper_chunk_trace(NULL, 0);
    CHECK(tr.t0_ms               == 0);
    CHECK(tr.t1_ms               == 0);
    CHECK(tr.temperature_used    == 0.0f);
    CHECK(tr.compression_ratio   == 0.0f);
    CHECK(tr.avg_logprob         == 0.0f);
    CHECK(tr.no_speech_prob      == 0.0f);
    CHECK(tr.no_speech_triggered == false);
    CHECK(tr.n_fallbacks         == 0);

    /* Out-of-range index against NULL must also return the zeroed
     * struct, not crash. */
    struct transcribe_whisper_chunk_trace tr2 =
        transcribe_get_whisper_chunk_trace(NULL, 42);
    CHECK(tr2.t0_ms == 0);
    CHECK(tr2.t1_ms == 0);
}

static void test_result_accessors_null(void) {
    const struct transcribe_context * ctx = NULL;

    /* Top level. */
    CHECK_STR_EMPTY(transcribe_full_text(ctx));
    CHECK(transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_NONE);
    CHECK(transcribe_n_segments(ctx) == 0);
    CHECK(transcribe_n_words(ctx)    == 0);
    CHECK(transcribe_n_tokens(ctx)   == 0);

    /* Segment. */
    CHECK_STR_EMPTY(transcribe_segment_text(ctx, 0));
    CHECK(transcribe_segment_t0_ms(ctx, 0)       == 0);
    CHECK(transcribe_segment_t1_ms(ctx, 0)       == 0);
    CHECK(transcribe_segment_first_word(ctx, 0)  == 0);
    CHECK(transcribe_segment_n_words(ctx, 0)     == 0);
    CHECK(transcribe_segment_first_token(ctx, 0) == 0);
    CHECK(transcribe_segment_n_tokens(ctx, 0)    == 0);

    /* Word. */
    CHECK_STR_EMPTY(transcribe_word_text(ctx, 0));
    CHECK(transcribe_word_t0_ms(ctx, 0)       == 0);
    CHECK(transcribe_word_t1_ms(ctx, 0)       == 0);
    CHECK(transcribe_word_seg_index(ctx, 0)   == 0);
    CHECK(transcribe_word_first_token(ctx, 0) == 0);
    CHECK(transcribe_word_n_tokens(ctx, 0)    == 0);

    /* Token. */
    CHECK(transcribe_token_id(ctx, 0) == 0);
    CHECK_STR_EMPTY(transcribe_token_text(ctx, 0));
    /* token_p must be NaN by contract when no probability is available. */
    CHECK(isnan(transcribe_token_p(ctx, 0)));
    CHECK(transcribe_token_t0_ms(ctx, 0)      == 0);
    CHECK(transcribe_token_t1_ms(ctx, 0)      == 0);
    CHECK(transcribe_token_seg_index(ctx, 0)  == 0);
    CHECK(transcribe_token_word_index(ctx, 0) == 0);
}

static void test_stream_factory(void) {
    /* The streaming factory must return zeroed extension pointers
     * regardless of which families are linked in. */
    struct transcribe_stream_params sp = transcribe_stream_default_params();
    CHECK(sp.parakeet == NULL);
}

static void test_stream_state_values(void) {
    /* The state enum values are part of the public ABI — pin them. */
    CHECK(TRANSCRIBE_STREAM_IDLE     == 0);
    CHECK(TRANSCRIBE_STREAM_ACTIVE   == 1);
    CHECK(TRANSCRIBE_STREAM_FINISHED == 2);
    CHECK(TRANSCRIBE_STREAM_FAILED   == 3);
}

static void test_stream_accessors_null(void) {
    const struct transcribe_context * ctx = NULL;
    /* Every streaming accessor is safe to call on NULL and returns the
     * documented sentinel. */
    CHECK(transcribe_stream_get_state(ctx)             == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_stream_revision(ctx)              == 0);
    CHECK(transcribe_stream_n_committed_segments(ctx)  == 0);
    CHECK(transcribe_stream_n_committed_words(ctx)     == 0);
    CHECK(transcribe_stream_n_committed_tokens(ctx)    == 0);
    CHECK(transcribe_stream_last_status(ctx)           == TRANSCRIBE_OK);
}

static void test_stream_entries_null(void) {
    /* NULL ctx into every entry point: begin/feed/finalize report
     * INVALID_ARG; reset is a no-op. */
    struct transcribe_params        rp = transcribe_default_params();
    struct transcribe_stream_params sp = transcribe_stream_default_params();
    float                           pcm[1] = { 0.0f };

    CHECK(transcribe_stream_begin(NULL, &rp, &sp)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_feed(NULL, pcm, 1, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_finalize(NULL, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);
    /* reset is void; calling it on NULL must not crash. */
    transcribe_stream_reset(NULL);
}

int main(void) {
    test_status_string();
    test_log_level_values();
    test_log_set_null();
    test_factories();
    test_log_set_publication();
    test_load_invalid();
    test_context_invalid();
    test_run_invalid();
    test_model_introspection_null();
    test_result_accessors_null();
    test_tokenize_null();
    test_abort_callback_null();
    test_whisper_chunk_trace_null();
    test_stream_factory();
    test_stream_state_values();
    test_stream_accessors_null();
    test_stream_entries_null();

    if (g_failures > 0) {
        fprintf(stderr, "api_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    fprintf(stdout, "api_smoke: ok\n");
    return EXIT_SUCCESS;
}
