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

#include <math.h>
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
    CHECK(rp.timestamps         == TRANSCRIBE_TIMESTAMPS_AUTO);
    CHECK(rp.language           == NULL);
    CHECK(rp.target_language    == NULL);
    CHECK(rp.strip_special_tags == true);
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

    if (g_failures > 0) {
        fprintf(stderr, "api_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    fprintf(stdout, "api_smoke: ok\n");
    return EXIT_SUCCESS;
}
