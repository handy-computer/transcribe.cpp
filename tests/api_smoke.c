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
 *   4. Confirm TRANSCRIBE_*_INIT macros populate sensible defaults and
 *      that zero-initialized structs are rejected with BAD_STRUCT_SIZE.
 *   5. Confirm the transcribe_ext_check / transcribe_model_accepts_ext_kind
 *      surface behaves under the documented edge cases.
 *
 * This test does NOT load a model. The contracts it asserts must remain
 * stable across implementation passes.
 */

#include "transcribe.h"
#include "transcribe/extensions.h"

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
        TRANSCRIBE_ERR_BAD_STRUCT_SIZE,
        TRANSCRIBE_ERR_UNSUPPORTED_PNC,
        TRANSCRIBE_ERR_UNSUPPORTED_ITN,
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

static void test_init_macros(void) {
    /* TRANSCRIBE_*_INIT must give back the same default values the
     * legacy factories return. The factories themselves wrap the INIT
     * macros, so the comparison is round-trip on every field. */
    struct transcribe_model_load_params mp_macro = TRANSCRIBE_MODEL_LOAD_PARAMS_INIT;
    CHECK(mp_macro.struct_size == sizeof(struct transcribe_model_load_params));
    CHECK(mp_macro.backend     == TRANSCRIBE_BACKEND_AUTO);
    CHECK(mp_macro.gpu_device  == -1);

    struct transcribe_session_params cp_macro = TRANSCRIBE_SESSION_PARAMS_INIT;
    CHECK(cp_macro.struct_size == sizeof(struct transcribe_session_params));
    CHECK(cp_macro.n_threads   == 0);
    CHECK(cp_macro.kv_type     == TRANSCRIBE_KV_TYPE_AUTO);

    struct transcribe_run_params rp_macro = TRANSCRIBE_RUN_PARAMS_INIT;
    CHECK(rp_macro.struct_size        == sizeof(struct transcribe_run_params));
    CHECK(rp_macro.task               == TRANSCRIBE_TASK_TRANSCRIBE);
    CHECK(rp_macro.timestamps         == TRANSCRIBE_TIMESTAMPS_NONE);
    CHECK(rp_macro.pnc                == TRANSCRIBE_PNC_MODE_DEFAULT);
    CHECK(rp_macro.itn                == TRANSCRIBE_ITN_MODE_DEFAULT);
    CHECK(rp_macro.language           == NULL);
    CHECK(rp_macro.target_language    == NULL);
    CHECK(rp_macro.strip_special_tags == true);
    CHECK(rp_macro.family             == NULL);

    struct transcribe_stream_params sp_macro = TRANSCRIBE_STREAM_PARAMS_INIT;
    CHECK(sp_macro.struct_size == sizeof(struct transcribe_stream_params));
    CHECK(sp_macro.family      == NULL);

    /* Output structs: struct_size set, rest zero-filled. The
     * zero-means-absent contract is what makes a new caller paired with
     * an older library see consistent tail-field reads. */
    struct transcribe_stream_update upd_macro = TRANSCRIBE_STREAM_UPDATE_INIT;
    CHECK(upd_macro.struct_size        == sizeof(struct transcribe_stream_update));
    CHECK(upd_macro.result_changed     == false);
    CHECK(upd_macro.is_final           == false);
    CHECK(upd_macro.revision           == 0);
    CHECK(upd_macro.input_received_ms  == 0);
    CHECK(upd_macro.audio_committed_ms == 0);
    CHECK(upd_macro.buffered_ms        == 0);

    struct transcribe_capabilities caps_macro = TRANSCRIBE_CAPABILITIES_INIT;
    CHECK(caps_macro.struct_size                        == sizeof(struct transcribe_capabilities));
    CHECK(caps_macro.native_sample_rate                 == 0);
    CHECK(caps_macro.n_languages                        == 0);
    CHECK(caps_macro.languages                          == NULL);
    CHECK(caps_macro.supports_language_detect           == false);
    CHECK(caps_macro.supports_translate                 == false);
    CHECK(caps_macro.supports_streaming                 == false);

    struct transcribe_timings tm_macro = TRANSCRIBE_TIMINGS_INIT;
    CHECK(tm_macro.struct_size == sizeof(struct transcribe_timings));
    CHECK(tm_macro.load_ms     == 0.0f);
    CHECK(tm_macro.mel_ms      == 0.0f);
    CHECK(tm_macro.encode_ms   == 0.0f);
    CHECK(tm_macro.decode_ms   == 0.0f);

    /* Family extension INIT macros wire struct_size + kind correctly. */
    struct transcribe_parakeet_stream_ext pk = TRANSCRIBE_PARAKEET_STREAM_EXT_INIT;
    CHECK(pk.ext.size          == sizeof(struct transcribe_parakeet_stream_ext));
    CHECK(pk.ext.kind          == TRANSCRIBE_EXT_KIND_PARAKEET_STREAM);
    CHECK(pk.att_context_right == -1);

    struct transcribe_parakeet_buffered_stream_ext pkb =
        TRANSCRIBE_PARAKEET_BUFFERED_STREAM_EXT_INIT;
    CHECK(pkb.ext.size  == sizeof(struct transcribe_parakeet_buffered_stream_ext));
    CHECK(pkb.ext.kind  == TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM);
    CHECK(pkb.left_ms   == -1);
    CHECK(pkb.chunk_ms  == -1);
    CHECK(pkb.right_ms  == -1);

    struct transcribe_moonshine_streaming_stream_ext ms =
        TRANSCRIBE_MOONSHINE_STREAMING_STREAM_EXT_INIT;
    CHECK(ms.ext.size               == sizeof(struct transcribe_moonshine_streaming_stream_ext));
    CHECK(ms.ext.kind               == TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM);
    CHECK(ms.min_decode_interval_ms == -1);

    struct transcribe_whisper_chunk_trace wtr =
        TRANSCRIBE_WHISPER_CHUNK_TRACE_INIT;
    CHECK(wtr.struct_size == sizeof(struct transcribe_whisper_chunk_trace));
    CHECK(wtr.t0_ms       == 0);
    CHECK(wtr.t1_ms       == 0);
    CHECK(wtr.n_fallbacks == 0);

    /* Whisper run extension: kind + size wired by the INIT macro, and
     * the field defaults match the family's shipping recipe. */
    struct transcribe_whisper_run_ext wrx = TRANSCRIBE_WHISPER_RUN_EXT_INIT;
    CHECK(wrx.ext.size                  == sizeof(struct transcribe_whisper_run_ext));
    CHECK(wrx.ext.kind                  == TRANSCRIBE_EXT_KIND_WHISPER_RUN);
    CHECK(wrx.initial_prompt            == NULL);
    CHECK(wrx.prompt_tokens             == NULL);
    CHECK(wrx.n_prompt_tokens           == 0);
    CHECK(wrx.prompt_condition          == TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT);
    CHECK(wrx.condition_on_prev_tokens  == false);
    CHECK(wrx.max_prev_context_tokens   == 223);
    CHECK(wrx.temperature               == 0.0f);
    CHECK(wrx.temperature_inc           == 0.2f);
    CHECK(wrx.compression_ratio_thold   == 2.4f);
    CHECK(wrx.logprob_thold             == -1.0f);
    CHECK(wrx.no_speech_thold           == 0.6f);
    CHECK(wrx.seed                      == 0u);
    CHECK(wrx.max_initial_timestamp     == 1.0f);

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
     */
    transcribe_log_set(dummy_log, &ud_a);
    transcribe_log_set(dummy_log, &ud_b);
    transcribe_log_set(NULL, NULL);

    CHECK(g_dummy_log_calls == 0);
}

static void test_load_invalid(void) {
    struct transcribe_model_load_params mp = TRANSCRIBE_MODEL_LOAD_PARAMS_INIT;
    struct transcribe_model *      m  = (struct transcribe_model *)0xdeadbeef;

    /* NULL out_model -> INVALID_ARG. */
    CHECK(transcribe_model_load_file("ignored", &mp, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);

    /* NULL path -> INVALID_ARG and out_model is cleared. */
    CHECK(transcribe_model_load_file(NULL, &mp, &m)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(m == NULL);

    /* NULL params -> "all defaults", NOT an error. The call proceeds
     * past parameter validation and fails only on the missing file,
     * proving NULL was accepted as defaults. Contrast with {0} below,
     * which claims struct_size == 0 and is rejected. */
    m = (struct transcribe_model *)0xdeadbeef;
    CHECK(transcribe_model_load_file("/__transcribe_smoke_does_not_exist__.gguf",
                                     NULL, &m)
          == TRANSCRIBE_ERR_FILE_NOT_FOUND);
    CHECK(m == NULL);

    /* {0} params -> BAD_STRUCT_SIZE. Distinct from INVALID_ARG so a
     * caller that forgot the INIT macro gets a targeted diagnostic. */
    struct transcribe_model_load_params mp0 = {0};
    m = (struct transcribe_model *)0xdeadbeef;
    CHECK(transcribe_model_load_file("ignored", &mp0, &m)
          == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(m == NULL);

    /* Otherwise-valid call against a path that does not exist on disk
     * -> FILE_NOT_FOUND. */
    m = (struct transcribe_model *)0xdeadbeef;
    CHECK(transcribe_model_load_file("/__transcribe_smoke_does_not_exist__.gguf",
                                     &mp, &m)
          == TRANSCRIBE_ERR_FILE_NOT_FOUND);
    CHECK(m == NULL);

    /* Free on NULL is a no-op. */
    transcribe_model_free(NULL);
}

static void test_context_invalid(void) {
    struct transcribe_session_params cp = TRANSCRIBE_SESSION_PARAMS_INIT;
    struct transcribe_session *      c  = (struct transcribe_session *)0xdeadbeef;

    CHECK(transcribe_session_init(NULL, &cp, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_session_init(NULL, &cp, &c)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(c == NULL);

    /* {0} params would be BAD_STRUCT_SIZE, but the model==NULL check
     * triggers INVALID_ARG first. Tested via a non-NULL params with a
     * smaller-than-required struct_size below isn't reachable here
     * without a real model; the dispatcher exercises the same code
     * path the size check uses when called with a valid model. */

    transcribe_session_free(NULL);
}

static void test_run_invalid(void) {
    struct transcribe_run_params rp     = TRANSCRIBE_RUN_PARAMS_INIT;
    float                    pcm[8] = { 0.0f };

    CHECK(transcribe_run(NULL, pcm, 8, &rp)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_run((struct transcribe_session *)0x1, NULL, 8, &rp)
          == TRANSCRIBE_ERR_INVALID_ARG);
    /* NULL params is no longer an error (means "all defaults"), so it
     * can't be probed with a fake session — it would proceed to
     * dereference it. NULL-params-defaults is exercised end-to-end in the
     * real-model smoke tests instead. */
    /* n_samples must be strictly positive: both negative and zero are
     * rejected before the session is dereferenced (matches stream_feed). */
    CHECK(transcribe_run((struct transcribe_session *)0x1, pcm, -1, &rp)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_run((struct transcribe_session *)0x1, pcm, 0, &rp)
          == TRANSCRIBE_ERR_INVALID_ARG);
}

static void test_open_invalid(void) {
    struct transcribe_session * s = (struct transcribe_session *)0xdeadbeef;

    /* NULL out_session -> INVALID_ARG. */
    CHECK(transcribe_open("ignored", NULL, NULL, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);

    /* NULL path forwards to load_file -> INVALID_ARG; out_session cleared. */
    CHECK(transcribe_open(NULL, NULL, NULL, &s)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(s == NULL);

    /* NULL load/session params == defaults; missing file -> FILE_NOT_FOUND
     * with no partial session leaked. */
    s = (struct transcribe_session *)0xdeadbeef;
    CHECK(transcribe_open("/__transcribe_smoke_does_not_exist__.gguf",
                          NULL, NULL, &s)
          == TRANSCRIBE_ERR_FILE_NOT_FOUND);
    CHECK(s == NULL);

    /* close / get_model are NULL-safe. */
    transcribe_close(NULL);
    CHECK(transcribe_get_model(NULL) == NULL);
}

static void test_model_introspection_null(void) {
    /* Per the contract, accessors are safe to call on NULL. */
    struct transcribe_capabilities caps = TRANSCRIBE_CAPABILITIES_INIT;
    /* Copy-out signature: NULL model -> INVALID_ARG; out buffer
     * untouched semantically (still zero-initialized by INIT). */
    CHECK(transcribe_model_get_capabilities(NULL, &caps)
          == TRANSCRIBE_ERR_INVALID_ARG);
    /* NULL out_caps -> INVALID_ARG. */
    CHECK(transcribe_model_get_capabilities(
              (const struct transcribe_model *)0x1, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);

    CHECK_STR_EMPTY(transcribe_model_arch_string(NULL));
    CHECK_STR_EMPTY(transcribe_model_variant_string(NULL));
    CHECK_STR_EMPTY(transcribe_model_backend(NULL));

    /* The kind probe is safe on NULL and returns false. */
    CHECK(transcribe_model_accepts_ext_kind(NULL,
              TRANSCRIBE_EXT_KIND_PARAKEET_STREAM) == false);

    /* The feature probe is also NULL-safe and returns false for every
     * known feature value plus any out-of-range enum. */
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_INITIAL_PROMPT)       == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK) == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_LONG_FORM)            == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_CANCELLATION)         == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_PNC)                  == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_ITN)                  == false);
    CHECK(transcribe_model_supports(NULL, (transcribe_feature) 9999)               == false);
}

static void test_ext_check(void) {
    /* NULL ext is always OK (family decides what NULL means). */
    CHECK(transcribe_ext_check(NULL, 0x12345678u, 8) == TRANSCRIBE_OK);

    /* Wrong kind -> INVALID_ARG. */
    struct transcribe_parakeet_stream_ext pk = TRANSCRIBE_PARAKEET_STREAM_EXT_INIT;
    CHECK(transcribe_ext_check(&pk.ext,
                               TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM,
                               sizeof(struct transcribe_parakeet_buffered_stream_ext))
          == TRANSCRIBE_ERR_INVALID_ARG);

    /* Right kind + size big enough -> OK. */
    CHECK(transcribe_ext_check(&pk.ext,
                               TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
                               sizeof(struct transcribe_parakeet_stream_ext))
          == TRANSCRIBE_OK);

    /* Size too small even to cover the common header -> BAD_STRUCT_SIZE
     * before reading kind. */
    struct transcribe_parakeet_stream_ext pk_tiny = TRANSCRIBE_PARAKEET_STREAM_EXT_INIT;
    pk_tiny.ext.size = sizeof(struct transcribe_ext) - 1;
    CHECK(transcribe_ext_check(&pk_tiny.ext,
                               TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
                               sizeof(struct transcribe_parakeet_stream_ext))
          == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    /* Right kind but size too small -> BAD_STRUCT_SIZE. */
    struct transcribe_parakeet_stream_ext pk_small = TRANSCRIBE_PARAKEET_STREAM_EXT_INIT;
    pk_small.ext.size = sizeof(struct transcribe_ext); /* below the struct's prefix */
    CHECK(transcribe_ext_check(&pk_small.ext,
                               TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
                               sizeof(struct transcribe_parakeet_stream_ext))
          == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
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
    /* Whisper-specific trace accessors honor the safe-sentinel contract:
     * NULL ctx is a successful copy-out of an all-zero struct (other than
     * the caller's struct_size). */
    CHECK(transcribe_get_whisper_chunk_count(NULL) == 0);

    struct transcribe_whisper_chunk_trace tr = TRANSCRIBE_WHISPER_CHUNK_TRACE_INIT;
    CHECK(transcribe_get_whisper_chunk_trace(NULL, 0, &tr) == TRANSCRIBE_OK);
    CHECK(tr.struct_size         == sizeof(struct transcribe_whisper_chunk_trace));
    CHECK(tr.t0_ms               == 0);
    CHECK(tr.t1_ms               == 0);
    CHECK(tr.temperature_used    == 0.0f);
    CHECK(tr.compression_ratio   == 0.0f);
    CHECK(tr.avg_logprob         == 0.0f);
    CHECK(tr.no_speech_prob      == 0.0f);
    CHECK(tr.no_speech_triggered == false);
    CHECK(tr.n_fallbacks         == 0);

    /* Out-of-range index against NULL succeeds the same way. */
    struct transcribe_whisper_chunk_trace tr2 = TRANSCRIBE_WHISPER_CHUNK_TRACE_INIT;
    CHECK(transcribe_get_whisper_chunk_trace(NULL, 42, &tr2) == TRANSCRIBE_OK);
    CHECK(tr2.t0_ms == 0);

    /* NULL out_trace is rejected. */
    CHECK(transcribe_get_whisper_chunk_trace(NULL, 0, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);

    /* Zero struct_size is rejected. */
    struct transcribe_whisper_chunk_trace tr_bad = {0};
    CHECK(transcribe_get_whisper_chunk_trace(NULL, 0, &tr_bad)
          == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

static void test_timings_null(void) {
    /* NULL ctx -> INVALID_ARG. NULL out -> INVALID_ARG. {0} out ->
     * BAD_STRUCT_SIZE. */
    struct transcribe_timings tm = TRANSCRIBE_TIMINGS_INIT;
    CHECK(transcribe_get_timings(NULL, &tm) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_get_timings((const struct transcribe_session *)0x1, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);
    struct transcribe_timings tm0 = {0};
    CHECK(transcribe_get_timings((const struct transcribe_session *)0x1, &tm0)
          == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

static void test_result_accessors_null(void) {
    const struct transcribe_session * ctx = NULL;

    /* Top level. */
    CHECK_STR_EMPTY(transcribe_full_text(ctx));
    CHECK(transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_NONE);
    CHECK(transcribe_n_segments(ctx) == 0);
    CHECK(transcribe_n_words(ctx)    == 0);
    CHECK(transcribe_n_tokens(ctx)   == 0);

    /* Segment row: NULL ctx → OK, struct stays zero-init (text=NULL). */
    struct transcribe_segment seg = TRANSCRIBE_SEGMENT_INIT;
    CHECK(transcribe_get_segment(ctx, 0, &seg) == TRANSCRIBE_OK);
    CHECK(seg.struct_size == sizeof(struct transcribe_segment));
    CHECK(seg.text        == NULL);
    CHECK(seg.t0_ms       == 0);
    CHECK(seg.t1_ms       == 0);
    CHECK(seg.first_word  == 0);
    CHECK(seg.n_words     == 0);
    CHECK(seg.first_token == 0);
    CHECK(seg.n_tokens    == 0);

    /* Word row. */
    struct transcribe_word wrd = TRANSCRIBE_WORD_INIT;
    CHECK(transcribe_get_word(ctx, 0, &wrd) == TRANSCRIBE_OK);
    CHECK(wrd.struct_size == sizeof(struct transcribe_word));
    CHECK(wrd.text        == NULL);
    CHECK(wrd.t0_ms       == 0);
    CHECK(wrd.t1_ms       == 0);
    CHECK(wrd.seg_index   == 0);
    CHECK(wrd.first_token == 0);
    CHECK(wrd.n_tokens    == 0);

    /* Token row. p is 0.0f (not NaN) on the zero-init path; bindings
     * distinguish "row not present" via text==NULL, not via NaN-on-p. */
    struct transcribe_token tok = TRANSCRIBE_TOKEN_INIT;
    CHECK(transcribe_get_token(ctx, 0, &tok) == TRANSCRIBE_OK);
    CHECK(tok.struct_size == sizeof(struct transcribe_token));
    CHECK(tok.text        == NULL);
    CHECK(tok.id          == 0);
    CHECK(tok.p           == 0.0f);
    CHECK(tok.t0_ms       == 0);
    CHECK(tok.t1_ms       == 0);
    CHECK(tok.seg_index   == 0);
    CHECK(tok.word_index  == 0);

    /* NULL out_ptr -> INVALID_ARG. */
    CHECK(transcribe_get_segment(ctx, 0, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_get_word   (ctx, 0, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_get_token  (ctx, 0, NULL) == TRANSCRIBE_ERR_INVALID_ARG);

    /* Uninitialized struct_size (== 0) -> BAD_STRUCT_SIZE. */
    struct transcribe_segment seg_bad = {0};
    CHECK(transcribe_get_segment(ctx, 0, &seg_bad)
          == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

static void test_stream_state_values(void) {
    /* The state enum values are part of the public ABI — pin them. */
    CHECK(TRANSCRIBE_STREAM_IDLE     == 0);
    CHECK(TRANSCRIBE_STREAM_ACTIVE   == 1);
    CHECK(TRANSCRIBE_STREAM_FINISHED == 2);
    CHECK(TRANSCRIBE_STREAM_FAILED   == 3);
}

static void test_stream_accessors_null(void) {
    const struct transcribe_session * ctx = NULL;
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
    struct transcribe_run_params        rp = TRANSCRIBE_RUN_PARAMS_INIT;
    struct transcribe_stream_params sp = TRANSCRIBE_STREAM_PARAMS_INIT;
    float                           pcm[1] = { 0.0f };

    CHECK(transcribe_stream_begin(NULL, &rp, &sp)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_feed(NULL, pcm, 1, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_finalize(NULL, NULL)
          == TRANSCRIBE_ERR_INVALID_ARG);
    /* reset is void; calling it on NULL must not crash. */
    transcribe_stream_reset(NULL);

    /* {0} update -> BAD_STRUCT_SIZE on feed/finalize. The struct_size
     * check fires before ctx is dereferenced, so passing a non-NULL
     * fake ctx pointer is fine here. */
    struct transcribe_stream_update upd0 = {0};
    CHECK(transcribe_stream_feed((struct transcribe_session *)0x1,
                                 pcm, 1, &upd0)
          == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(transcribe_stream_finalize((struct transcribe_session *)0x1, &upd0)
          == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

int main(void) {
    test_status_string();
    test_log_level_values();
    test_log_set_null();
    test_init_macros();
    test_log_set_publication();
    test_load_invalid();
    test_context_invalid();
    test_run_invalid();
    test_open_invalid();
    test_model_introspection_null();
    test_ext_check();
    test_result_accessors_null();
    test_tokenize_null();
    test_abort_callback_null();
    test_whisper_chunk_trace_null();
    test_timings_null();
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
