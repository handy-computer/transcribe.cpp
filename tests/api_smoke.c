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
 *   4. Confirm the params init functions populate sensible defaults
 *      and that every caller-owned struct (input AND output) rejects
 *      struct_size == 0 with BAD_STRUCT_SIZE — defaults are reached
 *      via a NULL pointer, never via a zero-sized struct.
 *   5. Confirm the transcribe_ext_check / transcribe_model_accepts_ext_kind
 *      surface behaves under the documented edge cases, including the
 *      RUN/STREAM slot split.
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

#define CHECK_STR_EMPTY(s)                                                                     \
    do {                                                                                       \
        const char * _s = (s);                                                                 \
        if (_s == NULL || _s[0] != '\0') {                                                     \
            fprintf(stderr, "FAIL %s:%d: expected empty string, got %s\n", __FILE__, __LINE__, \
                    _s == NULL ? "(null)" : _s);                                               \
            g_failures++;                                                                      \
        }                                                                                      \
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
        TRANSCRIBE_ERR_INPUT_TOO_LONG,
        TRANSCRIBE_ERR_OUTPUT_TRUNCATED,
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

static void test_version(void) {
    /* transcribe_version() returns a non-empty static string that equals the
     * stringized MAJOR.MINOR.PATCH macros the caller compiled against. This is
     * the exact-match contract the Python provider version gate relies on. */
    const char * v = transcribe_version();
    CHECK(v != NULL);
    CHECK(v[0] != '\0');
    CHECK(strcmp(v, TRANSCRIBE_VERSION) == 0);

    /* The numeric form stays consistent with the components. */
    CHECK(TRANSCRIBE_VERSION_NUMBER ==
          TRANSCRIBE_VERSION_MAJOR * 10000 + TRANSCRIBE_VERSION_MINOR * 100 + TRANSCRIBE_VERSION_PATCH);

    /* Commit is never NULL: "unknown" in a non-git build, a short SHA
     * otherwise. */
    const char * commit = transcribe_version_commit();
    CHECK(commit != NULL);
    CHECK(commit[0] != '\0');
}

static void test_abi_metadata(void) {
    /* The native ABI accessors a binding uses to verify its struct layout.
     * Known ids report this build's sizeof/alignof; an unknown id reports 0
     * (the documented "cannot verify" sentinel, never a real size). */
    CHECK(transcribe_abi_struct_size(TRANSCRIBE_ABI_RUN_PARAMS) == sizeof(struct transcribe_run_params));
    CHECK(transcribe_abi_struct_align(TRANSCRIBE_ABI_RUN_PARAMS) == _Alignof(struct transcribe_run_params));
    CHECK(transcribe_abi_struct_size(TRANSCRIBE_ABI_CAPABILITIES) == sizeof(struct transcribe_capabilities));
    CHECK(transcribe_abi_struct_size(TRANSCRIBE_ABI_SEGMENT) == sizeof(struct transcribe_segment));
    CHECK(transcribe_abi_struct_size((transcribe_abi_struct) 9999) == 0);
    CHECK(transcribe_abi_struct_align((transcribe_abi_struct) 9999) == 0);

    /* The reported size must equal the struct_size the init function stamps:
     * that is the exact value a binding compares its own sizeof against, so the
     * two surfaces (ABI accessor and init stamping) must agree. */
    struct transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    CHECK(transcribe_abi_struct_size(TRANSCRIBE_ABI_RUN_PARAMS) == rp.struct_size);

    struct transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    CHECK(transcribe_abi_struct_size(TRANSCRIBE_ABI_CAPABILITIES) == caps.struct_size);
}

static void test_backend_devices(void) {
    /* A build has at least one compute device once backends are available:
     * compiled-in builds register at startup; dynamic-backend builds
     * register via transcribe_init_backends. This smoke runs in both
     * configurations. */
    const int n_before = transcribe_backend_device_count();
    CHECK(n_before >= 0);

    /* init_backends argument contract. */
    CHECK(transcribe_init_backends(NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_init_backends("") == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_init_backends("/nonexistent-transcribe-artifact-dir") == TRANSCRIBE_ERR_FILE_NOT_FOUND);

    /* An existing directory with no modules: OK when backends are already
     * registered (static build), ERR_BACKEND when a dynamic build ends up
     * with zero devices. Idempotent: a repeat call reports the same and
     * never re-registers (device count stable). */
    const int st1 = transcribe_init_backends(".");
    CHECK(st1 == TRANSCRIBE_OK || st1 == TRANSCRIBE_ERR_BACKEND);
    const int n_after = transcribe_backend_device_count();
    const int st2     = transcribe_init_backends(".");
    CHECK(st2 == st1);
    CHECK(transcribe_backend_device_count() == n_after);

    if (n_after > 0) {
        struct transcribe_backend_device dev;
        transcribe_backend_device_init(&dev);
        CHECK(dev.struct_size == sizeof(dev));
        CHECK(transcribe_get_backend_device(0, &dev) == TRANSCRIBE_OK);
        CHECK(dev.name != NULL && dev.name[0] != '\0');
        CHECK(dev.description != NULL);
        CHECK(dev.kind != NULL && dev.kind[0] != '\0');

        CHECK(transcribe_backend_available(TRANSCRIBE_BACKEND_AUTO));
        CHECK(transcribe_backend_available(TRANSCRIBE_BACKEND_CPU));
    }

    /* Out-of-range probes answer cleanly: error status or false, never UB. */
    struct transcribe_backend_device dev2;
    transcribe_backend_device_init(&dev2);
    CHECK(transcribe_get_backend_device(-1, &dev2) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_get_backend_device(1 << 20, &dev2) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_get_backend_device(0, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(!transcribe_backend_available((transcribe_backend_request) 999));

    /* ABI accessors must know the new struct. */
    CHECK(transcribe_abi_struct_size(TRANSCRIBE_ABI_BACKEND_DEVICE) == sizeof(struct transcribe_backend_device));
    CHECK(transcribe_abi_struct_align(TRANSCRIBE_ABI_BACKEND_DEVICE) == _Alignof(struct transcribe_backend_device));
}

static void test_log_level_values(void) {
    /* These numeric values must mirror GGML_LOG_LEVEL_* exactly. If this
     * test ever fails, the public contract documented in transcribe.h
     * has been broken. */
    CHECK(TRANSCRIBE_LOG_LEVEL_NONE == 0);
    CHECK(TRANSCRIBE_LOG_LEVEL_INFO == 1);
    CHECK(TRANSCRIBE_LOG_LEVEL_WARN == 2);
    CHECK(TRANSCRIBE_LOG_LEVEL_ERROR == 3);
    CHECK(TRANSCRIBE_LOG_LEVEL_DEBUG == 4);
    CHECK(TRANSCRIBE_LOG_LEVEL_CONT == 5);
}

static void test_log_set_null(void) {
    /* Disabling the log sink must not crash. */
    transcribe_log_set(NULL, NULL);
}

static void test_init_macros(void) {
    /* Input params: the transcribe_*_params_init() functions fill every
     * field with its default. Every default is the zero value, which is
     * what makes `{0}` a valid defaults form too (tested separately). */
    struct transcribe_model_load_params mp_macro;
    transcribe_model_load_params_init(&mp_macro);
    CHECK(mp_macro.struct_size == sizeof(struct transcribe_model_load_params));
    CHECK(mp_macro.backend == TRANSCRIBE_BACKEND_AUTO);
    CHECK(mp_macro.gpu_device == 0);

    struct transcribe_session_params cp_macro;
    transcribe_session_params_init(&cp_macro);
    CHECK(cp_macro.struct_size == sizeof(struct transcribe_session_params));
    CHECK(cp_macro.n_threads == 0);
    CHECK(cp_macro.kv_type == TRANSCRIBE_KV_TYPE_AUTO);
    CHECK(cp_macro.n_ctx == 0); /* 0 = model max (session context cap) */

    struct transcribe_run_params rp_macro;
    transcribe_run_params_init(&rp_macro);
    CHECK(rp_macro.struct_size == sizeof(struct transcribe_run_params));
    CHECK(rp_macro.task == TRANSCRIBE_TASK_TRANSCRIBE);
    CHECK(rp_macro.timestamps == TRANSCRIBE_TIMESTAMPS_AUTO);
    CHECK(rp_macro.pnc == TRANSCRIBE_PNC_MODE_DEFAULT);
    CHECK(rp_macro.itn == TRANSCRIBE_ITN_MODE_DEFAULT);
    CHECK(rp_macro.language == NULL);
    CHECK(rp_macro.target_language == NULL);
    CHECK(rp_macro.keep_special_tags == false);
    CHECK(rp_macro.family == NULL);

    struct transcribe_stream_params sp_macro;
    transcribe_stream_params_init(&sp_macro);
    CHECK(sp_macro.struct_size == sizeof(struct transcribe_stream_params));
    CHECK(sp_macro.family == NULL);
    CHECK(sp_macro.commit_policy == TRANSCRIBE_STREAM_COMMIT_AUTO);
    CHECK(sp_macro.stable_prefix_agreement_n == 0);

    /* {0} is NOT accepted as a defaults shortcut in pre-1.0: a struct
     * with struct_size == 0 is rejected with BAD_STRUCT_SIZE regardless
     * of any other field's value. Defaults are reached by passing NULL
     * where the entry point accepts a nullable params pointer. The
     * field values below confirm only that aggregate zero-init still
     * gives the documented zero-value field defaults — useful as a
     * sanity check on the struct layout, not as a calling convention. */
    struct transcribe_run_params rp_zero = { 0 };
    CHECK(rp_zero.task == TRANSCRIBE_TASK_TRANSCRIBE);
    CHECK(rp_zero.keep_special_tags == false);
    CHECK(rp_zero.struct_size == 0);

    /* Output structs: struct_size set, rest zero-filled. The
     * zero-means-absent contract is what makes a new caller paired with
     * an older library see consistent tail-field reads. */
    struct transcribe_stream_update upd_macro;
    transcribe_stream_update_init(&upd_macro);
    CHECK(upd_macro.struct_size == sizeof(struct transcribe_stream_update));
    CHECK(upd_macro.result_changed == false);
    CHECK(upd_macro.is_final == false);
    CHECK(upd_macro.revision == 0);
    CHECK(upd_macro.input_received_ms == 0);
    CHECK(upd_macro.audio_committed_ms == 0);
    CHECK(upd_macro.buffered_ms == 0);
    CHECK(upd_macro.committed_changed == false);
    CHECK(upd_macro.tentative_changed == false);

    struct transcribe_stream_text text_macro;
    transcribe_stream_text_init(&text_macro);
    CHECK(text_macro.struct_size == sizeof(struct transcribe_stream_text));
    CHECK(text_macro.full_text == NULL);
    CHECK(text_macro.full_text_bytes == 0);
    CHECK(text_macro.committed_text == NULL);
    CHECK(text_macro.committed_text_bytes == 0);
    CHECK(text_macro.tentative_text == NULL);
    CHECK(text_macro.tentative_text_bytes == 0);
    CHECK(text_macro.raw_tentative_start_bytes == 0);

    struct transcribe_capabilities caps_macro;
    transcribe_capabilities_init(&caps_macro);
    CHECK(caps_macro.struct_size == sizeof(struct transcribe_capabilities));
    CHECK(caps_macro.native_sample_rate == 0);
    CHECK(caps_macro.n_languages == 0);
    CHECK(caps_macro.languages == NULL);
    CHECK(caps_macro.supports_language_detect == false);
    CHECK(caps_macro.supports_translate == false);
    CHECK(caps_macro.supports_streaming == false);

    struct transcribe_timings tm_macro;
    transcribe_timings_init(&tm_macro);
    CHECK(tm_macro.struct_size == sizeof(struct transcribe_timings));
    CHECK(tm_macro.load_ms == 0.0f);
    CHECK(tm_macro.mel_ms == 0.0f);
    CHECK(tm_macro.encode_ms == 0.0f);
    CHECK(tm_macro.decode_ms == 0.0f);

    /* Family extension init functions wire struct_size + kind correctly. */
    struct transcribe_parakeet_stream_ext pk;
    transcribe_parakeet_stream_ext_init(&pk);
    CHECK(pk.ext.size == sizeof(struct transcribe_parakeet_stream_ext));
    CHECK(pk.ext.kind == TRANSCRIBE_EXT_KIND_PARAKEET_STREAM);
    CHECK(pk.att_context_right == -1);

    struct transcribe_parakeet_buffered_stream_ext pkb;
    transcribe_parakeet_buffered_stream_ext_init(&pkb);
    CHECK(pkb.ext.size == sizeof(struct transcribe_parakeet_buffered_stream_ext));
    CHECK(pkb.ext.kind == TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM);
    CHECK(pkb.left_ms == -1);
    CHECK(pkb.chunk_ms == -1);
    CHECK(pkb.right_ms == -1);

    struct transcribe_moonshine_streaming_stream_ext ms;
    transcribe_moonshine_streaming_stream_ext_init(&ms);
    CHECK(ms.ext.size == sizeof(struct transcribe_moonshine_streaming_stream_ext));
    CHECK(ms.ext.kind == TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM);
    CHECK(ms.min_decode_interval_ms == -1);

    struct transcribe_voxtral_realtime_stream_ext vr;
    transcribe_voxtral_realtime_stream_ext_init(&vr);
    CHECK(vr.ext.size == sizeof(struct transcribe_voxtral_realtime_stream_ext));
    CHECK(vr.ext.kind == TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM);
    CHECK(vr.num_delay_tokens == -1);
    CHECK(vr.min_decode_interval_ms == -1);

    struct transcribe_whisper_chunk_trace wtr;
    transcribe_whisper_chunk_trace_init(&wtr);
    CHECK(wtr.struct_size == sizeof(struct transcribe_whisper_chunk_trace));
    CHECK(wtr.t0_ms == 0);
    CHECK(wtr.t1_ms == 0);
    CHECK(wtr.n_fallbacks == 0);

    /* Whisper run extension: kind + size wired by the init function,
     * and the field defaults match the family's shipping recipe. */
    struct transcribe_whisper_run_ext wrx;
    transcribe_whisper_run_ext_init(&wrx);
    CHECK(wrx.ext.size == sizeof(struct transcribe_whisper_run_ext));
    CHECK(wrx.ext.kind == TRANSCRIBE_EXT_KIND_WHISPER_RUN);
    CHECK(wrx.initial_prompt == NULL);
    CHECK(wrx.prompt_tokens == NULL);
    CHECK(wrx.n_prompt_tokens == 0);
    CHECK(wrx.prompt_condition == TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT);
    CHECK(wrx.condition_on_prev_tokens == false);
    CHECK(wrx.max_prev_context_tokens == 223);
    CHECK(wrx.temperature == 0.0f);
    CHECK(wrx.temperature_inc == 0.2f);
    CHECK(wrx.compression_ratio_thold == 2.4f);
    CHECK(wrx.logprob_thold == -1.0f);
    CHECK(wrx.no_speech_thold == 0.6f);
    CHECK(wrx.seed == 0u);
    CHECK(wrx.max_initial_timestamp == 1.0f);

    struct transcribe_qwen3_asr_run_ext qrx;
    transcribe_qwen3_asr_run_ext_init(&qrx);
    CHECK(qrx.ext.size == sizeof(struct transcribe_qwen3_asr_run_ext));
    CHECK(qrx.ext.kind == TRANSCRIBE_EXT_KIND_QWEN3_ASR_RUN);
    CHECK(qrx.context == NULL);

    /* The disabled-sentinel defines must be usable as compile-time
     * constants with the right signs. */
    CHECK(TRANSCRIBE_WHISPER_THOLD_DISABLED > 0.0f);
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
    (void) level;
    (void) msg;
    (void) userdata;
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
    struct transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    struct transcribe_model * m = (struct transcribe_model *) 0xdeadbeef;

    /* NULL out_model -> INVALID_ARG. */
    CHECK(transcribe_model_load_file("ignored", &mp, NULL) == TRANSCRIBE_ERR_INVALID_ARG);

    /* NULL path -> INVALID_ARG and out_model is cleared. */
    CHECK(transcribe_model_load_file(NULL, &mp, &m) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(m == NULL);

    /* NULL params -> "all defaults", NOT an error. Proceeds past param
     * validation and fails only on the missing file. */
    m = (struct transcribe_model *) 0xdeadbeef;
    CHECK(transcribe_model_load_file("/__transcribe_smoke_does_not_exist__.gguf", NULL, &m) ==
          TRANSCRIBE_ERR_FILE_NOT_FOUND);
    CHECK(m == NULL);

    /* {0} params (struct_size == 0) is REJECTED in pre-1.0: a struct
     * with zero struct_size is treated as a forgot-to-init bug, not as
     * a defaults shortcut, regardless of the rest of the field values.
     * Defaults are reached by passing a NULL pointer (above). */
    struct transcribe_model_load_params mp0 = { 0 };
    m                                       = (struct transcribe_model *) 0xdeadbeef;
    CHECK(transcribe_model_load_file("/__transcribe_smoke_does_not_exist__.gguf", &mp0, &m) ==
          TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(m == NULL);

    /* gpu_device selection is validated during load against the live device
     * registry (in load_common::init_backends), not as an upfront reserved-
     * field check — so a nonzero gpu_device no longer short-circuits to
     * INVALID_ARG before the file is even opened. A missing file still
     * surfaces as FILE_NOT_FOUND regardless of gpu_device. */
    struct transcribe_model_load_params mp_dev;
    transcribe_model_load_params_init(&mp_dev);
    mp_dev.gpu_device = 1;
    m                 = (struct transcribe_model *) 0xdeadbeef;
    CHECK(transcribe_model_load_file("/__transcribe_smoke_does_not_exist__.gguf", &mp_dev, &m) ==
          TRANSCRIBE_ERR_FILE_NOT_FOUND);
    CHECK(m == NULL);

    /* Otherwise-valid call against a path that does not exist on disk
     * -> FILE_NOT_FOUND. */
    m = (struct transcribe_model *) 0xdeadbeef;
    CHECK(transcribe_model_load_file("/__transcribe_smoke_does_not_exist__.gguf", &mp, &m) ==
          TRANSCRIBE_ERR_FILE_NOT_FOUND);
    CHECK(m == NULL);

    /* Free on NULL is a no-op. */
    transcribe_model_free(NULL);
}

static void test_context_invalid(void) {
    struct transcribe_session_params cp;
    transcribe_session_params_init(&cp);
    struct transcribe_session * c = (struct transcribe_session *) 0xdeadbeef;

    CHECK(transcribe_session_init(NULL, &cp, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_session_init(NULL, &cp, &c) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(c == NULL);

    /* model==NULL is checked before params, so it yields INVALID_ARG
     * regardless of what the params look like. {0} params with struct_size
     * == 0 would otherwise be rejected with BAD_STRUCT_SIZE (see the
     * model-load smoke). */

    transcribe_session_free(NULL);
}

static void test_run_invalid(void) {
    struct transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    float pcm[8] = { 0.0f };

    CHECK(transcribe_run(NULL, pcm, 8, &rp) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_run((struct transcribe_session *) 0x1, NULL, 8, &rp) == TRANSCRIBE_ERR_INVALID_ARG);
    /* NULL params is no longer an error (means "all defaults"), so it
     * can't be probed with a fake session — it would proceed to
     * dereference it. NULL-params-defaults is exercised end-to-end in the
     * real-model smoke tests instead. */
    /* n_samples must be strictly positive: both negative and zero are
     * rejected before the session is dereferenced (matches stream_feed). */
    CHECK(transcribe_run((struct transcribe_session *) 0x1, pcm, -1, &rp) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_run((struct transcribe_session *) 0x1, pcm, 0, &rp) == TRANSCRIBE_ERR_INVALID_ARG);
}

static void test_open_invalid(void) {
    struct transcribe_session * s = (struct transcribe_session *) 0xdeadbeef;

    /* NULL out_session -> INVALID_ARG. */
    CHECK(transcribe_open("ignored", NULL, NULL, NULL) == TRANSCRIBE_ERR_INVALID_ARG);

    /* NULL path forwards to load_file -> INVALID_ARG; out_session cleared. */
    CHECK(transcribe_open(NULL, NULL, NULL, &s) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(s == NULL);

    /* NULL load/session params == defaults; missing file -> FILE_NOT_FOUND
     * with no partial session leaked. */
    s = (struct transcribe_session *) 0xdeadbeef;
    CHECK(transcribe_open("/__transcribe_smoke_does_not_exist__.gguf", NULL, NULL, &s) ==
          TRANSCRIBE_ERR_FILE_NOT_FOUND);
    CHECK(s == NULL);

    /* close / get_model are NULL-safe. */
    transcribe_close(NULL);
    CHECK(transcribe_get_model(NULL) == NULL);
}

static void test_model_introspection_null(void) {
    /* Per the contract, accessors are safe to call on NULL. */
    struct transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    /* Copy-out signature: NULL model -> INVALID_ARG; out buffer
     * untouched semantically (still zero-initialized by INIT). */
    CHECK(transcribe_model_get_capabilities(NULL, &caps) == TRANSCRIBE_ERR_INVALID_ARG);
    /* NULL out_caps -> INVALID_ARG. */
    CHECK(transcribe_model_get_capabilities((const struct transcribe_model *) 0x1, NULL) == TRANSCRIBE_ERR_INVALID_ARG);

    CHECK_STR_EMPTY(transcribe_model_arch_string(NULL));
    CHECK_STR_EMPTY(transcribe_model_variant_string(NULL));
    CHECK_STR_EMPTY(transcribe_model_backend(NULL));

    /* The kind probe is safe on NULL and returns false for any slot. */
    CHECK(transcribe_model_accepts_ext_kind(NULL, TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_PARAKEET_STREAM) ==
          false);
    CHECK(transcribe_model_accepts_ext_kind(NULL, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_WHISPER_RUN) == false);
    CHECK(transcribe_model_accepts_ext_kind(NULL, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_QWEN3_ASR_RUN) == false);

    /* The feature probe is also NULL-safe and returns false for every
     * known feature value plus any out-of-range enum. */
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_INITIAL_PROMPT) == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK) == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_LONG_FORM) == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_CANCELLATION) == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_PNC) == false);
    CHECK(transcribe_model_supports(NULL, TRANSCRIBE_FEATURE_ITN) == false);
    CHECK(transcribe_model_supports(NULL, (transcribe_feature) 9999) == false);
}

static void test_ext_check(void) {
    /* NULL ext is always OK (family decides what NULL means). */
    CHECK(transcribe_ext_check(NULL, 0x12345678u, 8) == TRANSCRIBE_OK);

    /* Wrong kind -> INVALID_ARG. */
    struct transcribe_parakeet_stream_ext pk;
    transcribe_parakeet_stream_ext_init(&pk);
    CHECK(transcribe_ext_check(&pk.ext, TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM,
                               sizeof(struct transcribe_parakeet_buffered_stream_ext)) == TRANSCRIBE_ERR_INVALID_ARG);

    /* Right kind + size big enough -> OK. */
    CHECK(transcribe_ext_check(&pk.ext, TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
                               sizeof(struct transcribe_parakeet_stream_ext)) == TRANSCRIBE_OK);

    /* Size too small even to cover the common header -> BAD_STRUCT_SIZE
     * before reading kind. */
    struct transcribe_parakeet_stream_ext pk_tiny;
    transcribe_parakeet_stream_ext_init(&pk_tiny);
    pk_tiny.ext.size = sizeof(struct transcribe_ext) - 1;
    CHECK(transcribe_ext_check(&pk_tiny.ext, TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
                               sizeof(struct transcribe_parakeet_stream_ext)) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    /* Right kind but size too small -> BAD_STRUCT_SIZE. */
    struct transcribe_parakeet_stream_ext pk_small;
    transcribe_parakeet_stream_ext_init(&pk_small);
    pk_small.ext.size = sizeof(struct transcribe_ext); /* below the struct's prefix */
    CHECK(transcribe_ext_check(&pk_small.ext, TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
                               sizeof(struct transcribe_parakeet_stream_ext)) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    /* Qwen uses the same generic header but has its own typed minimum. */
    struct transcribe_qwen3_asr_run_ext qrx;
    transcribe_qwen3_asr_run_ext_init(&qrx);
    qrx.ext.size = sizeof(struct transcribe_ext);
    CHECK(transcribe_ext_check(&qrx.ext, TRANSCRIBE_EXT_KIND_QWEN3_ASR_RUN,
                               sizeof(struct transcribe_qwen3_asr_run_ext)) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

static void test_tokenize_null(void) {
    int32_t tokens[8];
    /* NULL model -> INT_MIN (hard error). */
    CHECK(transcribe_tokenize(NULL, "hello", tokens, 8) == INT_MIN);
    /* NULL text -> INT_MIN. */
    CHECK(transcribe_tokenize((const struct transcribe_model *) 0x1, NULL, tokens, 8) == INT_MIN);
}

static int g_abort_calls = 0;

static bool abort_cb_false(void * u) {
    (void) u;
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

    struct transcribe_whisper_chunk_trace tr;
    transcribe_whisper_chunk_trace_init(&tr);
    CHECK(transcribe_get_whisper_chunk_trace(NULL, 0, &tr) == TRANSCRIBE_OK);
    CHECK(tr.struct_size == sizeof(struct transcribe_whisper_chunk_trace));
    CHECK(tr.t0_ms == 0);
    CHECK(tr.t1_ms == 0);
    CHECK(tr.temperature_used == 0.0f);
    CHECK(tr.compression_ratio == 0.0f);
    CHECK(tr.avg_logprob == 0.0f);
    CHECK(tr.no_speech_prob == 0.0f);
    CHECK(tr.no_speech_triggered == false);
    CHECK(tr.n_fallbacks == 0);

    /* Out-of-range index against NULL succeeds the same way. */
    struct transcribe_whisper_chunk_trace tr2;
    transcribe_whisper_chunk_trace_init(&tr2);
    CHECK(transcribe_get_whisper_chunk_trace(NULL, 42, &tr2) == TRANSCRIBE_OK);
    CHECK(tr2.t0_ms == 0);

    /* NULL out_trace is rejected. */
    CHECK(transcribe_get_whisper_chunk_trace(NULL, 0, NULL) == TRANSCRIBE_ERR_INVALID_ARG);

    /* Zero struct_size is rejected. */
    struct transcribe_whisper_chunk_trace tr_bad = { 0 };
    CHECK(transcribe_get_whisper_chunk_trace(NULL, 0, &tr_bad) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

static void test_timings_null(void) {
    /* NULL ctx -> INVALID_ARG. NULL out -> INVALID_ARG. {0} out ->
     * BAD_STRUCT_SIZE. */
    struct transcribe_timings tm;
    transcribe_timings_init(&tm);
    CHECK(transcribe_get_timings(NULL, &tm) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_get_timings((const struct transcribe_session *) 0x1, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    struct transcribe_timings tm0 = { 0 };
    CHECK(transcribe_get_timings((const struct transcribe_session *) 0x1, &tm0) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

static void test_result_accessors_null(void) {
    const struct transcribe_session * ctx = NULL;

    /* Top level. */
    CHECK_STR_EMPTY(transcribe_full_text(ctx));
    CHECK(transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_NONE);
    CHECK(transcribe_n_segments(ctx) == 0);
    CHECK(transcribe_n_words(ctx) == 0);
    CHECK(transcribe_n_tokens(ctx) == 0);

    /* Segment row: NULL ctx → OK, struct stays zero-init (text=NULL). */
    struct transcribe_segment seg;
    transcribe_segment_init(&seg);
    CHECK(transcribe_get_segment(ctx, 0, &seg) == TRANSCRIBE_OK);
    CHECK(seg.struct_size == sizeof(struct transcribe_segment));
    CHECK(seg.text == NULL);
    CHECK(seg.t0_ms == 0);
    CHECK(seg.t1_ms == 0);
    CHECK(seg.first_word == 0);
    CHECK(seg.n_words == 0);
    CHECK(seg.first_token == 0);
    CHECK(seg.n_tokens == 0);

    /* Word row. */
    struct transcribe_word wrd;
    transcribe_word_init(&wrd);
    CHECK(transcribe_get_word(ctx, 0, &wrd) == TRANSCRIBE_OK);
    CHECK(wrd.struct_size == sizeof(struct transcribe_word));
    CHECK(wrd.text == NULL);
    CHECK(wrd.t0_ms == 0);
    CHECK(wrd.t1_ms == 0);
    CHECK(wrd.seg_index == 0);
    CHECK(wrd.first_token == 0);
    CHECK(wrd.n_tokens == 0);

    /* Token row. p is 0.0f (not NaN) on the zero-init path; bindings
     * distinguish "row not present" via text==NULL, not via NaN-on-p. */
    struct transcribe_token tok;
    transcribe_token_init(&tok);
    CHECK(transcribe_get_token(ctx, 0, &tok) == TRANSCRIBE_OK);
    CHECK(tok.struct_size == sizeof(struct transcribe_token));
    CHECK(tok.text == NULL);
    CHECK(tok.id == 0);
    CHECK(tok.p == 0.0f);
    CHECK(tok.t0_ms == 0);
    CHECK(tok.t1_ms == 0);
    CHECK(tok.seg_index == 0);
    CHECK(tok.word_index == 0);

    /* NULL out_ptr -> INVALID_ARG. */
    CHECK(transcribe_get_segment(ctx, 0, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_get_word(ctx, 0, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_get_token(ctx, 0, NULL) == TRANSCRIBE_ERR_INVALID_ARG);

    /* Uninitialized struct_size (== 0) -> BAD_STRUCT_SIZE. */
    struct transcribe_segment seg_bad = { 0 };
    CHECK(transcribe_get_segment(ctx, 0, &seg_bad) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

static void test_stream_state_values(void) {
    /* The state enum values are part of the public ABI — pin them. */
    CHECK(TRANSCRIBE_STREAM_IDLE == 0);
    CHECK(TRANSCRIBE_STREAM_ACTIVE == 1);
    CHECK(TRANSCRIBE_STREAM_FINISHED == 2);
    CHECK(TRANSCRIBE_STREAM_FAILED == 3);
    CHECK(TRANSCRIBE_STREAM_COMMIT_AUTO == 0);
    CHECK(TRANSCRIBE_STREAM_COMMIT_ON_FINALIZE == 1);
    CHECK(TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX == 2);
}

static void test_stream_accessors_null(void) {
    const struct transcribe_session * ctx = NULL;
    /* Every streaming accessor is safe to call on NULL and returns the
     * documented sentinel. */
    CHECK(transcribe_stream_get_state(ctx) == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_stream_revision(ctx) == 0);
    CHECK(transcribe_stream_n_committed_segments(ctx) == 0);
    CHECK(transcribe_stream_n_committed_words(ctx) == 0);
    CHECK(transcribe_stream_n_committed_tokens(ctx) == 0);
    CHECK(transcribe_stream_last_status(ctx) == TRANSCRIBE_OK);

    struct transcribe_stream_text text;
    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(ctx, &text) == TRANSCRIBE_OK);
    CHECK(text.full_text != NULL);
    CHECK(text.full_text[0] == '\0');
    CHECK(text.full_text_bytes == 0);
    CHECK(text.committed_text != NULL);
    CHECK(text.committed_text[0] == '\0');
    CHECK(text.committed_text_bytes == 0);
    CHECK(text.tentative_text != NULL);
    CHECK(text.tentative_text[0] == '\0');
    CHECK(text.tentative_text_bytes == 0);
}

static void test_stream_entries_null(void) {
    /* NULL ctx into every entry point: begin/feed/finalize report
     * INVALID_ARG; reset is a no-op. */
    struct transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    struct transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    float pcm[1] = { 0.0f };

    CHECK(transcribe_stream_begin(NULL, &rp, &sp) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_feed(NULL, pcm, 1, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_finalize(NULL, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
    /* reset is void; calling it on NULL must not crash. */
    transcribe_stream_reset(NULL);

    /* {0} update -> BAD_STRUCT_SIZE on feed/finalize. The struct_size
     * check fires before ctx is dereferenced, so passing a non-NULL
     * fake ctx pointer is fine here. */
    struct transcribe_stream_update upd0 = { 0 };
    CHECK(transcribe_stream_feed((struct transcribe_session *) 0x1, pcm, 1, &upd0) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(transcribe_stream_finalize((struct transcribe_session *) 0x1, &upd0) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    struct transcribe_stream_text text0 = { 0 };
    CHECK(transcribe_stream_get_text(NULL, &text0) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(transcribe_stream_get_text(NULL, NULL) == TRANSCRIBE_ERR_INVALID_ARG);
}

/* Input-limits ABI surface (no model needed): the session-limits struct, the
 * get_limits arg validation, and the supplemental truncation flag. The
 * model-dependent behavior (real effective values, INPUT_TOO_LONG / OUTPUT_
 * TRUNCATED firing) is exercised by the family real-smoke tests. */
static void test_session_limits_abi(void) {
    struct transcribe_session_limits lim;
    transcribe_session_limits_init(&lim);
    CHECK(lim.struct_size == sizeof(struct transcribe_session_limits));
    CHECK(lim.effective_n_ctx == 0);
    CHECK(lim.effective_max_audio_ms == 0);
    CHECK(lim.max_kv_bytes == 0);

    /* NULL args -> INVALID_ARG (checked before struct_size / model deref). */
    CHECK(transcribe_session_get_limits(NULL, &lim) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_session_get_limits((struct transcribe_session *) 0x1, NULL) == TRANSCRIBE_ERR_INVALID_ARG);

    /* Non-NULL session + struct_size 0 -> BAD_STRUCT_SIZE, returned before the
     * session is dereferenced (mirrors the stream-param fake-handle tests). */
    struct transcribe_session_limits lim0 = { 0 };
    CHECK(transcribe_session_get_limits((struct transcribe_session *) 0x1, &lim0) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    /* Supplemental truncation flag on a NULL session is false, not a crash. */
    CHECK(transcribe_was_truncated(NULL) == false);
}

int main(void) {
    test_status_string();
    test_version();
    test_abi_metadata();
    test_backend_devices();
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
    test_session_limits_abi();

    if (g_failures > 0) {
        fprintf(stderr, "api_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    fprintf(stdout, "api_smoke: ok\n");
    return EXIT_SUCCESS;
}
