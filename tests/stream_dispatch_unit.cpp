// stream_dispatch_unit.cpp - dispatcher-level streaming entry-point tests.
//
// These exercise the parts of the streaming surface that do not depend
// on a loaded model: state machine rejections, update zero-init,
// nullable-update handling, transcribe_run rejection while ACTIVE,
// reset behavior from each state, and the safe-sentinel returns of the
// streaming accessors when ctx is NULL or in IDLE.
//
// Capability-gated paths (NOT_IMPLEMENTED for non-streaming families,
// TRANSLATE rejection, language validation) require a real model and
// land in Phase 3 against an existing non-streaming arch.

#include "transcribe-context.h"
#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;
int g_begin_calls = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

void test_accessors_on_null_ctx() {
    CHECK(transcribe_stream_get_state(nullptr) == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_stream_revision(nullptr) == 0);
    CHECK(transcribe_stream_n_committed_segments(nullptr) == 0);
    CHECK(transcribe_stream_n_committed_words(nullptr) == 0);
    CHECK(transcribe_stream_n_committed_tokens(nullptr) == 0);
    CHECK(transcribe_stream_last_status(nullptr) == TRANSCRIBE_OK);
}

void test_accessors_on_idle_ctx() {
    transcribe_context ctx;
    CHECK(transcribe_stream_get_state(&ctx) == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_stream_revision(&ctx) == 0);
    CHECK(transcribe_stream_n_committed_segments(&ctx) == 0);
    CHECK(transcribe_stream_n_committed_words(&ctx) == 0);
    CHECK(transcribe_stream_n_committed_tokens(&ctx) == 0);
    CHECK(transcribe_stream_last_status(&ctx) == TRANSCRIBE_OK);
}

void test_default_params() {
    const transcribe_stream_params p = transcribe_stream_default_params();
    CHECK(p.family == nullptr);
    CHECK(p.struct_size == sizeof(transcribe_stream_params));
}

void test_begin_null_args() {
    // ctx == NULL
    const transcribe_params rp = transcribe_default_params();
    const transcribe_stream_params sp = transcribe_stream_default_params();
    CHECK(transcribe_stream_begin(nullptr, &rp, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    transcribe_context ctx;
    CHECK(transcribe_stream_begin(&ctx, nullptr, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_begin(&ctx, &rp, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    // None of these should have moved the context off IDLE.
    CHECK(transcribe_stream_get_state(&ctx) == TRANSCRIBE_STREAM_IDLE);
}

void test_begin_rejected_when_active() {
    transcribe_context ctx;
    ctx.stream_state = TRANSCRIBE_STREAM_ACTIVE;

    const transcribe_params rp = transcribe_default_params();
    const transcribe_stream_params sp = transcribe_stream_default_params();
    CHECK(transcribe_stream_begin(&ctx, &rp, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    // Still ACTIVE — rejection must not clear lifecycle.
    CHECK(transcribe_stream_get_state(&ctx) == TRANSCRIBE_STREAM_ACTIVE);
}

void test_begin_no_model_returns_not_implemented() {
    // ctx with no model still reaches the model/arch check.
    transcribe_context ctx;
    const transcribe_params rp = transcribe_default_params();
    const transcribe_stream_params sp = transcribe_stream_default_params();
    CHECK(transcribe_stream_begin(&ctx, &rp, &sp) ==
          TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_stream_get_state(&ctx) == TRANSCRIBE_STREAM_IDLE);
}

transcribe_status fake_stream_begin(
    transcribe_context *              ctx,
    const transcribe_params *         run_params,
    const transcribe_stream_params *  stream_params)
{
    (void)ctx;
    (void)run_params;
    (void)stream_params;
    ++g_begin_calls;
    return TRANSCRIBE_OK;
}

transcribe_status fake_stream_feed(
    transcribe_context *     ctx,
    const float *            pcm,
    int                      n_samples,
    transcribe_stream_update * update)
{
    (void)ctx;
    (void)pcm;
    (void)n_samples;
    (void)update;
    return TRANSCRIBE_OK;
}

transcribe_status fake_stream_finalize(
    transcribe_context *      ctx,
    transcribe_stream_update * update)
{
    (void)ctx;
    (void)update;
    return TRANSCRIBE_OK;
}

bool fake_accepts_no_ext(
    const transcribe_model * model,
    uint32_t                 kind)
{
    (void)model;
    (void)kind;
    return false;
}

void test_begin_rejects_unknown_ext_kind_before_hook() {
    const transcribe::Arch arch = {
        "fake-stream",
        nullptr,
        nullptr,
        nullptr,
        fake_stream_begin,
        fake_stream_feed,
        fake_stream_finalize,
        nullptr,
        fake_accepts_no_ext,
    };

    transcribe_model model;
    model.arch = &arch;
    model.caps.supports_streaming = true;
    model.caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    transcribe_context ctx;
    ctx.model        = &model;
    ctx.has_result   = true;
    ctx.full_text    = "previous result";

    transcribe_params rp = TRANSCRIBE_PARAMS_INIT;
    transcribe_stream_params sp = TRANSCRIBE_STREAM_PARAMS_INIT;
    const transcribe_ext ext = { sizeof(transcribe_ext), 0x58585858u };
    sp.family = &ext;

    g_begin_calls = 0;
    CHECK(transcribe_stream_begin(&ctx, &rp, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(g_begin_calls == 0);
    CHECK(transcribe_stream_get_state(&ctx) == TRANSCRIBE_STREAM_IDLE);
    CHECK(ctx.has_result);
    CHECK(ctx.full_text == "previous result");
}

void test_begin_rejects_tiny_ext_before_hook() {
    const transcribe::Arch arch = {
        "fake-stream",
        nullptr,
        nullptr,
        nullptr,
        fake_stream_begin,
        fake_stream_feed,
        fake_stream_finalize,
        nullptr,
        fake_accepts_no_ext,
    };

    transcribe_model model;
    model.arch = &arch;
    model.caps.supports_streaming = true;
    model.caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    transcribe_context ctx;
    ctx.model      = &model;
    ctx.has_result = true;
    ctx.full_text  = "previous result";

    transcribe_params rp = TRANSCRIBE_PARAMS_INIT;
    transcribe_stream_params sp = TRANSCRIBE_STREAM_PARAMS_INIT;
    transcribe_ext ext = { 0, 0 };
    sp.family = &ext;

    g_begin_calls = 0;
    CHECK(transcribe_stream_begin(&ctx, &rp, &sp) ==
          TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(g_begin_calls == 0);
    CHECK(transcribe_stream_get_state(&ctx) == TRANSCRIBE_STREAM_IDLE);
    CHECK(ctx.has_result);
    CHECK(ctx.full_text == "previous result");
}

void test_feed_rejects_idle() {
    transcribe_context ctx;
    float pcm = 0.0f;
    transcribe_stream_update upd = TRANSCRIBE_STREAM_UPDATE_INIT;
    upd.result_changed = true; // dirty sentinel — must be zeroed
    upd.is_final       = true;
    upd.revision       = 42;
    upd.input_received_ms = 999;

    CHECK(transcribe_stream_feed(&ctx, &pcm, 1, &upd) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    // Dispatcher zero-inits update before the state check returns,
    // preserving the caller's struct_size.
    CHECK(upd.struct_size       == sizeof(transcribe_stream_update));
    CHECK(upd.result_changed    == false);
    CHECK(upd.is_final          == false);
    CHECK(upd.revision          == 0);
    CHECK(upd.input_received_ms == 0);
}

void test_feed_rejects_finished_and_failed() {
    transcribe_context ctx;
    float pcm = 0.0f;

    ctx.stream_state = TRANSCRIBE_STREAM_FINISHED;
    CHECK(transcribe_stream_feed(&ctx, &pcm, 1, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(ctx.stream_state == TRANSCRIBE_STREAM_FINISHED);

    ctx.stream_state = TRANSCRIBE_STREAM_FAILED;
    CHECK(transcribe_stream_feed(&ctx, &pcm, 1, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(ctx.stream_state == TRANSCRIBE_STREAM_FAILED);
}

void test_feed_rejects_null_ctx_and_bad_input() {
    float pcm = 0.0f;
    CHECK(transcribe_stream_feed(nullptr, &pcm, 1, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    transcribe_context ctx;
    ctx.stream_state = TRANSCRIBE_STREAM_ACTIVE;

    // Negative n_samples
    CHECK(transcribe_stream_feed(&ctx, &pcm, -1, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    // Zero n_samples — polling without audio goes through the
    // accessors, not feed.
    CHECK(transcribe_stream_feed(&ctx, &pcm, 0, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    // pcm NULL is rejected unconditionally, regardless of n_samples.
    CHECK(transcribe_stream_feed(&ctx, nullptr, 16000, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_feed(&ctx, nullptr, 0, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    // The ACTIVE state survives every malformed-input rejection
    // because the dispatcher returns before the family hook runs.
    CHECK(transcribe_stream_get_state(&ctx) == TRANSCRIBE_STREAM_ACTIVE);
}

void test_feed_active_no_hook_returns_not_implemented() {
    transcribe_context ctx;
    ctx.stream_state = TRANSCRIBE_STREAM_ACTIVE;
    float pcm = 0.0f;
    // No model → defensive NOT_IMPLEMENTED branch.
    CHECK(transcribe_stream_feed(&ctx, &pcm, 1, nullptr) ==
          TRANSCRIBE_ERR_NOT_IMPLEMENTED);
}

void test_finalize_rejects_non_active() {
    transcribe_context ctx;

    // IDLE
    CHECK(transcribe_stream_finalize(&ctx, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    ctx.stream_state = TRANSCRIBE_STREAM_FINISHED;
    CHECK(transcribe_stream_finalize(&ctx, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    ctx.stream_state = TRANSCRIBE_STREAM_FAILED;
    CHECK(transcribe_stream_finalize(&ctx, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    CHECK(transcribe_stream_finalize(nullptr, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
}

void test_finalize_update_zeroinit() {
    transcribe_context ctx;
    transcribe_stream_update upd = TRANSCRIBE_STREAM_UPDATE_INIT;
    upd.result_changed   = true;
    upd.is_final         = false; // will be cleared then forced true on success path
    upd.revision         = 99;
    upd.audio_committed_ms = 7;

    // No model + ACTIVE → returns NOT_IMPLEMENTED, but the dispatcher
    // zero-inits update before returning so the caller sees a clean
    // struct (is_final is only forced true after a hook call, which
    // doesn't happen here — verify the early-return path leaves
    // update fully zeroed and preserves the caller's struct_size).
    ctx.stream_state = TRANSCRIBE_STREAM_ACTIVE;
    CHECK(transcribe_stream_finalize(&ctx, &upd) ==
          TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(upd.struct_size        == sizeof(transcribe_stream_update));
    CHECK(upd.result_changed     == false);
    CHECK(upd.is_final           == false);
    CHECK(upd.revision           == 0);
    CHECK(upd.audio_committed_ms == 0);
}

void test_reset_from_each_state() {
    // IDLE: no-op transition, must remain IDLE.
    {
        transcribe_context ctx;
        transcribe_stream_reset(&ctx);
        CHECK(ctx.stream_state == TRANSCRIBE_STREAM_IDLE);
    }
    // FINISHED: clears result + counters, returns to IDLE.
    {
        transcribe_context ctx;
        ctx.stream_state         = TRANSCRIBE_STREAM_FINISHED;
        ctx.has_result           = true;
        ctx.full_text            = "stale";
        ctx.stream_revision      = 7;
        ctx.n_committed_tokens   = 5;
        ctx.stream_last_status   = TRANSCRIBE_ERR_OOM;
        ctx.stream_audio_input_us = 12345;
        transcribe_stream_reset(&ctx);
        CHECK(ctx.stream_state          == TRANSCRIBE_STREAM_IDLE);
        CHECK(ctx.has_result            == false);
        CHECK(ctx.full_text.empty());
        CHECK(ctx.stream_revision       == 0);
        CHECK(ctx.n_committed_tokens    == 0);
        CHECK(ctx.stream_last_status    == TRANSCRIBE_OK);
        CHECK(ctx.stream_audio_input_us == 0);
    }
    // FAILED: same as FINISHED — clear everything, back to IDLE.
    {
        transcribe_context ctx;
        ctx.stream_state         = TRANSCRIBE_STREAM_FAILED;
        ctx.stream_last_status   = TRANSCRIBE_ERR_ABORTED;
        ctx.was_aborted          = true;
        ctx.n_committed_segments = 3;
        transcribe_stream_reset(&ctx);
        CHECK(ctx.stream_state          == TRANSCRIBE_STREAM_IDLE);
        CHECK(ctx.stream_last_status    == TRANSCRIBE_OK);
        CHECK(ctx.was_aborted           == false);
        CHECK(ctx.n_committed_segments  == 0);
    }
    // NULL ctx — no-op.
    transcribe_stream_reset(nullptr);
}

void test_run_rejected_while_active() {
    transcribe_context ctx;
    ctx.stream_state = TRANSCRIBE_STREAM_ACTIVE;
    ctx.has_result   = true;
    ctx.full_text    = "active stream result";

    const transcribe_params rp = transcribe_default_params();
    float pcm = 0.0f;
    const transcribe_status st = transcribe_run(&ctx, &pcm, 1, &rp);
    CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);
    // Rejection must NOT clear the active stream's result snapshot.
    CHECK(ctx.has_result);
    CHECK(ctx.full_text == "active stream result");
    CHECK(ctx.stream_state == TRANSCRIBE_STREAM_ACTIVE);
}

void test_run_clears_stream_snapshot_from_finished() {
    transcribe_context ctx;
    ctx.stream_state         = TRANSCRIBE_STREAM_FINISHED;
    ctx.stream_revision      = 12;
    ctx.n_committed_tokens   = 4;
    ctx.stream_last_status   = TRANSCRIBE_ERR_BACKEND;
    ctx.stream_audio_input_us = 55555;
    ctx.has_result           = true;
    ctx.full_text            = "old run text";

    const transcribe_params rp = transcribe_default_params();
    float pcm = 0.0f;
    // No model → NOT_IMPLEMENTED, but clear_result + state reset must
    // have already run by the time we get here.
    const transcribe_status st = transcribe_run(&ctx, &pcm, 1, &rp);
    CHECK(st == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(ctx.has_result          == false);
    CHECK(ctx.full_text.empty());
    CHECK(ctx.stream_revision     == 0);
    CHECK(ctx.n_committed_tokens  == 0);
    CHECK(ctx.stream_last_status  == TRANSCRIBE_OK);
    CHECK(ctx.stream_audio_input_us == 0);
    CHECK(ctx.stream_state        == TRANSCRIBE_STREAM_IDLE);
}

void test_clear_result_preserves_lifecycle() {
    // clear_result is called by run AND begin and explicitly preserves
    // stream_state — only the streaming dispatcher transitions
    // lifecycle. Verify directly so a future refactor doesn't silently
    // tangle the two.
    transcribe_context ctx;
    ctx.stream_state       = TRANSCRIBE_STREAM_ACTIVE;
    ctx.stream_revision    = 9;
    ctx.n_committed_words  = 2;
    ctx.has_result         = true;
    ctx.full_text          = "snapshot";

    ctx.clear_result();
    CHECK(ctx.stream_state         == TRANSCRIBE_STREAM_ACTIVE);
    CHECK(ctx.has_result           == false);
    CHECK(ctx.full_text.empty());
    CHECK(ctx.stream_revision      == 0);
    CHECK(ctx.n_committed_words    == 0);
}

void test_last_status_survives_reads() {
    transcribe_context ctx;
    ctx.stream_state       = TRANSCRIBE_STREAM_FAILED;
    ctx.stream_last_status = TRANSCRIBE_ERR_ABORTED;

    CHECK(transcribe_stream_last_status(&ctx) == TRANSCRIBE_ERR_ABORTED);
    // Multiple reads do not consume or alter the value.
    CHECK(transcribe_stream_last_status(&ctx) == TRANSCRIBE_ERR_ABORTED);
    CHECK(ctx.stream_state == TRANSCRIBE_STREAM_FAILED);
}

} // namespace

int main() {
    test_accessors_on_null_ctx();
    test_accessors_on_idle_ctx();
    test_default_params();
    test_begin_null_args();
    test_begin_rejected_when_active();
    test_begin_no_model_returns_not_implemented();
    test_begin_rejects_unknown_ext_kind_before_hook();
    test_begin_rejects_tiny_ext_before_hook();
    test_feed_rejects_idle();
    test_feed_rejects_finished_and_failed();
    test_feed_rejects_null_ctx_and_bad_input();
    test_feed_active_no_hook_returns_not_implemented();
    test_finalize_rejects_non_active();
    test_finalize_update_zeroinit();
    test_reset_from_each_state();
    test_run_rejected_while_active();
    test_run_clears_stream_snapshot_from_finished();
    test_clear_result_preserves_lifecycle();
    test_last_status_survives_reads();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
