// stream_dispatch_unit.cpp - dispatcher-level streaming entry-point tests.
//
// These exercise the parts of the streaming surface that do not depend
// on a loaded model: state machine rejections, update zero-init,
// nullable-update handling, transcribe_run rejection while ACTIVE,
// reset behavior from each state, and the safe-sentinel returns of the
// streaming accessors when session is NULL or in IDLE.
//
// Capability-gated paths (NOT_IMPLEMENTED for non-streaming families,
// TRANSLATE rejection, language validation) require a real model and
// land in Phase 3 against an existing non-streaming arch.

#include "transcribe-session.h"
#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int g_failures = 0;
int g_begin_calls = 0;
const char * g_feed_texts[8] = {};
int g_n_feed_texts = 0;
int g_feed_text_i = 0;
const char * g_finalize_text = "";
const char * g_token_texts[8] = {};
int g_n_token_rows = 0;
int g_forced_n_committed_tokens = -1;
int g_forced_n_committed_words = -1;
int g_forced_n_committed_segments = -1;
int64_t g_feed_audio_committed_us = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

void reset_sequence_globals() {
    for (int i = 0; i < 8; ++i) {
        g_feed_texts[i] = nullptr;
        g_token_texts[i] = nullptr;
    }
    g_n_feed_texts = 0;
    g_feed_text_i = 0;
    g_finalize_text = "";
    g_n_token_rows = 0;
    g_forced_n_committed_tokens = -1;
    g_forced_n_committed_words = -1;
    g_forced_n_committed_segments = -1;
    g_feed_audio_committed_us = 0;
}

void test_accessors_on_null_ctx() {
    CHECK(transcribe_stream_get_state(nullptr) == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_stream_revision(nullptr) == 0);
    CHECK(transcribe_stream_n_committed_segments(nullptr) == 0);
    CHECK(transcribe_stream_n_committed_words(nullptr) == 0);
    CHECK(transcribe_stream_n_committed_tokens(nullptr) == 0);
    CHECK(transcribe_stream_last_status(nullptr) == TRANSCRIBE_OK);
}

void test_accessors_on_idle_ctx() {
    transcribe_session session;
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_stream_revision(&session) == 0);
    CHECK(transcribe_stream_n_committed_segments(&session) == 0);
    CHECK(transcribe_stream_n_committed_words(&session) == 0);
    CHECK(transcribe_stream_n_committed_tokens(&session) == 0);
    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_OK);
}

void test_default_params() {
    transcribe_stream_params p; transcribe_stream_params_init(&p);
    CHECK(p.family == nullptr);
    CHECK(p.struct_size == sizeof(transcribe_stream_params));
}

void test_begin_null_args() {
    // session == NULL is still rejected.
    transcribe_run_params rp; transcribe_run_params_init(&rp);
    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    CHECK(transcribe_stream_begin(nullptr, &rp, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    // NULL run/stream params now mean "all defaults", so they are no
    // longer rejected as INVALID_ARG. With this model-less context the
    // call proceeds past parameter validation to the model/capability
    // gate (NOT_IMPLEMENTED); the point under test is that NULL params
    // is accepted, not the downstream gate's exact code.
    transcribe_session session;
    CHECK(transcribe_stream_begin(&session, nullptr, &sp) !=
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_begin(&session, &rp, nullptr) !=
          TRANSCRIBE_ERR_INVALID_ARG);

    // None of these should have moved the context off IDLE.
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_IDLE);
}

void test_begin_rejected_when_active() {
    transcribe_session session;
    session.stream_state = TRANSCRIBE_STREAM_ACTIVE;

    transcribe_run_params rp; transcribe_run_params_init(&rp);
    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    CHECK(transcribe_stream_begin(&session, &rp, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    // Still ACTIVE — rejection must not clear lifecycle.
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_ACTIVE);
}

void test_begin_no_model_returns_not_implemented() {
    // session with no model still reaches the model/arch check.
    transcribe_session session;
    transcribe_run_params rp; transcribe_run_params_init(&rp);
    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    CHECK(transcribe_stream_begin(&session, &rp, &sp) ==
          TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_IDLE);
}

transcribe_status fake_stream_begin(
    transcribe_session *              session,
    const transcribe_run_params *         run_params,
    const transcribe_stream_params *  stream_params)
{
    (void)session;
    (void)run_params;
    (void)stream_params;
    ++g_begin_calls;
    return TRANSCRIBE_OK;
}

transcribe_status fake_stream_feed(
    transcribe_session *     session,
    const float *            pcm,
    int                      n_samples,
    transcribe_stream_update * update)
{
    (void)session;
    (void)pcm;
    (void)n_samples;
    (void)update;
    return TRANSCRIBE_OK;
}

transcribe_status fake_stream_finalize(
    transcribe_session *      session,
    transcribe_stream_update * update)
{
    (void)session;
    (void)update;
    return TRANSCRIBE_OK;
}

transcribe_status fake_sequence_stream_begin(
    transcribe_session *              session,
    const transcribe_run_params *     run_params,
    const transcribe_stream_params *  stream_params)
{
    (void)session;
    (void)run_params;
    (void)stream_params;
    g_feed_text_i = 0;
    return TRANSCRIBE_OK;
}

transcribe_status fake_sequence_stream_feed(
    transcribe_session *       session,
    const float *              pcm,
    int                        n_samples,
    transcribe_stream_update * update)
{
    (void)pcm;
    (void)n_samples;
    (void)update;
    if (g_feed_text_i < g_n_feed_texts) {
        session->full_text = g_feed_texts[g_feed_text_i++];
        session->has_result = true;
        if (g_n_token_rows > 0) {
            session->tokens.clear();
            for (int i = 0; i < g_n_token_rows; ++i) {
                transcribe_session::TokenEntry tok;
                tok.text = g_token_texts[i] != nullptr ? g_token_texts[i] : "";
                session->tokens.push_back(tok);
            }
        }
    }
    session->stream_audio_committed_us = g_feed_audio_committed_us;
    if (g_forced_n_committed_tokens >= 0) {
        session->n_committed_tokens = g_forced_n_committed_tokens;
    }
    if (g_forced_n_committed_words >= 0) {
        session->n_committed_words = g_forced_n_committed_words;
    }
    if (g_forced_n_committed_segments >= 0) {
        session->n_committed_segments = g_forced_n_committed_segments;
    }
    if (update != nullptr) {
        update->audio_committed_ms = g_feed_audio_committed_us / 1000;
    }
    return TRANSCRIBE_OK;
}

transcribe_status fake_sequence_stream_finalize(
    transcribe_session *       session,
    transcribe_stream_update * update)
{
    (void)update;
    session->full_text = g_finalize_text != nullptr ? g_finalize_text : "";
    session->has_result = true;
    return TRANSCRIBE_OK;
}

bool fake_accepts_no_ext(
    const transcribe_model * model,
    transcribe_ext_slot      slot,
    uint32_t                 kind)
{
    (void)model;
    (void)slot;
    (void)kind;
    return false;
}

int g_validate_calls = 0;

// stream_validate that always rejects, modeling a family preflight that
// found a bad extension value. Must run BEFORE the dispatcher clears the
// snapshot; must NOT lead to stream_begin being called.
transcribe_status fake_stream_validate_reject(
    const transcribe_session *        session,
    const transcribe_run_params *     run_params,
    const transcribe_stream_params *  stream_params)
{
    (void)session;
    (void)run_params;
    (void)stream_params;
    ++g_validate_calls;
    return TRANSCRIBE_ERR_INVALID_ARG;
}

void test_begin_rejects_unknown_ext_kind_before_hook() {
    const transcribe::Arch arch = {
        "fake-stream",
        nullptr,
        nullptr,
        nullptr,
        nullptr,  // run_batch
        nullptr,                  // stream_validate
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

    transcribe_session session;
    session.model        = &model;
    session.has_result   = true;
    session.full_text    = "previous result";

    transcribe_run_params rp; transcribe_run_params_init(&rp);
    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    const transcribe_ext ext = { sizeof(transcribe_ext), 0x58585858u };
    sp.family = &ext;

    g_begin_calls = 0;
    CHECK(transcribe_stream_begin(&session, &rp, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(g_begin_calls == 0);
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_IDLE);
    CHECK(session.has_result);
    CHECK(session.full_text == "previous result");
}

void test_begin_rejects_tiny_ext_before_hook() {
    const transcribe::Arch arch = {
        "fake-stream",
        nullptr,
        nullptr,
        nullptr,
        nullptr,  // run_batch
        nullptr,                  // stream_validate
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

    transcribe_session session;
    session.model      = &model;
    session.has_result = true;
    session.full_text  = "previous result";

    transcribe_run_params rp; transcribe_run_params_init(&rp);
    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    transcribe_ext ext = { 0, 0 };
    sp.family = &ext;

    g_begin_calls = 0;
    CHECK(transcribe_stream_begin(&session, &rp, &sp) ==
          TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(g_begin_calls == 0);
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_IDLE);
    CHECK(session.has_result);
    CHECK(session.full_text == "previous result");
}

// A family preflight (stream_validate) that rejects must behave like the
// other pre-hook failures: the previous result snapshot is preserved, the
// lifecycle stays put (no FAILED transition), and stream_begin is never
// called. This pins the dispatcher contract that lets a family validate
// extension values before the snapshot is cleared (see transcribe-arch.h
// stream_validate). Without it, a caller-side config typo would destroy
// the prior utterance's transcript.
void test_begin_family_preflight_reject_preserves_snapshot() {
    const transcribe::Arch arch = {
        "fake-stream",
        nullptr,
        nullptr,
        nullptr,
        nullptr,                      // run_batch
        fake_stream_validate_reject,  // stream_validate rejects
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

    // Seed a session that previously FAILED, carrying a non-OK
    // last_status and a partial snapshot. Using a non-OK seed (rather
    // than the default OK) is deliberate: it proves the dispatcher
    // leaves last_status UNTOUCHED on a preflight reject, not that it
    // forces OK. A test that started from OK could not tell the two
    // apart.
    transcribe_session session;
    session.model              = &model;
    session.has_result         = true;
    session.full_text          = "previous result";
    session.stream_state       = TRANSCRIBE_STREAM_FAILED;     // a prior failed stream
    session.stream_last_status = TRANSCRIBE_ERR_BACKEND;       // its failing status

    transcribe_run_params rp; transcribe_run_params_init(&rp);
    transcribe_stream_params sp; transcribe_stream_params_init(&sp);

    g_validate_calls = 0;
    g_begin_calls    = 0;
    CHECK(transcribe_stream_begin(&session, &rp, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    // Preflight ran, begin did not.
    CHECK(g_validate_calls == 1);
    CHECK(g_begin_calls    == 0);
    // Snapshot preserved, lifecycle untouched (NOT cleared, NOT moved).
    CHECK(session.has_result);
    CHECK(session.full_text == "previous result");
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_FAILED);
    // last_status untouched — the prior non-OK value survives a
    // pre-clear rejection unchanged (the dispatcher does not reset it
    // to OK, nor overwrite it with the rejection's status).
    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_ERR_BACKEND);
}

void test_feed_rejects_idle() {
    transcribe_session session;
    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    upd.result_changed = true; // dirty sentinel — must be zeroed
    upd.is_final       = true;
    upd.revision       = 42;
    upd.input_received_ms = 999;

    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) ==
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
    transcribe_session session;
    float pcm = 0.0f;

    session.stream_state = TRANSCRIBE_STREAM_FINISHED;
    CHECK(transcribe_stream_feed(&session, &pcm, 1, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(session.stream_state == TRANSCRIBE_STREAM_FINISHED);

    session.stream_state = TRANSCRIBE_STREAM_FAILED;
    CHECK(transcribe_stream_feed(&session, &pcm, 1, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(session.stream_state == TRANSCRIBE_STREAM_FAILED);
}

void test_feed_rejects_null_ctx_and_bad_input() {
    float pcm = 0.0f;
    CHECK(transcribe_stream_feed(nullptr, &pcm, 1, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    transcribe_session session;
    session.stream_state = TRANSCRIBE_STREAM_ACTIVE;

    // Negative n_samples
    CHECK(transcribe_stream_feed(&session, &pcm, -1, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    // Zero n_samples — polling without audio goes through the
    // accessors, not feed.
    CHECK(transcribe_stream_feed(&session, &pcm, 0, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    // pcm NULL is rejected unconditionally, regardless of n_samples.
    CHECK(transcribe_stream_feed(&session, nullptr, 16000, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_feed(&session, nullptr, 0, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    // The ACTIVE state survives every malformed-input rejection
    // because the dispatcher returns before the family hook runs.
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_ACTIVE);
}

void test_feed_active_no_hook_returns_not_implemented() {
    transcribe_session session;
    session.stream_state = TRANSCRIBE_STREAM_ACTIVE;
    float pcm = 0.0f;
    // No model → defensive NOT_IMPLEMENTED branch.
    CHECK(transcribe_stream_feed(&session, &pcm, 1, nullptr) ==
          TRANSCRIBE_ERR_NOT_IMPLEMENTED);
}

void test_finalize_rejects_non_active() {
    transcribe_session session;

    // IDLE
    CHECK(transcribe_stream_finalize(&session, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    session.stream_state = TRANSCRIBE_STREAM_FINISHED;
    CHECK(transcribe_stream_finalize(&session, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    session.stream_state = TRANSCRIBE_STREAM_FAILED;
    CHECK(transcribe_stream_finalize(&session, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);

    CHECK(transcribe_stream_finalize(nullptr, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
}

void test_finalize_update_zeroinit() {
    transcribe_session session;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    upd.result_changed   = true;
    upd.is_final         = false; // will be cleared then forced true on success path
    upd.revision         = 99;
    upd.audio_committed_ms = 7;

    // No model + ACTIVE → returns NOT_IMPLEMENTED, but the dispatcher
    // zero-inits update before returning so the caller sees a clean
    // struct (is_final is only forced true after a hook call, which
    // doesn't happen here — verify the early-return path leaves
    // update fully zeroed and preserves the caller's struct_size).
    session.stream_state = TRANSCRIBE_STREAM_ACTIVE;
    CHECK(transcribe_stream_finalize(&session, &upd) ==
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
        transcribe_session session;
        transcribe_stream_reset(&session);
        CHECK(session.stream_state == TRANSCRIBE_STREAM_IDLE);
    }
    // FINISHED: clears result + counters, returns to IDLE.
    {
        transcribe_session session;
        session.stream_state         = TRANSCRIBE_STREAM_FINISHED;
        session.has_result           = true;
        session.full_text            = "stale";
        session.stream_revision      = 7;
        session.n_committed_tokens   = 5;
        session.stream_last_status   = TRANSCRIBE_ERR_OOM;
        session.stream_audio_input_us = 12345;
        transcribe_stream_reset(&session);
        CHECK(session.stream_state          == TRANSCRIBE_STREAM_IDLE);
        CHECK(session.has_result            == false);
        CHECK(session.full_text.empty());
        CHECK(session.stream_revision       == 0);
        CHECK(session.n_committed_tokens    == 0);
        CHECK(session.stream_last_status    == TRANSCRIBE_OK);
        CHECK(session.stream_audio_input_us == 0);
    }
    // FAILED: same as FINISHED — clear everything, back to IDLE.
    {
        transcribe_session session;
        session.stream_state         = TRANSCRIBE_STREAM_FAILED;
        session.stream_last_status   = TRANSCRIBE_ERR_ABORTED;
        session.was_aborted          = true;
        session.n_committed_segments = 3;
        transcribe_stream_reset(&session);
        CHECK(session.stream_state          == TRANSCRIBE_STREAM_IDLE);
        CHECK(session.stream_last_status    == TRANSCRIBE_OK);
        CHECK(session.was_aborted           == false);
        CHECK(session.n_committed_segments  == 0);
    }
    // NULL session — no-op.
    transcribe_stream_reset(nullptr);
}

void test_run_rejected_while_active() {
    transcribe_session session;
    session.stream_state = TRANSCRIBE_STREAM_ACTIVE;
    session.has_result   = true;
    session.full_text    = "active stream result";

    transcribe_run_params rp; transcribe_run_params_init(&rp);
    float pcm = 0.0f;
    const transcribe_status st = transcribe_run(&session, &pcm, 1, &rp);
    CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);
    // Rejection must NOT clear the active stream's result snapshot.
    CHECK(session.has_result);
    CHECK(session.full_text == "active stream result");
    CHECK(session.stream_state == TRANSCRIBE_STREAM_ACTIVE);
}

void test_run_clears_stream_snapshot_from_finished() {
    transcribe_session session;
    session.stream_state         = TRANSCRIBE_STREAM_FINISHED;
    session.stream_revision      = 12;
    session.n_committed_tokens   = 4;
    session.stream_last_status   = TRANSCRIBE_ERR_BACKEND;
    session.stream_audio_input_us = 55555;
    session.has_result           = true;
    session.full_text            = "old run text";

    transcribe_run_params rp; transcribe_run_params_init(&rp);
    float pcm = 0.0f;
    // No model → NOT_IMPLEMENTED, but clear_result + state reset must
    // have already run by the time we get here.
    const transcribe_status st = transcribe_run(&session, &pcm, 1, &rp);
    CHECK(st == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(session.has_result          == false);
    CHECK(session.full_text.empty());
    CHECK(session.stream_revision     == 0);
    CHECK(session.n_committed_tokens  == 0);
    CHECK(session.stream_last_status  == TRANSCRIBE_OK);
    CHECK(session.stream_audio_input_us == 0);
    CHECK(session.stream_state        == TRANSCRIBE_STREAM_IDLE);
}

void test_clear_result_preserves_lifecycle() {
    // clear_result is called by run AND begin and explicitly preserves
    // stream_state — only the streaming dispatcher transitions
    // lifecycle. Verify directly so a future refactor doesn't silently
    // tangle the two.
    transcribe_session session;
    session.stream_state       = TRANSCRIBE_STREAM_ACTIVE;
    session.stream_revision    = 9;
    session.n_committed_words  = 2;
    session.has_result         = true;
    session.full_text          = "snapshot";

    session.clear_result();
    CHECK(session.stream_state         == TRANSCRIBE_STREAM_ACTIVE);
    CHECK(session.has_result           == false);
    CHECK(session.full_text.empty());
    CHECK(session.stream_revision      == 0);
    CHECK(session.n_committed_words    == 0);
}

void test_last_status_survives_reads() {
    transcribe_session session;
    session.stream_state       = TRANSCRIBE_STREAM_FAILED;
    session.stream_last_status = TRANSCRIBE_ERR_ABORTED;

    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_ERR_ABORTED);
    // Multiple reads do not consume or alter the value.
    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_ERR_ABORTED);
    CHECK(session.stream_state == TRANSCRIBE_STREAM_FAILED);
}

const transcribe::Arch & sequence_arch() {
    static const transcribe::Arch arch = {
        "fake-sequence-stream",
        nullptr,
        nullptr,
        nullptr,
        nullptr,                  // run_batch
        nullptr,
        fake_sequence_stream_begin,
        fake_sequence_stream_feed,
        fake_sequence_stream_finalize,
        nullptr,
        fake_accepts_no_ext,
    };
    return arch;
}

const transcribe::Arch & sequence_arch_named_parakeet() {
    static const transcribe::Arch arch = {
        "parakeet",
        nullptr,
        nullptr,
        nullptr,
        nullptr,                  // run_batch
        nullptr,
        fake_sequence_stream_begin,
        fake_sequence_stream_feed,
        fake_sequence_stream_finalize,
        nullptr,
        fake_accepts_no_ext,
    };
    return arch;
}

const transcribe::Arch & sequence_arch_named_moonshine_streaming() {
    static const transcribe::Arch arch = {
        "moonshine_streaming",
        nullptr,
        nullptr,
        nullptr,
        nullptr,                  // run_batch
        nullptr,
        fake_sequence_stream_begin,
        fake_sequence_stream_feed,
        fake_sequence_stream_finalize,
        nullptr,
        fake_accepts_no_ext,
    };
    return arch;
}

void init_streaming_model(transcribe_model & model) {
    reset_sequence_globals();
    model.arch = &sequence_arch();
    model.caps.supports_streaming = true;
    model.caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;
}

void test_stream_text_on_finalize_policy() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "hello wor";
    g_n_feed_texts = 1;
    g_finalize_text = "hello world";
    g_feed_audio_committed_us = 123000;

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.commit_policy = TRANSCRIBE_STREAM_COMMIT_ON_FINALIZE;
    CHECK(transcribe_stream_begin(&session, nullptr, &sp) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(upd.result_changed);
    CHECK(!upd.committed_changed);
    CHECK(upd.tentative_changed);
    CHECK(upd.audio_committed_ms == 0);
    CHECK(transcribe_stream_n_committed_tokens(&session) == 0);

    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.full_text, "hello wor") == 0);
    CHECK(std::strcmp(text.committed_text, "") == 0);
    CHECK(std::strcmp(text.tentative_text, "hello wor") == 0);

    transcribe_stream_update fin; transcribe_stream_update_init(&fin);
    CHECK(transcribe_stream_finalize(&session, &fin) == TRANSCRIBE_OK);
    CHECK(fin.is_final);
    CHECK(fin.result_changed);
    CHECK(fin.committed_changed);
    CHECK(fin.tentative_changed);
    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.full_text, "hello world") == 0);
    CHECK(std::strcmp(text.committed_text, "hello world") == 0);
    CHECK(std::strcmp(text.tentative_text, "") == 0);
}

void test_stream_auto_known_family_uses_family_stable_prefix() {
    transcribe_model model;
    init_streaming_model(model);
    model.arch = &sequence_arch_named_parakeet();
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "family stable";
    g_n_feed_texts = 1;
    g_n_token_rows = 1;
    g_token_texts[0] = "family stable";
    g_forced_n_committed_tokens = 1;

    CHECK(transcribe_stream_begin(&session, nullptr, nullptr) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);

    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "family stable") == 0);
    CHECK(std::strcmp(text.tentative_text, "") == 0);
}

void test_stream_auto_unknown_family_falls_back_to_generic_stable_prefix() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "generic stable";
    g_feed_texts[1] = "generic stable";
    g_feed_texts[2] = "generic stable";
    g_n_feed_texts = 3;
    g_n_token_rows = 1;
    g_token_texts[0] = "generic stable";
    g_forced_n_committed_tokens = 1;

    CHECK(transcribe_stream_begin(&session, nullptr, nullptr) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);

    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "") == 0);
    CHECK(std::strcmp(text.tentative_text, "generic stable") == 0);

    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(!upd.committed_changed);

    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "") == 0);
    CHECK(std::strcmp(text.tentative_text, "generic stable") == 0);

    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(upd.committed_changed);

    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "generic stable") == 0);
    CHECK(std::strcmp(text.tentative_text, "") == 0);
}

void test_stream_stable_prefix_known_family_uses_family_boundary() {
    transcribe_model model;
    init_streaming_model(model);
    model.arch = &sequence_arch_named_moonshine_streaming();
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "token stable";
    g_n_feed_texts = 1;
    g_n_token_rows = 1;
    g_token_texts[0] = "token stable";
    g_forced_n_committed_tokens = 1;

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.commit_policy = TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX;
    sp.stable_prefix_agreement_n = 3;
    CHECK(transcribe_stream_begin(&session, nullptr, &sp) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(upd.committed_changed);

    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "token stable") == 0);
    CHECK(std::strcmp(text.tentative_text, "") == 0);
}

void test_stream_family_token_prefix_maps_trimmed_leading_space() {
    transcribe_model model;
    init_streaming_model(model);
    model.arch = &sequence_arch_named_moonshine_streaming();
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "And";
    g_n_feed_texts = 1;
    g_n_token_rows = 1;
    g_token_texts[0] = " And";
    g_forced_n_committed_tokens = 1;

    CHECK(transcribe_stream_begin(&session, nullptr, nullptr) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(upd.committed_changed);

    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.full_text, "And") == 0);
    CHECK(std::strcmp(text.committed_text, "And") == 0);
    CHECK(std::strcmp(text.tentative_text, "") == 0);
}

void test_stream_text_stable_prefix_does_not_rewrite_committed() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "hello wor";
    g_feed_texts[1] = "hello world";
    g_feed_texts[2] = "hullo world!";
    g_n_feed_texts = 3;
    g_finalize_text = "hullo world!!";
    g_feed_audio_committed_us = 0;

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.commit_policy = TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX;
    sp.stable_prefix_agreement_n = 2;
    CHECK(transcribe_stream_begin(&session, nullptr, &sp) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "") == 0);
    CHECK(std::strcmp(text.tentative_text, "hello wor") == 0);

    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(upd.committed_changed);
    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.full_text, "hello world") == 0);
    CHECK(std::strcmp(text.committed_text, "hello wor") == 0);
    CHECK(std::strcmp(text.tentative_text, "ld") == 0);

    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(!upd.committed_changed);
    CHECK(upd.tentative_changed);
    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.full_text, "hullo world!") == 0);
    CHECK(std::strcmp(text.committed_text, "hello wor") == 0);
    CHECK(std::strcmp(text.tentative_text, "ld!") == 0);

    transcribe_stream_update fin; transcribe_stream_update_init(&fin);
    CHECK(transcribe_stream_finalize(&session, &fin) == TRANSCRIBE_OK);
    CHECK(!fin.committed_changed);
    CHECK(fin.tentative_changed);
    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.full_text, "hullo world!!") == 0);
    CHECK(std::strcmp(text.committed_text, "hello wor") == 0);
    CHECK(std::strcmp(text.tentative_text, "") == 0);
    CHECK(text.full_text_bytes == std::strlen("hullo world!!"));
    CHECK(text.raw_tentative_start_bytes == text.full_text_bytes);
}

void test_stream_text_clamps_raw_tentative_offset_after_shrink() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "hello wor";
    g_feed_texts[1] = "hello world";
    g_feed_texts[2] = "hi";
    g_n_feed_texts = 3;
    g_finalize_text = "hi";

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.commit_policy = TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX;
    sp.stable_prefix_agreement_n = 2;
    CHECK(transcribe_stream_begin(&session, nullptr, &sp) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);

    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "hello wor") == 0);

    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);

    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.full_text, "hi") == 0);
    CHECK(std::strcmp(text.committed_text, "hello wor") == 0);
    CHECK(std::strcmp(text.tentative_text, "") == 0);
    CHECK(text.full_text_bytes == 2);
    CHECK(text.raw_tentative_start_bytes == text.full_text_bytes);
}

void test_stream_committed_row_hints_can_exceed_current_rows() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "abc";
    g_n_feed_texts = 1;
    g_n_token_rows = 1;
    g_token_texts[0] = "abc";
    g_forced_n_committed_tokens = 7;
    g_forced_n_committed_words = 3;
    g_forced_n_committed_segments = 2;

    CHECK(transcribe_stream_begin(&session, nullptr, nullptr) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(transcribe_n_tokens(&session) == 1);
    CHECK(transcribe_stream_n_committed_tokens(&session) == 7);
    CHECK(transcribe_stream_n_committed_words(&session) == 3);
    CHECK(transcribe_stream_n_committed_segments(&session) == 2);
}

void test_stream_text_stable_prefix_floors_utf8_boundary() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    static const char cafe_acute[] = "caf\303\251"; // cafe + U+00E9
    static const char cafe_circ[]  = "caf\303\252"; // cafe + U+00EA
    g_feed_texts[0] = cafe_acute;
    g_feed_texts[1] = cafe_circ;
    g_n_feed_texts = 2;
    g_finalize_text = cafe_circ;
    g_feed_audio_committed_us = 0;

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.commit_policy = TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX;
    sp.stable_prefix_agreement_n = 2;
    CHECK(transcribe_stream_begin(&session, nullptr, &sp) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);

    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(upd.committed_changed);

    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.full_text, cafe_circ) == 0);
    CHECK(std::strcmp(text.committed_text, "caf") == 0);
    CHECK(text.committed_text_bytes == 3);
    CHECK(std::strcmp(text.tentative_text, "\303\252") == 0);
    CHECK(text.raw_tentative_start_bytes == 3);
}

void test_stream_text_stable_prefix_respects_agreement_n() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "agree now";
    g_feed_texts[1] = "agree now";
    g_feed_texts[2] = "agree now";
    g_n_feed_texts = 3;
    g_finalize_text = "agree now";

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.commit_policy = TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX;
    sp.stable_prefix_agreement_n = 3;
    CHECK(transcribe_stream_begin(&session, nullptr, &sp) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);

    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "") == 0);
    CHECK(std::strcmp(text.tentative_text, "agree now") == 0);

    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(upd.committed_changed);

    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "agree now") == 0);
    CHECK(std::strcmp(text.tentative_text, "") == 0);
}

void test_old_stream_params_ignores_new_tail_fields() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.struct_size = offsetof(transcribe_stream_params, commit_policy);
    sp.commit_policy =
        static_cast<transcribe_stream_commit_policy>(99);
    sp.stable_prefix_agreement_n = 99;

    CHECK(transcribe_stream_begin(&session, nullptr, &sp) == TRANSCRIBE_OK);
    CHECK(session.stream_commit_policy == TRANSCRIBE_STREAM_COMMIT_AUTO);
    CHECK(session.stream_stable_prefix_agreement_n == 0);
}

void test_stream_text_stable_prefix_intentionally_commits_partial_words() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "turn le";
    g_feed_texts[1] = "turn left";
    g_n_feed_texts = 2;
    g_finalize_text = "turn left";
    g_feed_audio_committed_us = 0;

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.commit_policy = TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX;
    sp.stable_prefix_agreement_n = 2;
    CHECK(transcribe_stream_begin(&session, nullptr, &sp) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);

    transcribe_stream_text text; transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(&session, &text) == TRANSCRIBE_OK);
    CHECK(std::strcmp(text.committed_text, "turn le") == 0);
    CHECK(std::strcmp(text.tentative_text, "ft") == 0);
}

void test_stream_update_old_prefix_tail_untouched() {
    transcribe_model model;
    init_streaming_model(model);
    transcribe_session session;
    session.model = &model;

    g_feed_texts[0] = "old prefix";
    g_n_feed_texts = 1;
    g_finalize_text = "old prefix";
    g_feed_audio_committed_us = 0;

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.commit_policy = TRANSCRIBE_STREAM_COMMIT_ON_FINALIZE;
    CHECK(transcribe_stream_begin(&session, nullptr, &sp) == TRANSCRIBE_OK);

    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    upd.struct_size = offsetof(transcribe_stream_update, committed_changed);
    upd.committed_changed = true;
    upd.tentative_changed = false;

    float pcm = 0.0f;
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(upd.result_changed);
    CHECK(upd.revision == transcribe_stream_revision(&session));
    CHECK(upd.committed_changed == true);
    CHECK(upd.tentative_changed == false);
}

void test_begin_rejects_bad_commit_params_before_clear() {
    transcribe_session session;
    session.has_result = true;
    session.full_text = "previous";

    transcribe_run_params rp; transcribe_run_params_init(&rp);

    transcribe_stream_params sp; transcribe_stream_params_init(&sp);
    sp.commit_policy = static_cast<transcribe_stream_commit_policy>(99);
    CHECK(transcribe_stream_begin(&session, &rp, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(session.has_result);
    CHECK(session.full_text == "previous");

    transcribe_stream_params_init(&sp);
    sp.stable_prefix_agreement_n = 33;
    CHECK(transcribe_stream_begin(&session, &rp, &sp) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(session.has_result);
    CHECK(session.full_text == "previous");
}

// ---------------------------------------------------------------------------
// Mid-stream failure / cancellation coverage.
//
// The fake hooks above always succeed; these exercise the dispatcher's
// ACTIVE -> FAILED transition, stream_last_status preservation, and the
// TRANSCRIBE_ERR_ABORTED / was_aborted distinction, none of which the
// success-only sequence hooks reach.
// ---------------------------------------------------------------------------

bool g_abort_flag = false;

bool abort_cb_returns_flag(void * userdata) {
    (void)userdata;
    return g_abort_flag;
}

// Writes a partial result, then fails — models a family that produced
// some output before hitting a terminal error mid-feed. The partial
// snapshot must remain readable (the dispatcher does not clear it on the
// failure path).
transcribe_status fake_failing_stream_feed(
    transcribe_session *       session,
    const float *              pcm,
    int                        n_samples,
    transcribe_stream_update * update)
{
    (void)pcm;
    (void)n_samples;
    (void)update;
    session->full_text  = "partial before failure";
    session->has_result = true;
    return TRANSCRIBE_ERR_BACKEND;
}

transcribe_status fake_failing_stream_finalize(
    transcribe_session *       session,
    transcribe_stream_update * update)
{
    (void)session;
    (void)update;
    return TRANSCRIBE_ERR_BACKEND;
}

// Polls the session abort callback the way a real family hook must. When
// the callback fires, poll_abort() sets session->was_aborted and we
// return the terminal ABORTED status; otherwise the feed succeeds.
transcribe_status fake_aborting_stream_feed(
    transcribe_session *       session,
    const float *              pcm,
    int                        n_samples,
    transcribe_stream_update * update)
{
    (void)pcm;
    (void)n_samples;
    (void)update;
    if (session->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    session->full_text  = "ok chunk";
    session->has_result = true;
    return TRANSCRIBE_OK;
}

const transcribe::Arch & failing_arch() {
    static const transcribe::Arch arch = {
        "fake-failing-stream",
        nullptr,
        nullptr,
        nullptr,
        nullptr,                     // run_batch
        nullptr,
        fake_stream_begin,           // begin succeeds -> ACTIVE
        fake_failing_stream_feed,
        fake_failing_stream_finalize,
        nullptr,
        fake_accepts_no_ext,
    };
    return arch;
}

const transcribe::Arch & aborting_arch() {
    static const transcribe::Arch arch = {
        "fake-aborting-stream",
        nullptr,
        nullptr,
        nullptr,
        nullptr,                  // run_batch
        nullptr,
        fake_stream_begin,
        fake_aborting_stream_feed,
        fake_stream_finalize,
        nullptr,
        fake_accepts_no_ext,
    };
    return arch;
}

void setup_streaming_model(transcribe_model & model, const transcribe::Arch & arch) {
    model.arch = &arch;
    model.caps.supports_streaming = true;
    model.caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;
}

void test_feed_failure_transitions_to_failed() {
    transcribe_model model;
    setup_streaming_model(model, failing_arch());
    transcribe_session session;
    session.model = &model;

    CHECK(transcribe_stream_begin(&session, nullptr, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_ACTIVE);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_ERR_BACKEND);

    // ACTIVE -> FAILED, terminal status preserved, partial result readable.
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_FAILED);
    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_ERR_BACKEND);
    CHECK(session.full_text == "partial before failure");
    // Not a caller abort.
    CHECK(transcribe_was_aborted(&session) == false);
    // update was zero-inited by the dispatcher and never marked final on a
    // failure path.
    CHECK(upd.is_final == false);

    // A FAILED stream rejects further feeds and the terminal status survives
    // the rejected call.
    CHECK(transcribe_stream_feed(&session, &pcm, 1, nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_FAILED);
    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_ERR_BACKEND);

    // reset re-arms: status back to OK, lifecycle back to IDLE.
    transcribe_stream_reset(&session);
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_OK);
}

void test_finalize_failure_transitions_to_failed() {
    transcribe_model model;
    setup_streaming_model(model, failing_arch());
    transcribe_session session;
    session.model = &model;

    // Begin only (no feed — failing_arch's feed would fail first); finalize
    // an ACTIVE stream and let the family hook fail.
    CHECK(transcribe_stream_begin(&session, nullptr, nullptr) == TRANSCRIBE_OK);
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);
    CHECK(transcribe_stream_finalize(&session, &upd) == TRANSCRIBE_ERR_BACKEND);
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_FAILED);
    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_ERR_BACKEND);
    // is_final marks the call site, not success: the dispatcher forces it
    // true on any finalize call (including a failed one) so a caller can
    // tell a finalize-originated update from a feed-originated one.
    CHECK(upd.is_final == true);
}

void test_feed_abort_sets_was_aborted() {
    transcribe_model model;
    setup_streaming_model(model, aborting_arch());
    transcribe_session session;
    session.model = &model;
    transcribe_set_abort_callback(&session, abort_cb_returns_flag, nullptr);

    CHECK(transcribe_stream_begin(&session, nullptr, nullptr) == TRANSCRIBE_OK);

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);

    // First feed: callback says "don't abort" -> succeeds, not aborted.
    g_abort_flag = false;
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_ACTIVE);
    CHECK(transcribe_was_aborted(&session) == false);

    // Second feed: callback fires -> ABORTED, FAILED, was_aborted set, and the
    // terminal status is ABORTED (distinguishable from a plain backend error).
    g_abort_flag = true;
    CHECK(transcribe_stream_feed(&session, &pcm, 1, &upd) == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_stream_get_state(&session) == TRANSCRIBE_STREAM_FAILED);
    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_was_aborted(&session) == true);
    // Partial output from the successful first feed is still readable.
    CHECK(session.full_text == "ok chunk");

    // begin re-arms was_aborted for a fresh stream.
    g_abort_flag = false;
    CHECK(transcribe_stream_begin(&session, nullptr, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_was_aborted(&session) == false);
    CHECK(transcribe_stream_last_status(&session) == TRANSCRIBE_OK);
}

// Two sessions derived from one shared, read-only model must keep fully
// independent stream lifecycles — the documented concurrency contract
// (one stream per session; advancing or finalizing one never disturbs the
// other).
void test_two_sessions_independent_streams() {
    transcribe_model model;
    setup_streaming_model(model, aborting_arch());

    transcribe_session a;
    transcribe_session b;
    a.model = &model;
    b.model = &model;
    // Only b installs an abort callback; a must stay unaffected when b is
    // cancelled below.
    transcribe_set_abort_callback(&b, abort_cb_returns_flag, nullptr);
    g_abort_flag = false;

    float pcm = 0.0f;
    transcribe_stream_update upd; transcribe_stream_update_init(&upd);

    // A enters ACTIVE; B is untouched.
    CHECK(transcribe_stream_begin(&a, nullptr, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_get_state(&a) == TRANSCRIBE_STREAM_ACTIVE);
    CHECK(transcribe_stream_get_state(&b) == TRANSCRIBE_STREAM_IDLE);

    CHECK(transcribe_stream_feed(&a, &pcm, 1, &upd) == TRANSCRIBE_OK);
    const int a_rev = transcribe_stream_revision(&a);
    CHECK(a_rev > 0);
    CHECK(transcribe_stream_revision(&b) == 0);

    // B begins and advances independently; A stays ACTIVE with its own
    // revision unchanged by B's progress.
    CHECK(transcribe_stream_begin(&b, nullptr, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_feed(&b, &pcm, 1, &upd) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_get_state(&a) == TRANSCRIBE_STREAM_ACTIVE);
    CHECK(transcribe_stream_revision(&a) == a_rev);

    // Finalizing A moves only A to FINISHED; B remains ACTIVE.
    CHECK(transcribe_stream_finalize(&a, &upd) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_get_state(&a) == TRANSCRIBE_STREAM_FINISHED);
    CHECK(transcribe_stream_get_state(&b) == TRANSCRIBE_STREAM_ACTIVE);

    // Aborting B's stream leaves A's FINISHED result intact.
    g_abort_flag = true;
    CHECK(transcribe_stream_feed(&b, &pcm, 1, &upd) == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_stream_get_state(&b) == TRANSCRIBE_STREAM_FAILED);
    CHECK(transcribe_was_aborted(&b) == true);
    CHECK(transcribe_stream_get_state(&a) == TRANSCRIBE_STREAM_FINISHED);
    CHECK(transcribe_was_aborted(&a) == false);
    g_abort_flag = false;
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
    test_begin_family_preflight_reject_preserves_snapshot();
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
    test_stream_text_on_finalize_policy();
    test_stream_auto_known_family_uses_family_stable_prefix();
    test_stream_auto_unknown_family_falls_back_to_generic_stable_prefix();
    test_stream_stable_prefix_known_family_uses_family_boundary();
    test_stream_family_token_prefix_maps_trimmed_leading_space();
    test_stream_text_stable_prefix_does_not_rewrite_committed();
    test_stream_text_clamps_raw_tentative_offset_after_shrink();
    test_stream_committed_row_hints_can_exceed_current_rows();
    test_stream_text_stable_prefix_floors_utf8_boundary();
    test_stream_text_stable_prefix_respects_agreement_n();
    test_stream_text_stable_prefix_intentionally_commits_partial_words();
    test_stream_update_old_prefix_tail_untouched();
    test_old_stream_params_ignores_new_tail_fields();
    test_begin_rejects_bad_commit_params_before_clear();
    test_feed_failure_transitions_to_failed();
    test_finalize_failure_transitions_to_failed();
    test_feed_abort_sets_was_aborted();
    test_two_sessions_independent_streams();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
