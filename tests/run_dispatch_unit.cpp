// run_dispatch_unit.cpp - dispatcher-level transcribe_run behavior tests.

#include "transcribe-session.h"
#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// Existing behavior: with no arch->run wired, transcribe_run clears the
// previous snapshot and returns NOT_IMPLEMENTED.
void test_no_run_hook_clears_and_not_implemented() {
    transcribe_session session;
    session.full_text = "stale";
    session.has_result = true;
    session.t_mel_us = 1000;
    session.t_encode_us = 2000;
    session.t_decode_us = 3000;

    float pcm = 0.0f;
    transcribe_run_params params; transcribe_run_params_init(&params);
    const transcribe_status st = transcribe_run(&session, &pcm, 1, &params);

    CHECK(st == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(!session.has_result);
    CHECK(session.full_text.empty());

    transcribe_timings t; transcribe_timings_init(&t);
    CHECK(transcribe_get_timings(&session, &t) == TRANSCRIBE_OK);
    CHECK(t.mel_ms == 0.0f);
    CHECK(t.encode_ms == 0.0f);
    CHECK(t.decode_ms == 0.0f);
}

// ---------------------------------------------------------------------------
// run_validate pre-clear hook: the _RUN-slot analogue of stream_validate.
// A family-rejected run extension must NOT wipe the previous result
// snapshot, mirroring transcribe_stream_begin. (Regression test for the
// undersized-run-ext-clears-snapshot bug.)
// ---------------------------------------------------------------------------

constexpr uint32_t kFakeRunKind = 0xF00D;

bool g_run_called = false;
transcribe_status g_run_validate_status = TRANSCRIBE_OK;

transcribe_status fake_run(
    transcribe_session *        session,
    const float *               pcm,
    int                         n_samples,
    const transcribe_run_params * params)
{
    (void)pcm;
    (void)n_samples;
    (void)params;
    g_run_called = true;
    // A successful run installs a fresh result.
    session->full_text = "fresh result";
    session->has_result = true;
    return TRANSCRIBE_OK;
}

bool fake_accepts_run_kind(
    const transcribe_model * model,
    transcribe_ext_slot      slot,
    uint32_t                 kind)
{
    (void)model;
    return slot == TRANSCRIBE_EXT_SLOT_RUN && kind == kFakeRunKind;
}

transcribe_status fake_run_validate(
    const transcribe_session *   ctx,
    const transcribe_run_params * params)
{
    (void)ctx;
    (void)params;
    return g_run_validate_status;
}

const transcribe::Arch & run_validate_arch() {
    static const transcribe::Arch arch = {
        "fake-run",
        nullptr,                 // load
        nullptr,                 // init_context
        fake_run,
        nullptr,                 // stream_validate
        nullptr,                 // stream_begin
        nullptr,                 // stream_feed
        nullptr,                 // stream_finalize
        nullptr,                 // stream_reset
        fake_accepts_run_kind,
        fake_run_validate,
    };
    return arch;
}

// A run ext whose kind is accepted and whose header size is valid, but
// which run_validate rejects (modelling a too-small typed ext struct).
void test_run_validate_failure_preserves_snapshot() {
    transcribe_model model;
    model.arch = &run_validate_arch();

    transcribe_session session;
    session.model = &model;
    session.full_text = "previous result";
    session.has_result = true;

    transcribe_ext ext;
    ext.size = sizeof(transcribe_ext);   // passes the generic header check
    ext.kind = kFakeRunKind;             // accepted by the arch

    transcribe_run_params params; transcribe_run_params_init(&params);
    params.family = &ext;

    g_run_called = false;
    g_run_validate_status = TRANSCRIBE_ERR_BAD_STRUCT_SIZE;

    float pcm = 0.0f;
    const transcribe_status st = transcribe_run(&session, &pcm, 1, &params);

    // Family preflight rejected the call BEFORE the snapshot was cleared
    // and BEFORE the run hook ran. The prior transcript survives intact.
    CHECK(st == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(g_run_called == false);
    CHECK(session.has_result);
    CHECK(session.full_text == "previous result");
}

// Control: when run_validate passes, the dispatcher clears the snapshot and
// hands off to the run hook (which installs the fresh result). This proves
// the pre-clear gate protects the snapshot ONLY on a validation failure.
void test_run_validate_success_clears_and_runs() {
    transcribe_model model;
    model.arch = &run_validate_arch();

    transcribe_session session;
    session.model = &model;
    session.full_text = "previous result";
    session.has_result = true;

    transcribe_ext ext;
    ext.size = sizeof(transcribe_ext);
    ext.kind = kFakeRunKind;

    transcribe_run_params params; transcribe_run_params_init(&params);
    params.family = &ext;

    g_run_called = false;
    g_run_validate_status = TRANSCRIBE_OK;

    float pcm = 0.0f;
    const transcribe_status st = transcribe_run(&session, &pcm, 1, &params);

    CHECK(st == TRANSCRIBE_OK);
    CHECK(g_run_called == true);
    CHECK(session.has_result);
    CHECK(session.full_text == "fresh result");
}

} // namespace

int main() {
    test_no_run_hook_clears_and_not_implemented();
    test_run_validate_failure_preserves_snapshot();
    test_run_validate_success_clears_and_runs();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
