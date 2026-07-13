// run_dispatch_unit.cpp - dispatcher-level transcribe_run behavior tests.

#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Existing behavior: with no arch->run wired, transcribe_run clears the
// previous snapshot and returns NOT_IMPLEMENTED.
void test_no_run_hook_clears_and_not_implemented() {
    transcribe_session session;
    session.full_text   = "stale";
    session.has_result  = true;
    session.t_mel_us    = 1000;
    session.t_encode_us = 2000;
    session.t_decode_us = 3000;

    float                 pcm = 0.0f;
    transcribe_run_params params;
    transcribe_run_params_init(&params);
    const transcribe_status st = transcribe_run(&session, &pcm, 1, &params);

    CHECK(st == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(!session.has_result);
    CHECK(session.full_text.empty());

    transcribe_timings t;
    transcribe_timings_init(&t);
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

bool              g_run_called          = false;
transcribe_status g_run_validate_status = TRANSCRIBE_OK;

transcribe_status fake_run(transcribe_session *          session,
                           const float *                 pcm,
                           int                           n_samples,
                           const transcribe_run_params * params) {
    (void) pcm;
    (void) n_samples;
    (void) params;
    g_run_called        = true;
    // A successful run installs a fresh result.
    session->full_text  = "fresh result";
    session->has_result = true;
    return TRANSCRIBE_OK;
}

bool fake_accepts_run_kind(const transcribe_model * model, transcribe_ext_slot slot, uint32_t kind) {
    (void) model;
    return slot == TRANSCRIBE_EXT_SLOT_RUN && kind == kFakeRunKind;
}

transcribe_status fake_run_validate(const transcribe_session * ctx, const transcribe_run_params * params) {
    (void) ctx;
    (void) params;
    return g_run_validate_status;
}

const transcribe::Arch & run_validate_arch() {
    static const transcribe::Arch arch = {
        "fake-run",
        nullptr,  // load
        nullptr,  // init_context
        fake_run,
        nullptr,  // run_batch
        nullptr,  // stream_validate
        nullptr,  // stream_begin
        nullptr,  // stream_feed
        nullptr,  // stream_finalize
        nullptr,  // stream_reset
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
    session.model      = &model;
    session.full_text  = "previous result";
    session.has_result = true;

    transcribe_ext ext;
    ext.size = sizeof(transcribe_ext);  // passes the generic header check
    ext.kind = kFakeRunKind;            // accepted by the arch

    transcribe_run_params params;
    transcribe_run_params_init(&params);
    params.family = &ext;

    g_run_called          = false;
    g_run_validate_status = TRANSCRIBE_ERR_BAD_STRUCT_SIZE;

    float                   pcm = 0.0f;
    const transcribe_status st  = transcribe_run(&session, &pcm, 1, &params);

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
    session.model      = &model;
    session.full_text  = "previous result";
    session.has_result = true;

    transcribe_ext ext;
    ext.size = sizeof(transcribe_ext);
    ext.kind = kFakeRunKind;

    transcribe_run_params params;
    transcribe_run_params_init(&params);
    params.family = &ext;

    g_run_called          = false;
    g_run_validate_status = TRANSCRIBE_OK;

    float                   pcm = 0.0f;
    const transcribe_status st  = transcribe_run(&session, &pcm, 1, &params);

    CHECK(st == TRANSCRIBE_OK);
    CHECK(g_run_called == true);
    CHECK(session.has_result);
    CHECK(session.full_text == "fresh result");
}

// ---------------------------------------------------------------------------
// diarize enum validation: an out-of-range raw value is rejected with
// INVALID_ARG before the snapshot is cleared (mirrors the pnc/itn raw-enum
// gate in validate_run_params_common); a well-formed non-DEFAULT value
// passes validation and reaches the run hook (unsupported models only WARN).
// ---------------------------------------------------------------------------

void test_diarize_enum_validation() {
    transcribe_model model;
    model.arch = &run_validate_arch();

    transcribe_session session;
    session.model      = &model;
    session.full_text  = "previous result";
    session.has_result = true;

    transcribe_run_params params;
    transcribe_run_params_init(&params);
    const int bad_raw = 9999;
    std::memcpy(&params.diarize, &bad_raw, sizeof(bad_raw));

    g_run_called          = false;
    g_run_validate_status = TRANSCRIBE_OK;

    float             pcm = 0.0f;
    transcribe_status st  = transcribe_run(&session, &pcm, 1, &params);

    CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(g_run_called == false);
    CHECK(session.has_result);
    CHECK(session.full_text == "previous result");

    // Well-formed ON passes the gate (feature-less model WARNs, proceeds).
    transcribe_run_params_init(&params);
    params.diarize = TRANSCRIBE_DIARIZE_MODE_ON;
    st             = transcribe_run(&session, &pcm, 1, &params);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(g_run_called == true);
}

// ---------------------------------------------------------------------------
// Batch abort padding: a batch that aborts partway must still expose exactly
// n result slots. Utterances completed before the abort keep their real
// status; missing slots report TRANSCRIBE_ERR_ABORTED ("did not complete
// because the batch was aborted"). Exercised on the serial fallback path
// (run_batch == nullptr in run_validate_arch).
// ---------------------------------------------------------------------------

int g_abort_after = 0;  // number of poll_abort() calls allowed before firing
int g_abort_polls = 0;

bool fake_abort_cb(void * u) {
    (void) u;
    ++g_abort_polls;
    return g_abort_polls > g_abort_after;
}

void test_batch_abort_pads_missing_to_n() {
    transcribe_model model;
    model.arch = &run_validate_arch();  // run_batch == nullptr -> serial fallback

    transcribe_session session;
    session.model = &model;

    transcribe_run_params params;
    transcribe_run_params_init(&params);

    g_run_validate_status = TRANSCRIBE_OK;
    g_abort_polls         = 0;
    g_abort_after         = 1;  // utterance 0 runs; abort fires before utterance 1
    transcribe_set_abort_callback(&session, fake_abort_cb, nullptr);

    float         s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    const float * pcm[3]       = { &s0, &s1, &s2 };
    const int     n_samples[3] = { 1, 1, 1 };

    const transcribe_status st = transcribe_run_batch(&session, pcm, n_samples, 3, &params);

    // Whole-batch status is ABORTED, but every input still owns a slot.
    CHECK(st == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_batch_n_results(&session) == 3);
    CHECK(transcribe_batch_status(&session, 0) == TRANSCRIBE_OK);
    CHECK(transcribe_batch_status(&session, 1) == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_batch_status(&session, 2) == TRANSCRIBE_ERR_ABORTED);

    // Utterance 0 completed and aliases the legacy single-result accessors.
    CHECK(std::strcmp(transcribe_batch_full_text(&session, 0), "fresh result") == 0);
    CHECK(std::strcmp(transcribe_full_text(&session), "fresh result") == 0);

    // Synthesized slots carry no transcript.
    CHECK(std::strcmp(transcribe_batch_full_text(&session, 2), "") == 0);
}

// ---------------------------------------------------------------------------
// Fast-path counterpart: a family run_batch hook that completes one utterance
// and then returns TRANSCRIBE_ERR_ABORTED. The dispatcher must pad missing
// slots to n, just as it does for the serial fallback — covering the run_batch
// != nullptr branch in transcribe_run_batch.
// ---------------------------------------------------------------------------

transcribe_status fake_run_batch_abort(transcribe_session *          session,
                                       const float * const *         pcm,
                                       const int *                   n_samples,
                                       int                           n,
                                       const transcribe_run_params * params) {
    (void) pcm;
    (void) n_samples;
    (void) n;
    (void) params;
    // Complete utterance 0, then abort before producing the rest. The hook
    // retains only what it finished; the dispatcher pads missing slots.
    transcribe_session::ResultSet rs;
    rs.full_text  = "batch result 0";
    rs.has_result = true;
    rs.status     = TRANSCRIBE_OK;
    session->batch_results.push_back(std::move(rs));
    return TRANSCRIBE_ERR_ABORTED;
}

const transcribe::Arch & run_batch_abort_arch() {
    static const transcribe::Arch arch = {
        "fake-run-batch",
        nullptr,               // load
        nullptr,               // init_context
        fake_run,              // run (required; dispatcher gates on it)
        fake_run_batch_abort,  // run_batch (fast path)
        nullptr,               // stream_validate
        nullptr,               // stream_begin
        nullptr,               // stream_feed
        nullptr,               // stream_finalize
        nullptr,               // stream_reset
        nullptr,               // accepts_run_kind
        nullptr,               // run_validate
    };
    return arch;
}

void test_batch_fastpath_abort_pads_missing_to_n() {
    transcribe_model model;
    model.arch = &run_batch_abort_arch();

    transcribe_session session;
    session.model = &model;

    transcribe_run_params params;
    transcribe_run_params_init(&params);

    float         s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    const float * pcm[3]       = { &s0, &s1, &s2 };
    const int     n_samples[3] = { 1, 1, 1 };

    const transcribe_status st = transcribe_run_batch(&session, pcm, n_samples, 3, &params);

    // Same invariant as the serial fallback, via the fast-path branch.
    CHECK(st == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_batch_n_results(&session) == 3);
    CHECK(transcribe_batch_status(&session, 0) == TRANSCRIBE_OK);
    CHECK(transcribe_batch_status(&session, 1) == TRANSCRIBE_ERR_ABORTED);
    CHECK(transcribe_batch_status(&session, 2) == TRANSCRIBE_ERR_ABORTED);

    // Hook-completed utterance 0 aliases the legacy single-result accessors.
    CHECK(std::strcmp(transcribe_batch_full_text(&session, 0), "batch result 0") == 0);
    CHECK(std::strcmp(transcribe_full_text(&session), "batch result 0") == 0);
    CHECK(std::strcmp(transcribe_batch_full_text(&session, 2), "") == 0);
}

}  // namespace

int main() {
    test_no_run_hook_clears_and_not_implemented();
    test_run_validate_failure_preserves_snapshot();
    test_run_validate_success_clears_and_runs();
    test_diarize_enum_validation();
    test_batch_abort_pads_missing_to_n();
    test_batch_fastpath_abort_pads_missing_to_n();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
