// stream_capability_unit.cpp - Phase 3: capability gating against a
// real loaded model.
//
// The model-less context tests in stream_dispatch_unit.cpp exercise
// the streaming dispatcher's state machine and accessor sentinels.
// This test goes one level deeper and pins the model-side contract:
//
//   1. The default-capability path (no streaming KV in the GGUF, family
//      default supports_streaming=false): caps.supports_streaming reads
//      false, the new streaming_lookahead_ms / streaming_chunk_ms
//      hint fields are zero-initialized, and transcribe_stream_begin
//      returns NOT_IMPLEMENTED on the capability gate.
//
//   2. The KV-override path against a NON-STREAMING VARIANT (capability
//      KV says streaming=true, but the variant's encoder hparams are
//      not ChunkedLimited). Caps.supports_streaming reads true (the KV
//      path works), the dispatcher transitions into ACTIVE and calls
//      the family's stream_begin hook, and the hook itself rejects
//      with NOT_IMPLEMENTED because the parakeet family only supports
//      streaming for cache-aware ChunkedLimited variants (today:
//      nemotron-speech-streaming-en-0.6b). The dispatcher then
//      transitions the stream to FAILED and preserves the status in
//      stream_last_status. This guards against the failure mode where
//      a future converter flips the capability flag on a variant whose
//      compute graph cannot actually stream.
//
// Both cases share the parakeet family. The two GGUF fixtures differ
// only in the capability KV; everything else (tokenizer, hparams,
// weight catalog) is identical so the test isolates capability
// behavior cleanly.

#include "transcribe.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#  error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

const std::string g_fixtures_dir = TRANSCRIBE_TEST_FIXTURES_DIR;

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

// Wraps the load + context-init dance so each test case stays tight.
// On failure prints a diagnostic, increments g_failures, and returns
// false; the caller then bails so it doesn't deref nullptrs.
bool load_and_init(const char *                  fixture_name,
                   struct transcribe_model **    out_model,
                   struct transcribe_context **  out_ctx)
{
    *out_model = nullptr;
    *out_ctx   = nullptr;

    const std::string p = g_fixtures_dir + "/" + fixture_name;
    if (!file_exists(p)) {
        std::fprintf(stderr,
                     "stream_capability_unit: fixture not found: %s\n"
                     "  regenerate with: cmake --build build --target fixtures\n",
                     p.c_str());
        ++g_failures;
        return false;
    }

    transcribe_model_params mp = transcribe_model_default_params();
    const transcribe_status load_st =
        transcribe_model_load_file(p.c_str(), &mp, out_model);
    if (load_st != TRANSCRIBE_OK || *out_model == nullptr) {
        std::fprintf(stderr,
                     "FAIL load %s: status=%s, model=%p\n",
                     fixture_name, transcribe_status_string(load_st),
                     (void *)*out_model);
        ++g_failures;
        return false;
    }

    transcribe_context_params cp = transcribe_context_default_params();
    const transcribe_status init_st =
        transcribe_context_init(*out_model, &cp, out_ctx);
    if (init_st != TRANSCRIBE_OK || *out_ctx == nullptr) {
        std::fprintf(stderr,
                     "FAIL context_init %s: status=%s, ctx=%p\n",
                     fixture_name, transcribe_status_string(init_st),
                     (void *)*out_ctx);
        transcribe_model_free(*out_model);
        *out_model = nullptr;
        ++g_failures;
        return false;
    }
    return true;
}

// Capability-default fixture: supports_streaming reads false, the
// streaming timing hints are zero (the "unsupported or unknown"
// sentinel), and begin reports NOT_IMPLEMENTED.
void test_supports_streaming_false() {
    struct transcribe_model *   model = nullptr;
    struct transcribe_context * ctx   = nullptr;
    if (!load_and_init("tokenizer_minimal.gguf", &model, &ctx)) return;

    transcribe_capabilities caps_buf = TRANSCRIBE_CAPABILITIES_INIT;
    const bool caps_ok =
        transcribe_model_get_capabilities(model, &caps_buf) == TRANSCRIBE_OK;
    const transcribe_capabilities * caps = caps_ok ? &caps_buf : nullptr;
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK(caps->supports_streaming     == false);
        // New hint fields default to zero; the parakeet family does
        // not advertise any specific lookahead or chunk size today.
        CHECK(caps->streaming_lookahead_ms == 0);
        CHECK(caps->streaming_chunk_ms     == 0);
    }

    transcribe_stream_params  sp = transcribe_stream_default_params();
    transcribe_params         rp = transcribe_default_params();
    const transcribe_status   st = transcribe_stream_begin(ctx, &rp, &sp);

    CHECK(st == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    // Pre-hook rejection: state unchanged from IDLE, last_status not
    // disturbed (capability gate fires before the lifecycle transition).
    CHECK(transcribe_stream_get_state(ctx)    == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_stream_last_status(ctx)  == TRANSCRIBE_OK);
    CHECK(transcribe_stream_revision(ctx)     == 0);

    transcribe_context_free(ctx);
    transcribe_model_free(model);
}

// Capability-KV-override fixture: stt.capability.streaming=true flips
// the capability flag on, and the parakeet family now wires the
// stream_begin/feed/finalize hooks. The dispatcher passes the cap
// gate and the hook-triple gate, transitions to ACTIVE, then calls
// stream_begin — which itself rejects with NOT_IMPLEMENTED because
// the toy fixture is not a ChunkedLimited variant. The dispatcher
// records the failing status and transitions the stream to FAILED.
// Pins the "capability KV alone is not enough; the family's compute
// graph must actually support streaming" guard.
void test_supports_streaming_true_variant_offline() {
    struct transcribe_model *   model = nullptr;
    struct transcribe_context * ctx   = nullptr;
    if (!load_and_init("tokenizer_minimal_streaming.gguf", &model, &ctx)) return;

    transcribe_capabilities caps_buf = TRANSCRIBE_CAPABILITIES_INIT;
    const bool caps_ok =
        transcribe_model_get_capabilities(model, &caps_buf) == TRANSCRIBE_OK;
    const transcribe_capabilities * caps = caps_ok ? &caps_buf : nullptr;
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        // KV path works: the loader read stt.capability.streaming=true
        // and flipped the family default.
        CHECK(caps->supports_streaming     == true);
        // Hint fields still zero — they live in apply_family_invariants,
        // not in the capability KV reader, and parakeet leaves them at 0.
        CHECK(caps->streaming_lookahead_ms == 0);
        CHECK(caps->streaming_chunk_ms     == 0);
    }

    transcribe_stream_params  sp = transcribe_stream_default_params();
    transcribe_params         rp = transcribe_default_params();
    const transcribe_status   st = transcribe_stream_begin(ctx, &rp, &sp);

    // Family-level rejection: caps says yes, the dispatcher's gates
    // pass, but the parakeet stream_begin hook refuses because the
    // toy fixture's encoder is Regular (full attention), not
    // ChunkedLimited. Post-hook rejection → state is FAILED and the
    // status is preserved on the context.
    CHECK(st == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_stream_get_state(ctx)   == TRANSCRIBE_STREAM_FAILED);
    CHECK(transcribe_stream_last_status(ctx) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);

    transcribe_context_free(ctx);
    transcribe_model_free(model);
}

// Sanity: after begin rejects with NOT_IMPLEMENTED, transcribe_run
// still works against the same context (the streaming rejection
// must not have poisoned context state). We don't run actual
// inference here — the parakeet minimal fixture lacks the bits to
// produce real output — but transcribe_run reaching the family run()
// handler (which would proceed to do real work) is enough to prove
// the streaming reject didn't leave the context in a poisoned state.
//
// We check this by calling run with a one-sample buffer and accepting
// either OK or any of the legitimate run-time errors (the family
// might reject because the fixture isn't a real model). What we
// reject is the streaming-specific INVALID_ARG-while-active path
// — that would mean begin's failure incorrectly left state at ACTIVE.
void test_run_after_failed_begin_does_not_get_stuck() {
    struct transcribe_model *   model = nullptr;
    struct transcribe_context * ctx   = nullptr;
    if (!load_and_init("tokenizer_minimal.gguf", &model, &ctx)) return;

    transcribe_stream_params  sp = transcribe_stream_default_params();
    transcribe_params         rp = transcribe_default_params();
    CHECK(transcribe_stream_begin(ctx, &rp, &sp) ==
          TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_stream_get_state(ctx) == TRANSCRIBE_STREAM_IDLE);

    // transcribe_run from IDLE: must not return the ACTIVE-rejection
    // INVALID_ARG path. Whatever the family does next is fine — we
    // only care that the streaming bookkeeping didn't strand the
    // context.
    const float pcm[16] = { 0.0f };
    const transcribe_status run_st = transcribe_run(ctx, pcm, 16, &rp);
    // INVALID_ARG specifically from the ACTIVE-stream rejection
    // would be a regression. The dispatcher's other INVALID_ARG paths
    // (NULL args, bad enum) are not reachable here since we pass
    // well-formed params.
    if (run_st == TRANSCRIBE_ERR_INVALID_ARG) {
        std::fprintf(stderr,
                     "FAIL run-after-failed-begin: got INVALID_ARG, "
                     "stream state may have been stranded\n");
        ++g_failures;
    }
    // State should still be IDLE — transcribe_run forces it on entry.
    CHECK(transcribe_stream_get_state(ctx) == TRANSCRIBE_STREAM_IDLE);

    transcribe_context_free(ctx);
    transcribe_model_free(model);
}

} // namespace

int main() {
    test_supports_streaming_false();
    test_supports_streaming_true_variant_offline();
    test_run_after_failed_begin_does_not_get_stuck();

    if (g_failures > 0) {
        std::fprintf(stderr, "stream_capability_unit: %d failures\n",
                     g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "stream_capability_unit: ok\n");
    return EXIT_SUCCESS;
}
