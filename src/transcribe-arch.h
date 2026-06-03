// transcribe-arch.h - per-family architecture trait + registry.
//
// This header is INTERNAL. The public C ABI in include/transcribe.h knows
// nothing about Arch; it dispatches through find_arch() inside
// transcribe_model_load_file and the rest of the lifecycle entry points.
//
// Each model family (parakeet, eventually whisper, sense-voice, etc.)
// provides exactly one transcribe::Arch instance whose function pointers
// implement that family's load / init_context / run behavior. The
// registry is an explicit array (no static initializers, no dlopen, no
// global registration tricks) so the set of supported architectures is
// grep-able from one file.
//
// Adding a new family is a two-line edit in transcribe-arch.cpp plus a
// new src/arch/<family>/model.cpp that defines the Arch instance.
//
// The Arch trait contains only the things that genuinely vary per
// family. Lifecycle teardown (free_model / free_context) is handled
// by virtual destructors on transcribe_model / transcribe_session,
// and the introspection accessors (capabilities, backend_string,
// arch_string, variant_string) read directly from the base struct.

#pragma once

#include "transcribe.h"

namespace transcribe {

class Loader;

// Per-family trait. Function pointers may be null if the corresponding
// entry point is not yet implemented for that family; the central
// dispatch in transcribe.cpp converts null entries into
// TRANSCRIBE_ERR_NOT_IMPLEMENTED so a partially-wired family does not
// crash mid-run.
//
// Ownership conventions:
//
//   load: the Loader has already opened the GGUF and read the
//   architecture/variant strings. The handler may call
//   loader.release_gguf() to take ownership of the gguf_context, in which
//   case it is responsible for freeing it (typically in the per-family
//   model's destructor). If the handler does not call release_gguf(),
//   the Loader's destructor frees the context on stack unwinding.
//
//   On success, *out_model points at a heap-allocated derived
//   transcribe_model. On failure, *out_model is left null and the
//   handler returns a non-OK status.
//
//   init_context: same convention. On success *out_ctx points at a
//   heap-allocated derived transcribe_session. The dispatcher has
//   already validated that model and params are non-null.
//
//   Cleanup is NOT in the trait: the central dispatcher's
//   transcribe_model_free / transcribe_session_free use `delete` against
//   the base, and the virtual destructors do the right thing.
struct Arch {
    // The string compared against general.architecture in the GGUF KV.
    // Must be a NUL-terminated string with static lifetime.
    const char * name;

    transcribe_status (*load)(
        Loader &                               loader,
        const transcribe_model_load_params *        params,
        struct transcribe_model **             out_model);

    transcribe_status (*init_context)(
        struct transcribe_model *              model,
        const transcribe_session_params *      params,
        struct transcribe_session **           out_ctx);

    transcribe_status (*run)(
        struct transcribe_session *            ctx,
        const float *                          pcm,
        int                                    n_samples,
        const transcribe_run_params *              params);

    // Optional offline batch fast path. When non-null, transcribe_run_batch
    // delegates here to process all `n` utterances in a single dispatch
    // (e.g. a batched encoder graph) for device-level throughput. When
    // null, the central dispatcher falls back to calling run() once per
    // utterance and snapshotting each result, so EVERY family supports the
    // batch API for correctness regardless of whether it wires this hook;
    // a family opts into the fast path by implementing it.
    //
    // Contract:
    //   - The dispatcher has already validated the shared run_params
    //     (struct size, _RUN-slot ext shape/kind, enum range, timestamp
    //     ceiling, language, TRANSLATE support, run_validate) ONCE and
    //     cleared both the scratch result slot and ctx->batch_results.
    //   - pcm[i] / n_samples[i] are the i-th utterance (i in [0, n)).
    //     16 kHz mono float32, same as run(). The hook is responsible for
    //     per-utterance input validation (pcm[i] == null or
    //     n_samples[i] <= 0 is a per-utterance failure, recorded as a
    //     non-OK ResultSet::status with has_result == false, NOT a
    //     whole-batch error).
    //   - On an OK return the hook MUST push exactly `n` entries onto
    //     ctx->batch_results, one per utterance in order, and SHOULD leave
    //     the scratch slot mirroring batch_results[0] on return (the
    //     dispatcher also enforces this). Poll ctx->poll_abort() between
    //     utterances / decode steps; on abort, push the utterances retained
    //     as completed results and return TRANSCRIBE_ERR_ABORTED — the hook
    //     need NOT pad missing slots. The dispatcher pads batch_results back
    //     up to `n` with TRANSCRIBE_ERR_ABORTED ResultSets (status meaning
    //     "did not complete because the batch was aborted"), so after an OK
    //     or ABORTED return batch_results.size() == n.
    //   - The function return value is the whole-batch status: OK when the
    //     dispatch ran (per-utterance failures live in each ResultSet),
    //     non-OK only for whole-batch faults (OOM, abort, backend error).
    //     For a whole-batch fault other than abort (OOM, backend error) the
    //     result count is unspecified; the non-OK return is the signal.
    transcribe_status (*run_batch)(
        struct transcribe_session *            ctx,
        const float * const *                  pcm,
        const int *                            n_samples,
        int                                    n,
        const transcribe_run_params *              params);

    // Optional streaming hooks. stream_begin / stream_feed /
    // stream_finalize form a required triple: a family that wants to
    // support streaming MUST wire all three. The dispatcher checks
    // the triple at begin time and returns NOT_IMPLEMENTED if any of
    // them is NULL (a partially-wired family must never let a caller
    // enter ACTIVE and then get stuck on the next call). stream_reset
    // is optional — when NULL, transcribe_stream_reset still wipes
    // dispatcher-owned state and runs the clear_result path, which
    // is sufficient when a family doesn't need to release per-
    // utterance buffers explicitly.
    //
    // Hook responsibilities:
    //
    //   stream_validate (optional): pre-flight validation of family-
    //     specific parameters and stream_params->family extension values.
    //     Called BEFORE the dispatcher clears the previous result
    //     snapshot and BEFORE the lifecycle moves out of its current
    //     state. Must be pure: zero mutation of ctx or model. On non-OK
    //     return the dispatcher returns the status to the caller with
    //     the previous result snapshot fully preserved and the stream
    //     lifecycle untouched (no FAILED transition). Families that
    //     accept extension structs SHOULD validate enum-from-menu and
    //     value-shape rules here so a caller typo does not destroy the
    //     prior utterance's transcript.
    //
    //   stream_begin: install per-utterance state on the derived
    //     context. The central dispatcher has already run
    //     stream_validate (if any), cleared the result snapshot, and
    //     set ctx->stream_state to ACTIVE. On non-OK return the
    //     dispatcher transitions to FAILED and preserves the status in
    //     stream_last_status.  Families MAY repeat cheap validation
    //     here as defense in depth, but the canonical place to reject
    //     bad caller input is stream_validate.
    //
    //   stream_feed: consume the PCM (n_samples may be 0) and update
    //     the result vectors / committed counts / audio cursors on
    //     ctx in place. When update is non-null, fill it with the
    //     family's per-call change metadata; the dispatcher has
    //     already zero-initialized it. Poll ctx->poll_abort() at
    //     chunk and decode-step boundaries; if it fires, return
    //     TRANSCRIBE_ERR_ABORTED (the dispatcher will transition to
    //     FAILED and record the status, and transcribe_was_aborted
    //     will distinguish abort from other failures).
    //
    //   stream_finalize: flush buffered audio, satisfy lookahead,
    //     emit remaining text, mark all output as committed. The
    //     dispatcher transitions to FINISHED on TRANSCRIBE_OK and
    //     FAILED otherwise.
    //
    //   stream_reset: release the family's per-utterance state and
    //     buffered audio contents (keeping allocated buffers for
    //     reuse). The dispatcher clears the result snapshot and
    //     forces stream_state back to IDLE around this call.
    transcribe_status (*stream_validate)(
        const struct transcribe_session *      ctx,
        const transcribe_run_params *          run_params,
        const transcribe_stream_params *       stream_params);

    transcribe_status (*stream_begin)(
        struct transcribe_session *            ctx,
        const transcribe_run_params *              run_params,
        const transcribe_stream_params *       stream_params);

    transcribe_status (*stream_feed)(
        struct transcribe_session *            ctx,
        const float *                          pcm,
        int                                    n_samples,
        transcribe_stream_update *             update);

    transcribe_status (*stream_finalize)(
        struct transcribe_session *            ctx,
        transcribe_stream_update *             update);

    void (*stream_reset)(
        struct transcribe_session *            ctx);

    // Optional kind+slot probe. Returns true when the loaded model
    // variant accepts the named extension kind on the given slot
    // (transcribe_run_params::family for _RUN, transcribe_stream_params::family
    // for _STREAM). NULL means "no family extension kinds accepted in
    // any slot" — the dispatcher's transcribe_model_accepts_ext_kind()
    // returns false in that case.
    //
    // Acceptance is per-loaded-variant AND per-slot:
    //   - parakeet returns true for (_STREAM, PARAKEET_STREAM) on
    //     cache-aware variants and for (_STREAM, PARAKEET_BUFFERED_STREAM)
    //     on chunked-attention variants — never both on the same loaded
    //     model — and always false for the _RUN slot.
    //   - moonshine_streaming returns true for
    //     (_STREAM, MOONSHINE_STREAMING_STREAM) only.
    //   - whisper returns true for (_RUN, WHISPER_RUN) only.
    // A family that genuinely has no extension surface for a slot must
    // return false rather than NULL'ing this hook; NULL is reserved for
    // "no extension surface at all."
    bool (*accepts_ext_kind)(
        const struct transcribe_model *        model,
        transcribe_ext_slot                    slot,
        uint32_t                               kind);

    // Optional pre-clear validation for the one-shot transcribe_run path,
    // the _RUN-slot analogue of stream_validate. The dispatcher calls it
    // as the LAST gate before clear_result wipes the previous snapshot,
    // and AFTER its own pre-clear checks: run_params struct size, the
    // _RUN-slot ext header size + kind acceptance, and the run-param
    // checks (enum range, timestamp ceiling, language, TRANSLATE support).
    // Must be pure: zero mutation of ctx or model. On non-OK return the
    // dispatcher returns the status with the previous result snapshot
    // fully preserved, so a rejected run ext cannot destroy the prior
    // transcript.
    //
    // Scope of the snapshot-preservation guarantee: a family SHOULD put
    // its run-ext value validation here (the way parakeet's stream_validate
    // vets its (L,C,R) menu) so a caller typo is caught pre-clear. The
    // guarantee only extends to what this hook actually checks. A family
    // that defers some value checks to its run() handler trades them out
    // of the guarantee: a correctly-shaped but semantically-malformed ext
    // can still be rejected post-clear. Whisper validates only the per-kind
    // minimum ext size here (transcribe_ext_check against the full ext
    // struct) and intentionally leaves its prompt-semantics checks in run()
    // — an accepted gap on the one-shot path; see docs/follow-ups.md.
    //
    // Placed last in the struct so families that do not accept a _RUN ext
    // can leave it value-initialized to NULL; the dispatcher skips a NULL
    // hook and the generic header-size + kind checks remain in force.
    transcribe_status (*run_validate)(
        const struct transcribe_session *      ctx,
        const transcribe_run_params *          params);
};

// Look up an architecture by name. Returns nullptr if no registered
// family matches. The lookup is O(n) over the registry; n is small
// (single-digit) and lookups happen once per load.
const Arch * find_arch(const char * name);

} // namespace transcribe
