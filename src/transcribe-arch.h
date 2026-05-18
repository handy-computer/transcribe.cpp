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
// by virtual destructors on transcribe_model / transcribe_context,
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
//   heap-allocated derived transcribe_context. The dispatcher has
//   already validated that model and params are non-null.
//
//   Cleanup is NOT in the trait: the central dispatcher's
//   transcribe_model_free / transcribe_context_free use `delete` against
//   the base, and the virtual destructors do the right thing.
struct Arch {
    // The string compared against general.architecture in the GGUF KV.
    // Must be a NUL-terminated string with static lifetime.
    const char * name;

    transcribe_status (*load)(
        Loader &                               loader,
        const transcribe_model_params *        params,
        struct transcribe_model **             out_model);

    transcribe_status (*init_context)(
        struct transcribe_model *              model,
        const transcribe_context_params *      params,
        struct transcribe_context **           out_ctx);

    transcribe_status (*run)(
        struct transcribe_context *            ctx,
        const float *                          pcm,
        int                                    n_samples,
        const transcribe_params *              params);

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
    //   stream_begin: install per-utterance state on the derived
    //     context. The central dispatcher has already validated
    //     params, cleared the result snapshot, and set ctx->stream_state
    //     to ACTIVE. On non-OK return the dispatcher transitions to
    //     FAILED and preserves the status in stream_last_status.
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
    transcribe_status (*stream_begin)(
        struct transcribe_context *            ctx,
        const transcribe_params *              run_params,
        const transcribe_stream_params *       stream_params);

    transcribe_status (*stream_feed)(
        struct transcribe_context *            ctx,
        const float *                          pcm,
        int                                    n_samples,
        transcribe_stream_update *             update);

    transcribe_status (*stream_finalize)(
        struct transcribe_context *            ctx,
        transcribe_stream_update *             update);

    void (*stream_reset)(
        struct transcribe_context *            ctx);

    // Optional kind probe. Returns true when the loaded model variant
    // accepts the named extension kind on transcribe_stream_params::family
    // (or, after Phase 2, transcribe_params::family). NULL means "no
    // family extension kinds accepted" — the dispatcher's
    // transcribe_model_accepts_ext_kind() returns false in that case.
    //
    // Acceptance is per-loaded-variant, not per-family: e.g. parakeet
    // returns true for PARAKEET_STREAM on cache-aware variants and for
    // PARAKEET_BUFFERED_STREAM on chunked-attention variants, never both
    // on the same loaded model.
    bool (*accepts_ext_kind)(
        const struct transcribe_model *        model,
        uint32_t                               kind);
};

// Look up an architecture by name. Returns nullptr if no registered
// family matches. The lookup is O(n) over the registry; n is small
// (single-digit) and lookups happen once per load.
const Arch * find_arch(const char * name);

} // namespace transcribe
