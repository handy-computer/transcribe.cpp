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
};

// Look up an architecture by name. Returns nullptr if no registered
// family matches. The lookup is O(n) over the registry; n is small
// (single-digit) and lookups happen once per load.
const Arch * find_arch(const char * name);

} // namespace transcribe
