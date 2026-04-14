// transcribe-weights-util.h - shared GGUF tensor validation helpers.
//
// INTERNAL. C++17. Consumed only from other .cpp files inside src/.
// Not part of the public include/transcribe.h ABI.
//
// Every per-family weights.cpp needs to resolve tensor names from the
// loaded GGUF's ctx_meta and validate the type + shape matches what
// the family's encoder / decoder graph expects. The pattern is
// identical across families (parakeet and cohere both duplicated it
// verbatim with only the log-tag differing), so it lives here.
//
// Usage: the per-family weights.cpp defines a short GET_* macro
// wrapper that captures the family tag string:
//
//     #define PARAKEET_TAG "parakeet"
//     #define GET_F32(slot, name, ...) do { \
//         ggml_tensor * _t = transcribe::weights::find_tensor( \
//             gguf, ctx_meta, (name), {GGML_TYPE_F32}, {__VA_ARGS__}, \
//             PARAKEET_TAG); \
//         if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
//         (slot) = _t; \
//     } while (0)
//
// The GET_* macros themselves stay per-family because the allowed
// type lists encode a family-specific quantization policy (parakeet
// ships F32/F16/Q8_0/Q4_K/Q5_K/Q6_K, cohere additionally accepts
// BF16/Q4_0/Q4_1/Q5_0/Q5_1). Extracting the allowlists into shared
// code would paper over that policy boundary.

#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

struct gguf_context;

namespace transcribe::weights {

// Resolve `name` in `ctx_meta` and verify that:
//   - the tensor exists,
//   - its type is in `allowed_types`,
//   - ne[0..n-1] matches `expected_ne` (where n = expected_ne.size()),
//   - ne[n..GGML_MAX_DIMS-1] are all 1 (no unexpected rank).
//
// On success returns the borrowed tensor pointer. On failure logs a
// human-readable diagnostic to stderr (prefixed with `error_tag`) and
// returns nullptr. `gguf` is currently unused by the implementation
// but kept in the signature so the helper can cross-reference
// gguf_find_tensor in the future without touching every call site.
ggml_tensor * find_tensor(const gguf_context *              gguf,
                          ggml_context *                    ctx_meta,
                          const char *                      name,
                          std::initializer_list<ggml_type>  allowed_types,
                          std::initializer_list<int64_t>    expected_ne,
                          const char *                      error_tag);

// Format a per-layer tensor name. The caller passes a printf fmt
// containing exactly one %d; the layer index is substituted. Returns
// a pointer to a thread-local static buffer — the next call within the
// same thread invalidates the previous pointer.
//
// This previously lived duplicated in every per-family weights.cpp.
// Keeping it as const char * avoids touching every GET_* macro call
// site (find_tensor takes const char *). The 128-byte buffer is ample
// for any realistic layer name (longest is ~50 chars + 3-digit index).
inline const char * lname(const char * fmt, int layer_idx) {
    thread_local char buf[128];
    std::snprintf(buf, sizeof(buf), fmt, layer_idx);
    return buf;
}

} // namespace transcribe::weights
