// transcribe-load-common.h - shared model-load scaffolding.
//
// INTERNAL. C++17. Consumed only from other .cpp files inside src/.
// Not part of the public include/transcribe.h ABI.
//
// Every per-family load() walks the same three steps after the GGUF
// catalog has been built:
//
//   1. Initialize backends (GPU → ACCEL → CPU) honoring
//      transcribe_model_params::use_gpu.
//   2. Allocate a backend buffer for every tensor in ctx_meta on the
//      preferred (first) backend.
//   3. Stream each tensor's bytes from the GGUF data section into
//      its backend slot via ggml_backend_tensor_set.
//
// Parakeet and Cohere had character-for-character copies of all
// three steps, modulo a "parakeet:"/"cohere:" log prefix. This
// header hoists them. Per-family `load()` still owns the model,
// params validation, weights catalog, fusion passes, and destructor
// wiring; what moves here is the uniform scaffolding in between.

#pragma once

#include "transcribe.h"

#include <string>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace transcribe::load_common {

// Discover and initialize backends via ggml's device registry.
// Order: [optional GPU/iGPU] → ACCEL (BLAS, etc.) → CPU.
//
//   force_cpu:   true skips the GPU/iGPU discovery pass. ACCEL
//                backends still run (they operate on host memory).
//   error_tag:   log prefix, e.g. "parakeet".
//   out:         appended in priority order. On success, out.back()
//                is the CPU backend. out[0] is the preferred (first
//                discovered) backend -- GPU if one was found and
//                force_cpu is false, otherwise the first ACCEL, else
//                CPU.
//
// Returns TRANSCRIBE_OK on success. Returns TRANSCRIBE_ERR_GGUF if
// CPU backend initialization fails (there is no fallback past CPU).
// GPU / ACCEL discovery failures are non-fatal and logged at their
// discovery site.
transcribe_status init_backends(bool                           force_cpu,
                                const char *                   error_tag,
                                std::vector<ggml_backend_t> &  out);

// Stream every tensor in `ctx_meta` from the GGUF data section on
// disk into its already-allocated backend buffer slot via
// ggml_backend_tensor_set. The backend buffer must already be
// allocated (typically via ggml_backend_alloc_ctx_tensors) before
// calling this.
//
//   path:      filesystem path to the GGUF file, reopened fresh via
//              std::ifstream. Stable for the life of the call.
//   gguf_data: the full gguf_context from gguf_init_from_file with
//              no_alloc=true. Used to resolve each tensor's data
//              offset.
//   ctx_meta:  the ggml_context that built_*_weights() populated
//              (tensors with backend-bound data slots).
//   error_tag: log prefix for diagnostics.
//
// Returns TRANSCRIBE_OK on success, TRANSCRIBE_ERR_GGUF on any
// I/O or tensor-not-found failure. On failure ctx_meta's tensors
// may be partially populated; the caller should discard the
// whole model state on error.
transcribe_status stream_tensor_data(const std::string &   path,
                                     const gguf_context *  gguf_data,
                                     ggml_context *        ctx_meta,
                                     const char *          error_tag);

// A single F16 → F32 promotion target: `dst_slot` is a pointer into a
// family weights struct (e.g. &block.conv_pw1_w) that currently holds
// the F16 source tensor and will be repointed to the F32 replacement.
struct ConvPwF32Slot {
    ggml_tensor ** dst_slot;
    ggml_tensor *  src;
};

// Dequantize a list of pointwise-conv weights from F16 to F32 when the
// primary backend is CPU. Used by conformer families (parakeet, cohere)
// whose encoder conv modules do a conv1d followed by a 1x1 pointwise
// conv; with F16 weights, the 1x1 path is bottlenecked by per-element
// dequantize inside matmul. Promoting the pointwise weights to F32 at
// load time eliminates that hot spot for the CPU backend only.
//
// Ownership on success: the caller receives a new ggml_context and
// backend buffer (written via *out_ctx / *out_buffer) that hold the F32
// replacements. Each slot's *dst_slot is repointed to the matching F32
// tensor. The original F16 tensors remain live in their original buffer
// (ggml's backend buffer model does not support freeing individual
// tensors); the cost is ~235 MB for cohere, much less for parakeet.
//
// Behavior:
//   - If `backends` is empty or the primary backend is not CPU: no-op,
//     returns TRANSCRIBE_OK; *out_ctx and *out_buffer are NOT touched.
//   - If `slots` is empty: no-op, same return semantics.
//   - If the F16 → float type trait is missing: logs a warning and
//     returns TRANSCRIBE_OK without modifying anything.
//
// error_tag: log prefix, e.g. "parakeet" or "cohere".
transcribe_status promote_conv_pw_f16_to_f32_on_cpu(
    const std::vector<ggml_backend_t> & backends,
    const std::vector<ConvPwF32Slot> &  slots,
    const char *                        error_tag,
    ggml_context **                     out_ctx,
    ggml_backend_buffer_t *             out_buffer);

} // namespace transcribe::load_common
