// transcribe-load-common.h - shared model-load scaffolding.
//
// INTERNAL, C++17. Not part of the public ABI.
//
// Every per-family load() walks the same two steps after building the
// GGUF catalog:
//
//   1. Resolve a BackendPlan from transcribe_model_load_params::backend
//      (init_backends).
//   2. Give every tensor in ctx_meta storage on the plan's primary backend
//      and fill it from the GGUF data section (load_weights). Where the
//      device can use host memory directly the tensors point into a
//      read-only memory map of the file; otherwise their bytes are copied
//      into a backend buffer.
//
// These functions hoist that scaffolding; per-family load() still owns the
// model, params validation, weights catalog, fusion, and destructor wiring.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-mmap.h"
#include "transcribe.h"

#include <string>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
struct ggml_backend_device;
typedef struct ggml_backend_device * ggml_backend_dev_t;
struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace transcribe::load_common {

// True when `dev` is a Metal device with no simdgroup matrix multiply (below
// MTLGPUFamilyApple7), which silently produces garbage transcripts (Handy
// issue #1608). `be` must be the initialized backend for `dev`. Honors the
// TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM hook; returns false when the Metal
// query is compiled out (non-Metal or GGML_BACKEND_DL builds). Exposed for
// the gate unit test.
bool metal_backend_lacks_simdgroup_mm(ggml_backend_t be, ggml_backend_dev_t dev);

// Resolve a BackendPlan from a caller's backend request by walking
// ggml's device registry.
//
//   requested:   the caller's transcribe_backend_request.
//                  AUTO   - best available: GPU/IGPU → ACCEL → CPU.
//                  CPU    - strict CPU only. No GPU, no ACCEL.
//                  METAL  - require Metal; error if not present.
//                  VULKAN - require Vulkan; error if not present.
//                Library-internal classification of what each value
//                actually picks up is documented in
//                transcribe-backend.h.
//   device:      NULL applies the requested backend's automatic policy. A
//                non-NULL process-local handle selects that exact registered
//                device. AUTO accepts a GPU, IGPU, or CPU; explicit backend
//                requests require a matching device. ACCEL devices cannot be
//                primary. Exact selection never falls back to another primary.
//   error_tag:   log prefix, e.g. "parakeet".
//   out:         populated on success. out.primary is the backend
//                that owns the weight buffer; out.scheduler_list
//                always has at least the primary and typically has
//                CPU last as a fallback. See BackendPlan for the
//                full semantic.
//
// Returns:
//   TRANSCRIBE_OK         on success.
//   TRANSCRIBE_ERR_INVALID_ARG if device is foreign, is an ACCEL device, or
//                          does not match a specific backend request.
//   TRANSCRIBE_ERR_BACKEND if the caller asked for a specific
//                          backend (METAL / VULKAN / CUDA / ROCM) that could not
//                          be initialized, or if the CPU backend
//                          itself fails to initialize (there is no
//                          fallback past CPU).
transcribe_status init_backends(transcribe_backend_request requested,
                                transcribe_device_t        device,
                                const char *               error_tag,
                                BackendPlan &              out);

// Storage behind a model's GGUF-backed weight tensors (its ctx_meta).
//
// Mapped tensors point straight into a read-only MappedFile of the GGUF,
// wrapped as a backend buffer (`mapped_buffer`). Every other tensor lives in
// an ordinary backend buffer filled from the file (`copy_buffer`). With
// mapping off, unavailable, or impossible for every tensor, copy_buffer
// holds them all: the classic path.
//
// Mapped weights are READ-ONLY: writing one (ggml_backend_tensor_set, an
// in-place graph op, a load-time fix-up) crashes. Load-time transforms must
// write their results into separate tensors in their own buffer, as
// fuse_batch_norm and promote_conv_pw_f16_to_f32_on_cpu do. A family should
// never write a ctx_meta tensor after load anyway: weights are shared by
// every session on the model.
class WeightStorage {
  public:
    WeightStorage() = default;

    ~WeightStorage() { reset(); }

    WeightStorage(const WeightStorage &)             = delete;
    WeightStorage & operator=(const WeightStorage &) = delete;
    WeightStorage(WeightStorage &&)                  = delete;
    WeightStorage & operator=(WeightStorage &&)      = delete;

    // Free both buffers, then unmap. Call it where the family used to free
    // its weight buffer (before the plan's backends); the destructor calls it
    // again harmlessly.
    void reset() noexcept;

    // True when `t`'s data lives in the file mapping.
    bool is_mapped(const ggml_tensor * t) const noexcept;

  private:
    friend transcribe_status allocate_weights(const std::string &                  path,
                                              const gguf_context *                 gguf_data,
                                              ggml_context *                       ctx_meta,
                                              const BackendPlan &                  plan,
                                              const transcribe_model_load_params * params,
                                              const char *                         error_tag,
                                              WeightStorage &                      out);
    friend transcribe_status allocate_copied_weights(ggml_context *      ctx_meta,
                                                     const BackendPlan & plan,
                                                     const char *        error_tag,
                                                     WeightStorage &     out);

    MappedFile            file_;
    ggml_backend_buffer_t mapped_buffer_ = nullptr;
    ggml_backend_buffer_t copy_buffer_   = nullptr;
};

// Whether a load may memory-map weights: false when params->no_mmap is set
// or the TRANSCRIBE_NO_MMAP environment flag is on. `params` may be NULL
// (defaults); otherwise it must be the full-size view that
// transcribe_model_load_file passes to the family loaders.
bool mmap_allowed(const transcribe_model_load_params * params);

// Give every tensor in `ctx_meta` storage on plan.primary, without filling
// the copied ones.
//
// When mmap_allowed(params), the platform can map files, and the primary
// device can wrap host memory as a buffer (ggml caps buffer_from_host_ptr +
// mmap_support: the CPU everywhere and Apple-silicon Metal), the GGUF at
// `path` is mapped read-only and every tensor whose file bytes are usable
// as-is (same type and size as in the file, inside it, and aligned for the
// device) points into it. The rest, and everything on other devices (Vulkan,
// CUDA, ROCm upload into device memory), goes into an ordinary buffer.
// Mapping failures are not errors: they log and fall back to copying.
//
// Follow with stream_tensor_data(..., &out) (or a family streamer that skips
// out.is_mapped(t)) to fill the copied tensors. load_weights does both.
//
// Returns TRANSCRIBE_OK, TRANSCRIBE_ERR_OOM when the copy buffer cannot be
// allocated, or TRANSCRIBE_ERR_BACKEND when the device rejects a mapped
// tensor. On error `out` may hold partial storage; the caller discards the
// whole model, whose destructor resets it.
transcribe_status allocate_weights(const std::string &                  path,
                                   const gguf_context *                 gguf_data,
                                   ggml_context *                       ctx_meta,
                                   const BackendPlan &                  plan,
                                   const transcribe_model_load_params * params,
                                   const char *                         error_tag,
                                   WeightStorage &                      out);

// Allocate every still-unallocated tensor in `ctx_meta` in an ordinary
// buffer on plan.primary (no mapping). For loaders whose weights are not
// GGUF file bytes (legacy whisper.cpp / Silero .bin files).
transcribe_status allocate_copied_weights(ggml_context *      ctx_meta,
                                          const BackendPlan & plan,
                                          const char *        error_tag,
                                          WeightStorage &     out);

// Stream tensor data from the GGUF data section on disk into
// already-allocated backend buffer slots: read straight into host buffers,
// uploaded through ggml_backend_tensor_set in bounded (16 MiB) chunks
// otherwise. Tensors that `skip` maps are left alone (they already hold the
// file bytes); with skip == nullptr every tensor in ctx_meta is streamed.
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
                                     const char *          error_tag,
                                     const WeightStorage * skip = nullptr);

// allocate_weights + stream_tensor_data: the weight load every GGUF family
// does after init_backends. Same contract as allocate_weights, plus
// TRANSCRIBE_ERR_GGUF from the stream.
transcribe_status load_weights(const std::string &                  path,
                               const gguf_context *                 gguf_data,
                               ggml_context *                       ctx_meta,
                               const BackendPlan &                  plan,
                               const transcribe_model_load_params * params,
                               const char *                         error_tag,
                               WeightStorage &                      out);

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
//   - If plan.primary_kind != BackendKind::Cpu: no-op, returns
//     TRANSCRIBE_OK; *out_ctx and *out_buffer are NOT touched. This is
//     the important semantic: strict-CPU is keyed off the classified
//     kind, not off ACCEL-vs-CPU ordering in a backend list, so a
//     request for TRANSCRIBE_BACKEND_CPU reliably triggers promotion
//     even on machines where ACCEL would otherwise sort ahead of CPU.
//   - If `slots` is empty: no-op, same return semantics.
//   - If the F16 → float type trait is missing: logs a warning and
//     returns TRANSCRIBE_OK without modifying anything.
//
// error_tag: log prefix, e.g. "parakeet" or "cohere".
transcribe_status promote_conv_pw_f16_to_f32_on_cpu(const BackendPlan &                plan,
                                                    const std::vector<ConvPwF32Slot> & slots,
                                                    const char *                       error_tag,
                                                    ggml_context **                    out_ctx,
                                                    ggml_backend_buffer_t *            out_buffer);

// Read a small F32 tensor from the GGUF file into a std::vector<float>.
//
// Validates:
//   - tensor exists in the GGUF index (returns Absent if not)
//   - tensor type is GGML_TYPE_F32 (returns BadType if not)
//   - byte count is divisible by sizeof(float)
//   - byte count matches expected_elems * sizeof(float) when
//     expected_elems > 0 (pass 0 to skip the size check)
//   - ifstream read consumed exactly the right number of bytes
//
// On success, `out` is resized and filled. On every other return
// (including Absent), `out` is cleared so stale data cannot leak.
// On any failure other than Absent, a diagnostic is logged using
// `error_tag` and `tensor_name`.
enum class ReadF32Result {
    Ok,       // tensor found, validated, and read successfully
    Absent,   // tensor not in the GGUF index (caller should compute)
    BadType,  // tensor exists but is not F32
    BadSize,  // tensor exists but byte count is wrong
    ReadErr,  // I/O error during read
};

ReadF32Result read_f32_tensor_checked(gguf_context *       gguf_ctx,
                                      const std::string &  gguf_path,
                                      const char *         tensor_name,
                                      size_t               expected_elems,  // 0 = skip size check
                                      const char *         error_tag,
                                      std::vector<float> & out);

}  // namespace transcribe::load_common
