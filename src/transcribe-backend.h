// transcribe-backend.h - internal backend selection types.
//
// INTERNAL. C++17. Consumed only from other .cpp files inside src/.
// Not part of the public include/transcribe.h ABI.
//
// Rationale
// ---------
// The public API exposes a small `transcribe_backend_request` enum
// (auto|cpu|metal|vulkan). Internally the library needs two related
// things that don't belong in the public header:
//
//   1. A typed classification of whatever ggml backend we actually
//      landed on — `BackendKind`. This replaces string-matching on
//      `ggml_backend_name()` scattered across model code (flash-attn
//      policy, primary-backend-is-CPU policy, etc).
//
//   2. A small "backend plan" struct — `BackendPlan` — that holds the
//      original request, the primary backend handle, its classified
//      kind, and the full scheduler list in priority order. Every
//      per-family load() resolves one of these, stores it on the
//      model, and hands it to helpers that need to key off the
//      primary backend (F16→F32 conv pointwise promotion, flash-attn
//      default decisions).
//
// Keeping these out of transcribe-load-common.h means
// transcribe-flash-policy can include this header without dragging
// in the whole load scaffolding, and future consumers can ask
// "which backend is this?" without going through the loader.

#pragma once

#include "transcribe.h"

#include <vector>

struct ggml_backend;
typedef struct ggml_backend *     ggml_backend_t;
struct ggml_backend_device;
typedef struct ggml_backend_device * ggml_backend_dev_t;

namespace transcribe {

// Classified backend kind. This is a library-internal classification
// based on the ggml backend registry name; it is stable across ggml
// releases because it keys off ggml_backend_dev_type() plus a small
// name lookup, not on reg names that may churn.
//
// Only the kinds the library actually needs to branch on are listed;
// anything else is Unknown.
enum class BackendKind {
    Unknown = 0,
    Cpu,       // ggml CPU backend (strict system memory)
    Metal,     // Apple Metal
    Vulkan,    // Vulkan compute
    Cuda,      // NVIDIA CUDA
    Sycl,      // Intel oneAPI / SYCL
    Accel,     // BLAS / AMX / other host-memory accelerator
    OtherGpu,  // GPU/IGPU device we don't have a special case for
};

// Human-readable kind label for logs. Never nullptr.
const char * kind_name(BackendKind kind);

// Classify a backend device into a BackendKind. Uses
// ggml_backend_dev_type for the GPU/IGPU/ACCEL/CPU dimension and the
// reg name ("Metal", "Vulkan", "CUDA", "SYCL", ...) to resolve the
// vendor. Never returns Unknown for a valid device pointer.
BackendKind classify_device(ggml_backend_dev_t dev);

// A resolved backend plan. Produced by load_common::init_backends
// from a transcribe_backend_request and consumed by every helper
// that needs to know where the graph will run.
//
//   requested:       the original caller request, preserved for
//                    logging / diagnostics.
//   primary:         the first (highest-priority) backend in the
//                    scheduler list. This is the backend that owns
//                    the weight buffer and runs most of the graph.
//   primary_kind:    classified kind of `primary`. Helpers check this
//                    directly instead of calling ggml_backend_name
//                    and string-matching.
//   scheduler_list:  every backend that should participate in the
//                    ggml scheduler, in priority order. Primary
//                    first, then ACCEL (when appropriate), then CPU
//                    last as the fallback. Cleaned up in reverse in
//                    the model destructor.
struct BackendPlan {
    transcribe_backend_request  requested    = TRANSCRIBE_BACKEND_AUTO;
    ggml_backend_t              primary      = nullptr;
    BackendKind                 primary_kind = BackendKind::Unknown;
    std::vector<ggml_backend_t> scheduler_list;
};

} // namespace transcribe
