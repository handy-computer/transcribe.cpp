// transcribe-backend.h - internal backend selection types.
//
// INTERNAL, C++17. Not part of the public ABI.
//
// The public API exposes a small transcribe_backend_request enum.
// Internally the library needs two things that don't belong in the public
// header: a typed classification of the ggml backend it landed on
// (BackendKind, replacing string-matching on ggml_backend_name()), and a
// BackendPlan holding the request, primary backend handle, its kind, and
// the scheduler list in priority order. Every per-family load() resolves a
// plan and hands it to helpers that key off the primary backend (F16→F32
// conv promotion, flash-attn defaults).

#pragma once

#include "ggml-backend.h"
#include "transcribe.h"

#include <cstdint>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_device;
typedef struct ggml_backend_device * ggml_backend_dev_t;

namespace transcribe {

// Library-internal classification from the ggml backend device type plus
// registry name. Only the kinds the library branches on are listed; anything
// else is Unknown.
enum class BackendKind {
    Unknown = 0,
    Cpu,       // ggml CPU backend (strict system memory)
    Metal,     // Apple Metal
    Vulkan,    // Vulkan compute
    Cuda,      // NVIDIA CUDA
    Rocm,      // AMD ROCm
    Sycl,      // Intel oneAPI / SYCL
    Accel,     // BLAS / AMX / other host-memory accelerator
    OtherGpu,  // GPU/IGPU device we don't have a special case for
};

// Human-readable kind label for logs. Never nullptr.
const char * kind_name(BackendKind kind);

// Unit-testable classification core. Production code normally calls
// classify_device(), which fetches these two inputs from ggml.
BackendKind classify_backend_type(enum ggml_backend_dev_type dev_type, const char * reg_name);

// Classify a backend device into a BackendKind. Uses
// ggml_backend_dev_type for the GPU/IGPU/ACCEL/CPU dimension and the
// reg name ("MTL", "Vulkan", "CUDA", "ROCm", "SYCL", ...) to resolve
// the vendor. Never returns Unknown for a valid device pointer.
BackendKind classify_device(ggml_backend_dev_t dev);

// Probe order for GPU device selection. `dev_types` holds the ggml
// device type of every registry device, indexed by registry position;
// the returned indices are the candidates to try: every discrete GPU
// (registry order) before any integrated GPU (registry order), non-GPU
// devices excluded. Discrete-first matters because Vulkan enumeration
// on hybrid-graphics machines often lists the display iGPU before the
// dGPU. Unit-testable core for try_init_kind.
std::vector<size_t> gpu_probe_order(const std::vector<enum ggml_backend_dev_type> & dev_types);

// Hardware identity of a GPU device as reported by its ggml backend. Today
// only the Vulkan backend reports it, through the
// ggml_backend_vk_get_device_hw_info proc address added by
// patches/ggml/0002-vulkan-expose-device-hw-info.patch. Every field is 0
// when the backend does not report it; vendor_id == 0 means "unknown".
struct GpuHwInfo {
    uint32_t vendor_id         = 0;  // PCI vendor id: 0x8086 Intel, 0x1002 AMD, 0x10de NVIDIA
    uint32_t device_id         = 0;  // PCI device id
    uint32_t driver_id         = 0;  // VkDriverId; log/diagnostic only
    uint32_t shader_core_count = 0;  // SMs / CUs / Xe-cores as counted by ggml; 0 = unknown
};

// Query hardware identity for `dev`. Returns false and leaves `out` untouched
// when the backend does not expose it (non-Vulkan device, or a Vulkan module
// built without the patch). Never throws.
//
// Test hook: TRANSCRIBE_TEST_GPU_HW_IDS=<vendor>:<device>[:<cores>] (hex ids,
// decimal core count) replaces the reported identity of every GPU/IGPU
// device and additionally makes the denylist treat the device as integrated,
// so the AUTO skip path can be exercised on any machine with a GPU.
bool query_gpu_hw_info(ggml_backend_dev_t dev, GpuHwInfo & out) noexcept;

// Everything the AUTO denylist decides on, as plain data so the policy is
// unit-testable without hardware. Filled by gpu_deny_input() in production.
struct GpuDenyInput {
    enum ggml_backend_dev_type dev_type = GGML_BACKEND_DEVICE_TYPE_GPU;
    GpuHwInfo                  hw;
};

// Build the denylist input for a registry device (type + hardware identity,
// with the test hook applied). Never throws.
GpuDenyInput gpu_deny_input(ggml_backend_dev_t dev) noexcept;

// The AUTO-selection denylist. Returns a short human-readable reason when the
// device is an integrated GPU known to be slower than the CPU for ASR — so
// AUTO should skip it — or nullptr when it is eligible. Pure function of its
// input: discrete GPUs, unknown vendors and unknown device ids are always
// eligible (new hardware is never blocked by omission). Explicit backend or
// device selection is never affected by this; callers only warn there.
//
// Rules (see src/transcribe-gpu-denylist-data.h for the id sets):
//   Intel  device id in the pre-Xe set (Gen 4 .. Gen 11).
//   AMD    device id in the GCN 1-3 APU set, or a reported compute-unit count
//          of 1..2 (the display-only stubs on Ryzen 7000/9000 desktop parts and
//          Mendocino).
const char * gpu_auto_deny_reason(const GpuDenyInput & in);

// True when TRANSCRIBE_NO_GPU_DENYLIST is set: AUTO considers every GPU, as it
// did before the denylist existed.
bool gpu_denylist_disabled();

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

// No-throw wrappers for ggml backend teardown. Family destructors are
// implicitly noexcept, so raw backend frees must not appear in library code;
// tests/lint_teardown.cmake enforces this. NULL is a no-op.
//
// Test hook: non-empty TRANSCRIBE_TEST_TEARDOWN_THROW injects an internal
// throw after the real free, proving containment without leaking the handle.
void safe_backend_free(ggml_backend_t backend) noexcept;
void safe_buffer_free(ggml_backend_buffer_t buffer) noexcept;
void safe_sched_free(ggml_backend_sched_t sched) noexcept;

// Release a session's per-run compute scratch: the scheduler (whose
// allocator only ever grows) first, then the no_alloc graph context. Both
// are nulled; families re-create them lazily on the next run. Used by
// transcribe_session::release_scratch and the base destructor, keeping the
// sched / compute_ctx free order in one place. NULLs are no-ops.
void release_compute_scratch(ggml_backend_sched_t & sched, struct ggml_context *& compute_ctx) noexcept;

}  // namespace transcribe
