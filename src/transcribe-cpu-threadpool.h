// transcribe-cpu-threadpool.h - DL-safe access to the CPU backend's
// threadpool entry points.
//
// INTERNAL, C++17. Not part of the public ABI.
//
// ggml_threadpool_new / ggml_threadpool_free / ggml_backend_cpu_set_threadpool
// are defined by the CPU backend, which under GGML_BACKEND_DL is a loadable
// module (libggml-cpu.so) rather than part of libggml-base: referencing them
// directly leaves unresolved symbols that fail the link of every consumer
// executable in the shared/DL posture. Reach them through the backend
// registry instead, as we already do for ggml_backend_set_n_threads
// (transcribe-batch-util.cpp).
//
// ggml_threadpool_params_default() is not one of these: it lives in
// ggml-base (ggml.c) and stays directly linkable, so callers still include
// ggml.h (or ggml-cpu.h) for the params struct.
//
// All helpers are best-effort: when an entry point is unavailable (non-CPU
// backend, exotic build) they report failure instead of aborting, and the
// caller keeps ggml's default behaviour of a transient per-compute pool.

#pragma once

#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace transcribe {

typedef ggml_threadpool_t (*pfn_cpu_threadpool_new)(ggml_threadpool_params *);
typedef void (*pfn_cpu_threadpool_free)(ggml_threadpool_t);
typedef void (*pfn_cpu_threadpool_set)(ggml_backend_t, ggml_threadpool_t);

// Resolve a backend entry point by name via the registry. Returns nullptr
// when `backend` is null/unregistered or the symbol is not exported.
// `backend` must be alive for the lookup to succeed.
inline void * cpu_backend_proc(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg != nullptr ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

// Create a threadpool on the backend's CPU module. Returns nullptr when the
// entry point is unavailable or the pool could not be created; callers then
// fall back to the transient pools ggml spawns per graph_compute.
inline ggml_threadpool_t cpu_threadpool_new(ggml_backend_t backend, ggml_threadpool_params * params) {
    auto fn = (pfn_cpu_threadpool_new) cpu_backend_proc(backend, "ggml_threadpool_new");
    return fn != nullptr ? fn(params) : nullptr;
}

// Attach `tp` to `backend`; nullptr detaches. Returns false when the entry
// point is unavailable, in which case nothing was attached and `tp` is still
// the caller's to free.
inline bool cpu_threadpool_set(ggml_backend_t backend, ggml_threadpool_t tp) {
    auto fn = (pfn_cpu_threadpool_set) cpu_backend_proc(backend, "ggml_backend_cpu_set_threadpool");
    if (fn == nullptr) {
        return false;
    }
    fn(backend, tp);
    return true;
}

// Free a threadpool via the backend's CPU module. Resolving the free fn needs
// the backend alive, so callers must invoke this BEFORE freeing the backend.
inline void cpu_threadpool_free(ggml_backend_t backend, ggml_threadpool_t tp) {
    if (tp == nullptr) {
        return;
    }
    auto fn = (pfn_cpu_threadpool_free) cpu_backend_proc(backend, "ggml_threadpool_free");
    if (fn != nullptr) {
        fn(tp);
    }
}

}  // namespace transcribe
