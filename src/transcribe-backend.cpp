// transcribe-backend.cpp - internal backend selection helpers.
//
// See transcribe-backend.h for rationale. This file owns the
// device-classification rules: given a ggml_backend_dev_t, what
// library-level BackendKind does it correspond to?

#include "transcribe-backend.h"

#include "ggml.h"
#include "transcribe-log.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace transcribe {

const char * kind_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Cpu:
            return "cpu";
        case BackendKind::Metal:
            return "metal";
        case BackendKind::Vulkan:
            return "vulkan";
        case BackendKind::Cuda:
            return "cuda";
        case BackendKind::Rocm:
            return "rocm";
        case BackendKind::Sycl:
            return "sycl";
        case BackendKind::Accel:
            return "accel";
        case BackendKind::OtherGpu:
            return "gpu";
        case BackendKind::Unknown:
        default:
            return "unknown";
    }
}

// Return true if `reg_name` (the ggml backend registry name) starts
// with the given prefix. ggml's registry names look like "MTL",
// "Vulkan", "CUDA", "ROCm", "SYCL", "BLAS", "CPU", etc. Prefix matching is
// intentional: registry names can get version suffixes or device
// index suffixes in some ggml builds.
static bool reg_name_is(const char * reg_name, const char * prefix) {
    if (reg_name == nullptr || prefix == nullptr) {
        return false;
    }
    return std::strncmp(reg_name, prefix, std::strlen(prefix)) == 0;
}

BackendKind classify_backend_type(enum ggml_backend_dev_type dev_type, const char * reg_name) {
    if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return BackendKind::Cpu;
    }
    if (dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        return BackendKind::Accel;
    }
    if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return BackendKind::Unknown;
    }

    if (reg_name_is(reg_name, "MTL") || reg_name_is(reg_name, "Metal")) {
        return BackendKind::Metal;
    } else if (reg_name_is(reg_name, "Vulkan")) {
        return BackendKind::Vulkan;
    } else if (reg_name_is(reg_name, "CUDA")) {
        return BackendKind::Cuda;
    } else if (reg_name_is(reg_name, "ROCm")) {
        return BackendKind::Rocm;
    } else if (reg_name_is(reg_name, "SYCL")) {
        return BackendKind::Sycl;
    }

    return BackendKind::OtherGpu;
}

BackendKind classify_device(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return BackendKind::Unknown;
    }

    // First cut: ggml's device-type classification. This tells us CPU
    // vs GPU vs IGPU vs ACCEL without any name matching. For GPU and
    // IGPU devices, the registry name resolves the vendor-specific kind.
    const auto         dev_type = ggml_backend_dev_type(dev);
    ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
    const char *       reg_name = (reg != nullptr) ? ggml_backend_reg_name(reg) : nullptr;
    return classify_backend_type(dev_type, reg_name);
}

std::vector<size_t> gpu_probe_order(const std::vector<enum ggml_backend_dev_type> & dev_types) {
    std::vector<size_t> order;
    order.reserve(dev_types.size());
    for (size_t i = 0; i < dev_types.size(); ++i) {
        if (dev_types[i] == GGML_BACKEND_DEVICE_TYPE_GPU) {
            order.push_back(i);
        }
    }
    for (size_t i = 0; i < dev_types.size(); ++i) {
        if (dev_types[i] == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            order.push_back(i);
        }
    }
    return order;
}

namespace {

// Shared body for safe_* teardown wrappers. The test hook fires after the
// real free; present-but-empty is inert.
template <typename Fn> void contained_free(const char * what, Fn && do_free) noexcept {
    try {
        do_free();
        if (const char * hook = std::getenv("TRANSCRIBE_TEST_TEARDOWN_THROW"); hook != nullptr && hook[0] != '\0') {
            throw std::runtime_error("TRANSCRIBE_TEST_TEARDOWN_THROW fault injection");
        }
    } catch (const std::exception & e) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s threw during teardown (contained; resource may leak): %s", what,
                e.what());
    } catch (...) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                "%s threw an unknown exception during teardown (contained; resource may leak)", what);
    }
}

}  // namespace

namespace {

// No real buffer size, so the first observation of a scheduler always logs.
constexpr size_t k_never_logged = static_cast<size_t>(-1);

// Scratch high-water marks per scheduler. Families that rebuild and reallocate
// their step graph on every decoded token (whisper/canary on CPU, or whenever
// debug dumping is on) would otherwise emit one identical line per token. The
// scheduler's buffer only ever grows within a run, so a line is emitted exactly
// when some graph raises it.
//
// Process-global under a mutex, not thread-local: a session may be moved
// between threads (include/transcribe.h), so the allocating thread and the
// thread that frees the session need not match. A thread-local map would leave
// stale entries on the allocating thread and could suppress the first line of a
// later scheduler that reuses the address. The lock is noise next to the graph
// allocation it follows.
std::mutex                                                    t_scratch_mutex;
std::unordered_map<ggml_backend_sched_t, std::vector<size_t>> t_scratch_high_water;

// Drop a scheduler's marks. Called from teardown, which is noexcept: the lock
// cannot realistically fail, but an escaping exception there would terminate.
void forget_sched_scratch(ggml_backend_sched_t sched) noexcept {
    try {
        const std::lock_guard<std::mutex> lock(t_scratch_mutex);
        t_scratch_high_water.erase(sched);
    } catch (...) {
        // Bookkeeping for a debug line only; never let it escape.
    }
}

void log_sched_scratch(ggml_backend_sched_t sched, const char * tag, int line, long long extent) {
    // Thread-local scratch for the entries that rose, so the per-token no-op
    // case allocates nothing. (This thread_local carries no semantics; the
    // marks themselves are global.)
    static thread_local std::vector<std::pair<const char *, size_t>> risen;
    risen.clear();

    {
        const std::lock_guard<std::mutex> lock(t_scratch_mutex);
        std::vector<size_t> &             high_water = t_scratch_high_water[sched];
        const int                         n_backends = ggml_backend_sched_get_n_backends(sched);
        if (static_cast<int>(high_water.size()) != n_backends) {
            high_water.assign(static_cast<size_t>(n_backends), k_never_logged);
        }
        for (int i = 0; i < n_backends; ++i) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
            const size_t   size    = ggml_backend_sched_get_buffer_size(sched, backend);
            if (size == high_water[static_cast<size_t>(i)]) {
                continue;
            }
            high_water[static_cast<size_t>(i)] = size;
            risen.emplace_back(ggml_backend_name(backend), size);
        }
    }

    // Log outside the lock: the host callback may be invoked on this thread and
    // is allowed to re-enter the library, which must not deadlock on our mutex.
    for (const auto & [name, size] : risen) {
        if (extent >= 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "%s@%d: sched buffer %s -> %.1f MB (extent %lld)", tag, line, name,
                    static_cast<double>(size) / 1e6, extent);
        } else {
            log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "%s@%d: sched buffer %s -> %.1f MB", tag, line, name,
                    static_cast<double>(size) / 1e6);
        }
    }
}

}  // namespace

void safe_backend_free(ggml_backend_t backend) noexcept {
    if (backend == nullptr) {
        return;
    }
    contained_free("ggml_backend_free", [&] { ggml_backend_free(backend); });
}

void safe_buffer_free(ggml_backend_buffer_t buffer) noexcept {
    if (buffer == nullptr) {
        return;
    }
    contained_free("ggml_backend_buffer_free", [&] { ggml_backend_buffer_free(buffer); });
}

void safe_sched_free(ggml_backend_sched_t sched) noexcept {
    if (sched == nullptr) {
        return;
    }
    contained_free("ggml_backend_sched_free", [&] { ggml_backend_sched_free(sched); });
    // Drop this scheduler's scratch high-water marks (see log_sched_scratch).
    forget_sched_scratch(sched);
}

void release_compute_scratch(ggml_backend_sched_t & sched, struct ggml_context *& compute_ctx) noexcept {
    safe_sched_free(sched);
    sched = nullptr;
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

bool alloc_sched_graph(ggml_backend_sched_t sched,
                       struct ggml_cgraph * graph,
                       const char *         tag,
                       int                  line,
                       long long            extent) {
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        return false;
    }
    log_sched_scratch(sched, tag, line, extent);
    return true;
}

}  // namespace transcribe
