// transcribe-backend.cpp - internal backend selection helpers.
//
// See transcribe-backend.h for rationale. This file owns the
// device-classification rules: given a ggml_backend_dev_t, what
// library-level BackendKind does it correspond to?

#include "transcribe-backend.h"

#include "ggml-vulkan.h"
#include "ggml.h"
#include "transcribe-env.h"
#include "transcribe-gpu-denylist-data.h"
#include "transcribe-log.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>

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

constexpr uint32_t kVendorIntel = 0x8086;
constexpr uint32_t kVendorAmd   = 0x1002;

template <size_t N> bool id_in(const uint16_t (&table)[N], uint32_t device_id) {
    if (device_id > 0xffff) {
        return false;
    }
    return std::binary_search(std::begin(table), std::end(table), static_cast<uint16_t>(device_id));
}

// Parse the TRANSCRIBE_TEST_GPU_HW_IDS hook. Returns false when unset,
// empty, or malformed (malformed values are ignored with a warning rather
// than trusted).
bool test_hw_ids_override(GpuHwInfo & out) {
    const char * spec = env::str("TRANSCRIBE_TEST_GPU_HW_IDS");
    if (spec == nullptr) {
        return false;
    }
    unsigned  vendor = 0;
    unsigned  device = 0;
    unsigned  cores  = 0;
    const int n      = std::sscanf(spec, "%x:%x:%u", &vendor, &device, &cores);
    if (n < 2) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                "TRANSCRIBE_TEST_GPU_HW_IDS=\"%s\" is not <vendor>:<device>[:<cores>]; ignored", spec);
        return false;
    }
    out                   = GpuHwInfo{};
    out.vendor_id         = vendor;
    out.device_id         = device;
    out.shader_core_count = (n >= 3) ? cores : 0;
    return true;
}

}  // namespace

bool query_gpu_hw_info(ggml_backend_dev_t dev, GpuHwInfo & out) noexcept {
    if (dev == nullptr) {
        return false;
    }
    try {
        const auto dev_type = ggml_backend_dev_type(dev);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return false;
        }
        if (test_hw_ids_override(out)) {
            return true;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg == nullptr) {
            return false;
        }
        // Resolved by name so this works identically for a compiled-in backend
        // and a dynamically loaded module; nullptr means the module predates
        // the patch (or is not Vulkan) and the device simply has no identity.
        auto fn = reinterpret_cast<ggml_backend_vk_get_device_hw_info_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_get_device_hw_info"));
        if (fn == nullptr) {
            return false;
        }
        ggml_vk_device_hw_info info = {};
        if (!fn(dev, &info)) {
            return false;
        }
        out.vendor_id         = info.vendor_id;
        out.device_id         = info.device_id;
        out.driver_id         = info.driver_id;
        out.shader_core_count = info.shader_core_count;
        return true;
    } catch (const std::exception & e) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "device hardware query threw: %s - treating identity as unknown", e.what());
        return false;
    } catch (...) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                "device hardware query threw an unknown exception - treating identity as unknown");
        return false;
    }
}

GpuDenyInput gpu_deny_input(ggml_backend_dev_t dev) noexcept {
    GpuDenyInput in;
    if (dev == nullptr) {
        return in;
    }
    try {
        in.dev_type = ggml_backend_dev_type(dev);
    } catch (...) {
        return in;
    }
    if (in.dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && in.dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return in;
    }
    if (test_hw_ids_override(in.hw)) {
        // The hook pretends the device is the named integrated part.
        in.dev_type = GGML_BACKEND_DEVICE_TYPE_IGPU;
        return in;
    }
    query_gpu_hw_info(dev, in.hw);
    return in;
}

const char * gpu_auto_deny_reason(const GpuDenyInput & in) {
    // Only integrated parts are ever denied: a discrete GPU, however old, has
    // its own memory bus and is not the failure mode this guards against.
    if (in.dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return nullptr;
    }
    switch (in.hw.vendor_id) {
        case kVendorIntel:
            if (id_in(gpu_denylist_data::kIntelPreXe, in.hw.device_id)) {
                return "pre-Xe Intel integrated graphics (Gen 11 or older) is slower than the CPU for ASR";
            }
            return nullptr;
        case kVendorAmd:
            if (id_in(gpu_denylist_data::kAmdGcnApu, in.hw.device_id)) {
                return "GCN 1-3 AMD APU graphics is slower than the CPU for ASR";
            }
            if (in.hw.shader_core_count >= 1 && in.hw.shader_core_count <= 2) {
                return "AMD integrated graphics with only 1-2 compute units is a display adapter, slower than the CPU "
                       "for ASR";
            }
            return nullptr;
        default:
            return nullptr;
    }
}

bool gpu_denylist_disabled() {
    return env::flag("TRANSCRIBE_NO_GPU_DENYLIST");
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
}

void release_compute_scratch(ggml_backend_sched_t & sched, struct ggml_context *& compute_ctx) noexcept {
    safe_sched_free(sched);
    sched = nullptr;
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

}  // namespace transcribe
