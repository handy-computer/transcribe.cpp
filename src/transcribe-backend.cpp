// transcribe-backend.cpp - internal backend selection helpers.
//
// See transcribe-backend.h for rationale. This file owns the
// device-classification rules: given a ggml_backend_dev_t, what
// library-level BackendKind does it correspond to?

#include "transcribe-backend.h"

#include "ggml-backend.h"

#include <cstring>

namespace transcribe {

const char * kind_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Cpu:      return "cpu";
        case BackendKind::Metal:    return "metal";
        case BackendKind::Vulkan:   return "vulkan";
        case BackendKind::Cuda:     return "cuda";
        case BackendKind::Sycl:     return "sycl";
        case BackendKind::Accel:    return "accel";
        case BackendKind::OtherGpu: return "gpu";
        case BackendKind::Unknown:
        default:                    return "unknown";
    }
}

// Return true if `reg_name` (the ggml backend registry name) starts
// with the given prefix. ggml's registry names look like "Metal",
// "Vulkan", "CUDA", "SYCL", "BLAS", "CPU", etc. Prefix matching is
// intentional: registry names can get version suffixes or device
// index suffixes in some ggml builds.
static bool reg_name_is(const char * reg_name, const char * prefix) {
    if (reg_name == nullptr || prefix == nullptr) return false;
    return std::strncmp(reg_name, prefix, std::strlen(prefix)) == 0;
}

BackendKind classify_device(ggml_backend_dev_t dev) {
    if (dev == nullptr) return BackendKind::Unknown;

    // First cut: ggml's device-type classification. This tells us
    // CPU vs GPU vs IGPU vs ACCEL without any name matching.
    const auto dev_type = ggml_backend_dev_type(dev);
    if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return BackendKind::Cpu;
    }
    if (dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        return BackendKind::Accel;
    }

    // GPU or IGPU: resolve the vendor by looking at the backend
    // registry name. The reg is the parent container for all devices
    // of a given backend family, so its name is stable even when the
    // device name includes a per-device suffix.
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * reg_name =
        (reg != nullptr) ? ggml_backend_reg_name(reg) : nullptr;

    if      (reg_name_is(reg_name, "Metal"))  return BackendKind::Metal;
    else if (reg_name_is(reg_name, "Vulkan")) return BackendKind::Vulkan;
    else if (reg_name_is(reg_name, "CUDA"))   return BackendKind::Cuda;
    else if (reg_name_is(reg_name, "SYCL"))   return BackendKind::Sycl;

    return BackendKind::OtherGpu;
}

} // namespace transcribe
