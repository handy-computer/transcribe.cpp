// backend_classification_unit.cpp - unit tests for backend kind classification.
//
// The production classifier reads ggml device handles from the runtime
// registry. This test covers the pure classification core so the Metal
// registry-name mapping stays tested without requiring Metal hardware in CI.

#include "transcribe-backend.h"

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

#define CHECK_EQ(actual, expected)                                          \
    do {                                                                    \
        const auto _a = (actual);                                           \
        const auto _e = (expected);                                         \
        if (_a != _e) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: got %s, expected %s\n",       \
                         __FILE__, __LINE__,                               \
                         transcribe::kind_name(_a),                        \
                         transcribe::kind_name(_e));                       \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

} // namespace

int main() {
    using transcribe::BackendKind;
    using transcribe::classify_backend_type;

    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_CPU, "CPU"),
             BackendKind::Cpu);
    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "BLAS"),
             BackendKind::Accel);

    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "MTL"),
             BackendKind::Metal);
    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "MTL0"),
             BackendKind::Metal);
    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "Metal"),
             BackendKind::Metal);
    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_IGPU, "Vulkan"),
             BackendKind::Vulkan);
    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA"),
             BackendKind::Cuda);
    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "SYCL"),
             BackendKind::Sycl);
    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, "WebGPU"),
             BackendKind::OtherGpu);
    CHECK_EQ(classify_backend_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr),
             BackendKind::OtherGpu);

    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
