// backend_probe_order_unit.cpp - unit tests for GPU probe ordering.
//
// The runtime backend selector probes GPU candidates in gpu_probe_order:
// every discrete GPU before any integrated GPU, preserving registry order
// within each tier. This test covers the pure ordering core so the
// discrete-before-integrated policy stays tested without requiring a
// hybrid-graphics machine in CI.

#include "transcribe-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK_ORDER(types, expected)                                         \
    do {                                                                     \
        const std::vector<size_t> _got = transcribe::gpu_probe_order(types); \
        const std::vector<size_t> _exp = (expected);                         \
        if (_got != _exp) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d: got {", __FILE__, __LINE__);   \
            for (size_t v : _got) {                                          \
                std::fprintf(stderr, " %zu", v);                             \
            }                                                                \
            std::fprintf(stderr, " }, expected {");                          \
            for (size_t v : _exp) {                                          \
                std::fprintf(stderr, " %zu", v);                             \
            }                                                                \
            std::fprintf(stderr, " }\n");                                    \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

}  // namespace

int main() {
    using DevTypes = std::vector<enum ggml_backend_dev_type>;
    using Order    = std::vector<size_t>;

    // No devices, and no GPU devices at all.
    CHECK_ORDER(DevTypes{}, Order{});
    CHECK_ORDER((DevTypes{ GGML_BACKEND_DEVICE_TYPE_CPU, GGML_BACKEND_DEVICE_TYPE_ACCEL }), Order{});

    // Single-GPU machines: the sole GPU is the only candidate,
    // whichever tier it is in.
    CHECK_ORDER((DevTypes{ GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_CPU }), (Order{ 0 }));
    CHECK_ORDER((DevTypes{ GGML_BACKEND_DEVICE_TYPE_CPU, GGML_BACKEND_DEVICE_TYPE_IGPU }), (Order{ 1 }));

    // The hybrid-graphics case that motivated the tiering: the iGPU
    // enumerates first, but the discrete GPU must be probed first.
    CHECK_ORDER((DevTypes{ GGML_BACKEND_DEVICE_TYPE_IGPU, GGML_BACKEND_DEVICE_TYPE_GPU }), (Order{ 1, 0 }));
    CHECK_ORDER((DevTypes{ GGML_BACKEND_DEVICE_TYPE_IGPU, GGML_BACKEND_DEVICE_TYPE_CPU, GGML_BACKEND_DEVICE_TYPE_GPU }),
                (Order{ 2, 0 }));

    // Registry order is preserved within each tier.
    CHECK_ORDER((DevTypes{ GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU, GGML_BACKEND_DEVICE_TYPE_GPU,
                           GGML_BACKEND_DEVICE_TYPE_CPU, GGML_BACKEND_DEVICE_TYPE_IGPU }),
                (Order{ 0, 2, 1, 4 }));

    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
