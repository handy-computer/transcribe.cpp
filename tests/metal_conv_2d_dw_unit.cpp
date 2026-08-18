#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

ggml_backend_t backend_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == type) {
            if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
                return backend;
            }
        }
    }
    return nullptr;
}

ggml_backend_t metal_backend() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strncmp(ggml_backend_dev_name(dev), "MTL", 3) == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

std::vector<float> run_depthwise(ggml_backend_t backend) {
    ggml_init_params params{};
    params.mem_size    = 1024 * 1024;
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * weights = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 1, 1, 2);
    ggml_tensor * input   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 7, 1, 2, 2);
    ggml_tensor * output  = ggml_conv_2d_dw_direct(ctx, weights, input, 1, 1, 1, 0, 1, 1);
    ggml_cgraph * graph   = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t    buffer      = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const std::vector<float> weight_data = { 0.25f, -0.5f, 0.75f, -1.0f, 0.125f, 0.5f };
    const std::vector<float> input_data  = {
        1, 2, 3, 4, 5, 6, 7, -1, 0.5f, 2, -3, 4, -5, 6, 7, 6, 5, 4, 3, 2, 1, 3, -2, 1, 0, -1, 2, -3,
    };
    ggml_backend_tensor_set(weights, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "depthwise graph compute failed\n");
        std::abort();
    }

    std::vector<float> result(ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

}  // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_t metal = metal_backend();
    if (metal == nullptr) {
        std::fprintf(stderr, "SKIP: Metal backend unavailable\n");
        return 77;
    }
    ggml_backend_t cpu = backend_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        ggml_backend_free(metal);
        return EXIT_FAILURE;
    }

    const std::vector<float> expected = run_depthwise(cpu);
    const std::vector<float> actual   = run_depthwise(metal);
    ggml_backend_free(cpu);
    ggml_backend_free(metal);

    if (actual.size() != expected.size()) {
        return EXIT_FAILURE;
    }
    const size_t result_bytes = actual.size() * sizeof(float);
    if (std::memcmp(actual.data(), expected.data(), result_bytes) != 0) {
        std::fprintf(stderr, "Metal output is not byte-identical to CPU output\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "metal_conv_2d_dw_unit: %zu values are byte-identical\n", actual.size());
    return EXIT_SUCCESS;
}
