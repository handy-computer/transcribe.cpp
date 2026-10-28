// conformer_glu_unit.cpp - unit test for transcribe::conformer::glu.
//
// Checks the GLU values on a batched [2*C, T, B] input, and that the sigmoid
// reads a contiguous tensor: ggml-cuda runs unary ops only on contiguous
// sources, so a strided input would move the sigmoid to the CPU.

#include "conformer/conformer.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: could not initialize CPU backend\n");
        return 77;
    }

    ggml_init_params params{};
    params.mem_size    = 1024 * 1024;
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        ggml_backend_free(backend);
        return EXIT_FAILURE;
    }

    const int     C = 4;
    const int     T = 3;
    const int     B = 2;
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2 * C, T, B);
    ggml_tensor * y = transcribe::conformer::glu(ctx, x);

    CHECK(y->ne[0] == C && y->ne[1] == T && y->ne[2] == B);
    const ggml_tensor * sigmoid = y->src[1];
    CHECK(sigmoid->op == GGML_OP_UNARY && ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID);
    CHECK(ggml_is_contiguous(sigmoid->src[0]));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return EXIT_FAILURE;
    }

    std::vector<float> input(2 * C * T * B);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.25f * static_cast<float>(i) - 3.0f;
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));

    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> output(C * T * B);
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
    for (int row = 0; row < T * B; ++row) {
        for (int c = 0; c < C; ++c) {
            const float gate     = input[row * 2 * C + c];
            const float value    = input[row * 2 * C + C + c];
            const float expected = gate / (1.0f + std::exp(-value));
            CHECK(std::fabs(output[row * C + c] - expected) <= 1e-5f * (1.0f + std::fabs(expected)));
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (g_failures != 0) {
        std::fprintf(stderr, "conformer_glu_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "conformer_glu_unit: ok\n");
    return EXIT_SUCCESS;
}
