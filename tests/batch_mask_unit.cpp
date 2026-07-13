#include "arch/parakeet/parakeet.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-batch-util.h"

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
    using transcribe::parakeet::pre_encode_time_out;

    // Symmetric and causal padding agree for odd inputs. Causal padding keeps
    // one additional frame for even inputs, including at every x8 stage.
    CHECK(pre_encode_time_out(5, false) == 3);
    CHECK(pre_encode_time_out(5, true) == 3);
    CHECK(pre_encode_time_out(6, false) == 3);
    CHECK(pre_encode_time_out(6, true) == 4);
    int regular = 100;
    int causal  = 100;
    for (int i = 0; i < 3; ++i) {
        regular = pre_encode_time_out(regular, false);
        causal  = pre_encode_time_out(causal, true);
    }
    CHECK(regular == 13);
    CHECK(causal == 14);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: could not initialize CPU backend\n");
        return 77;
    }

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        ggml_backend_free(backend);
        return EXIT_FAILURE;
    }

    constexpr int T = 4;
    constexpr int B = 2;
    ggml_tensor * mask    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, B);
    ggml_tensor * softmax = ggml_soft_max(ctx, mask);
    ggml_cgraph *  graph   = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, softmax);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return EXIT_FAILURE;
    }

    // Row 0 has two valid keys. Row 1 models a padded query whose complete
    // local-attention window falls on padded keys.
    transcribe::fill_keypad_mask(mask, { 2, 0 }, T, B);

    std::vector<float> mask_values(T * B);
    ggml_backend_tensor_get(mask, mask_values.data(), 0, mask_values.size() * sizeof(float));
    CHECK(mask_values[0] == 0.0f);
    CHECK(mask_values[1] == 0.0f);
    for (int i = 2; i < T * B; ++i) {
        CHECK(std::isfinite(mask_values[static_cast<size_t>(i)]));
        CHECK(mask_values[static_cast<size_t>(i)] < -1e20f);
    }

    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> probabilities(T * B);
    ggml_backend_tensor_get(softmax, probabilities.data(), 0, probabilities.size() * sizeof(float));
    for (float value : probabilities) {
        CHECK(std::isfinite(value));
    }
    CHECK(std::fabs(probabilities[0] - 0.5f) < 1e-6f);
    CHECK(std::fabs(probabilities[1] - 0.5f) < 1e-6f);
    CHECK(probabilities[2] == 0.0f);
    CHECK(probabilities[3] == 0.0f);
    for (int i = T; i < T * B; ++i) {
        CHECK(std::fabs(probabilities[static_cast<size_t>(i)] - 0.25f) < 1e-6f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (g_failures != 0) {
        std::fprintf(stderr, "batch_mask_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "batch_mask_unit: ok\n");
    return EXIT_SUCCESS;
}
