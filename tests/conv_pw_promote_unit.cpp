// conv_pw_promote_unit.cpp - unit test for promote_conv_pw_f16_to_f32_on_cpu.
//
// Tests the F16 → F32 promotion path directly rather than via a GGUF
// fixture. The existing synthetic Parakeet fixture only emits F32
// tensors, so a fixture-based test would be a false positive. This
// test programmatically creates F16 tensors backed by the CPU backend,
// feeds them through the promotion function, and verifies the output
// values round-trip correctly.
//
// Cases covered:
//   - CPU primary: promotion fires, dst slots are repointed, values
//     round-trip from F16 → F32 correctly.
//   - Non-CPU primary (Metal/GPU-like kind): promotion is a no-op,
//     dst slots are unchanged.
//   - Empty slots: no-op, no crash.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-load-common.h"

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

#define CHECK_EQ(actual, expected)                                                                              \
    do {                                                                                                        \
        const auto _a = (actual);                                                                               \
        const auto _e = (expected);                                                                             \
        if (_a != _e) {                                                                                         \
            std::fprintf(stderr, "FAIL %s:%d: got %d, expected %d\n", __FILE__, __LINE__, static_cast<int>(_a), \
                         static_cast<int>(_e));                                                                 \
            ++g_failures;                                                                                       \
        }                                                                                                       \
    } while (0)

// Bail macro for setup prerequisites: if the condition fails, log and
// return EXIT_FAILURE immediately rather than crashing on a null deref.
#define REQUIRE(cond)                                                                  \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "SETUP FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return EXIT_FAILURE;                                                       \
        }                                                                              \
    } while (0)

}  // namespace

int main() {
    using namespace transcribe;
    using transcribe::load_common::ConvPwF32Slot;
    using transcribe::load_common::promote_conv_pw_f16_to_f32_on_cpu;

    // -----------------------------------------------------------------
    // Set up a CPU backend and a small F16 tensor.
    // -----------------------------------------------------------------
    ggml_backend_t cpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_be == nullptr) {
        std::fprintf(stderr, "SKIP: could not init CPU backend\n");
        return 77;
    }

    // Create a ggml context with two small F16 tensors.
    const int64_t    n_elem      = 16;
    const size_t     ctx_size    = 2 * ggml_tensor_overhead() + 256;
    ggml_init_params init_params = { ctx_size, nullptr, true };
    ggml_context *   ctx         = ggml_init(init_params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * t1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elem);
    REQUIRE(t1 != nullptr);
    ggml_set_name(t1, "test.conv_pw1");
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elem);
    REQUIRE(t2 != nullptr);
    ggml_set_name(t2, "test.conv_pw2");

    // Allocate and fill with known F16 values.
    ggml_backend_buffer_t src_buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
    REQUIRE(src_buffer != nullptr);

    // Write deterministic F16 values: 1.0, 2.0, ..., 16.0 for t1;
    // -1.0, -2.0, ..., -16.0 for t2.
    {
        std::vector<ggml_fp16_t> f16_data(n_elem);
        for (int64_t i = 0; i < n_elem; ++i) {
            f16_data[i] = ggml_fp32_to_fp16(static_cast<float>(i + 1));
        }
        ggml_backend_tensor_set(t1, f16_data.data(), 0, n_elem * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n_elem; ++i) {
            f16_data[i] = ggml_fp32_to_fp16(-static_cast<float>(i + 1));
        }
        ggml_backend_tensor_set(t2, f16_data.data(), 0, n_elem * sizeof(ggml_fp16_t));
    }

    // -----------------------------------------------------------------
    // Case 1: CPU primary → promotion fires.
    // -----------------------------------------------------------------
    {
        BackendPlan plan;
        plan.primary      = cpu_be;
        plan.primary_kind = BackendKind::Cpu;
        plan.scheduler_list.push_back(cpu_be);

        ggml_tensor *              slot1 = t1;
        ggml_tensor *              slot2 = t2;
        std::vector<ConvPwF32Slot> slots = {
            { &slot1, t1 },
            { &slot2, t2 },
        };

        ggml_context *        out_ctx    = nullptr;
        ggml_backend_buffer_t out_buffer = nullptr;

        transcribe_status st = promote_conv_pw_f16_to_f32_on_cpu(plan, slots, "test-promote", &out_ctx, &out_buffer);

        CHECK_EQ(st, TRANSCRIBE_OK);
        REQUIRE(out_ctx != nullptr);
        REQUIRE(out_buffer != nullptr);

        // Slots should have been repointed to new F32 tensors.
        REQUIRE(slot1 != t1);
        REQUIRE(slot2 != t2);
        CHECK_EQ(static_cast<int>(slot1->type), static_cast<int>(GGML_TYPE_F32));
        CHECK_EQ(static_cast<int>(slot2->type), static_cast<int>(GGML_TYPE_F32));

        // Read back F32 values and verify round-trip.
        std::vector<float> f32_out(n_elem);

        ggml_backend_tensor_get(slot1, f32_out.data(), 0, n_elem * sizeof(float));
        for (int64_t i = 0; i < n_elem; ++i) {
            // F16 round-trip is exact for small integers.
            float expected = static_cast<float>(i + 1);
            CHECK(std::fabs(f32_out[i] - expected) < 1e-4f);
        }

        ggml_backend_tensor_get(slot2, f32_out.data(), 0, n_elem * sizeof(float));
        for (int64_t i = 0; i < n_elem; ++i) {
            float expected = -static_cast<float>(i + 1);
            CHECK(std::fabs(f32_out[i] - expected) < 1e-4f);
        }

        // Clean up the promotion artifacts. Note: we don't free cpu_be
        // here because it's shared with the plan's scheduler_list; the
        // promotion function doesn't own the primary backend.
        ggml_backend_buffer_free(out_buffer);
        ggml_free(out_ctx);
    }

    // -----------------------------------------------------------------
    // Case 2: Non-CPU primary → no-op.
    // -----------------------------------------------------------------
    {
        // Simulate a GPU primary by setting primary_kind to Metal.
        // We still use the CPU backend handle (it's just for the kind
        // check — the function early-returns before touching the
        // backend).
        BackendPlan plan;
        plan.primary      = cpu_be;
        plan.primary_kind = BackendKind::Metal;
        plan.scheduler_list.push_back(cpu_be);

        ggml_tensor *              slot1 = t1;
        std::vector<ConvPwF32Slot> slots = {
            { &slot1, t1 },
        };

        ggml_context *        out_ctx    = nullptr;
        ggml_backend_buffer_t out_buffer = nullptr;

        transcribe_status st = promote_conv_pw_f16_to_f32_on_cpu(plan, slots, "test-noop", &out_ctx, &out_buffer);

        CHECK_EQ(st, TRANSCRIBE_OK);
        // No-op: outparams not touched, slot not repointed.
        CHECK(out_ctx == nullptr);
        CHECK(out_buffer == nullptr);
        CHECK(slot1 == t1);
    }

    // -----------------------------------------------------------------
    // Case 3: Empty slots → no-op regardless of primary kind.
    // -----------------------------------------------------------------
    {
        BackendPlan plan;
        plan.primary      = cpu_be;
        plan.primary_kind = BackendKind::Cpu;
        plan.scheduler_list.push_back(cpu_be);

        std::vector<ConvPwF32Slot> slots;  // empty

        ggml_context *        out_ctx    = nullptr;
        ggml_backend_buffer_t out_buffer = nullptr;

        transcribe_status st = promote_conv_pw_f16_to_f32_on_cpu(plan, slots, "test-empty", &out_ctx, &out_buffer);

        CHECK_EQ(st, TRANSCRIBE_OK);
        CHECK(out_ctx == nullptr);
        CHECK(out_buffer == nullptr);
    }

    // -----------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------
    ggml_backend_buffer_free(src_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_be);

    if (g_failures > 0) {
        std::fprintf(stderr, "\n%d assertion(s) FAILED\n", g_failures);
    }

    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
