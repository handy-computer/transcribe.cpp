// backend_init_unit.cpp - unit tests for init_backends() runtime behavior.
//
// Covers the runtime backend-selection matrix that the classification
// unit (backend_classification_unit.cpp) does not reach:
//
//   - Strict CPU: primary_kind == Cpu, scheduler_list.size() == 1,
//     and the sole scheduler handle classifies as Cpu via
//     ggml_backend_get_device() + classify_device().
//   - Explicit Metal/Vulkan: asserts based on what init_backends()
//     actually returns, not on registry probes — a device can be
//     registered but fail initialization.
//   - AUTO: always succeeds; asserts based on the returned primary_kind.
//   - Invalid enum: returns TRANSCRIBE_ERR_INVALID_ARG.

#include "transcribe-load-common.h"
#include "transcribe-backend.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

#define CHECK_EQ(actual, expected)                                          \
    do {                                                                    \
        const auto _a = (actual);                                           \
        const auto _e = (expected);                                         \
        if (_a != _e) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: got %d, expected %d\n",       \
                         __FILE__, __LINE__,                                \
                         static_cast<int>(_a),                              \
                         static_cast<int>(_e));                             \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

// Bail macro for setup prerequisites: if the condition fails, log and
// return EXIT_FAILURE immediately rather than crashing on a null deref.
#define REQUIRE(cond)                                                       \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "SETUP FAIL %s:%d: %s\n",                  \
                         __FILE__, __LINE__, #cond);                        \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

// Free every backend in the scheduler list. Mirrors what the model
// destructor does: reverse order, then clear.
void free_plan(transcribe::BackendPlan & plan) {
    for (auto it = plan.scheduler_list.rbegin();
         it != plan.scheduler_list.rend(); ++it) {
        ggml_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

} // namespace

int main() {
    using namespace transcribe;
    using transcribe::load_common::init_backends;

    // ---------------------------------------------------------------
    // 1. Strict CPU
    // ---------------------------------------------------------------
    {
        BackendPlan plan;
        transcribe_status st = init_backends(
            TRANSCRIBE_BACKEND_CPU, "test-cpu", plan);
        REQUIRE(st == TRANSCRIBE_OK);
        CHECK_EQ(plan.primary_kind, BackendKind::Cpu);
        REQUIRE(plan.primary != nullptr);
        REQUIRE(plan.scheduler_list.size() == 1);

        // The sole scheduler handle must classify as CPU via the
        // device-level classifier, not string matching.
        ggml_backend_dev_t dev =
            ggml_backend_get_device(plan.scheduler_list[0]);
        REQUIRE(dev != nullptr);
        CHECK_EQ(classify_device(dev), BackendKind::Cpu);

        free_plan(plan);
    }

    // ---------------------------------------------------------------
    // 1b. CPU + ACCEL — primary still classifies as CPU; scheduler may
    // include extra accel handles depending on what ggml registered.
    // ---------------------------------------------------------------
    {
        BackendPlan plan;
        transcribe_status st = init_backends(
            TRANSCRIBE_BACKEND_CPU_ACCEL, "test-cpu-accel", plan);
        REQUIRE(st == TRANSCRIBE_OK);
        CHECK_EQ(plan.primary_kind, BackendKind::Cpu);
        REQUIRE(plan.primary != nullptr);
        REQUIRE(plan.scheduler_list.size() >= 1);

        // CPU must sit last in the scheduler list (ggml requirement);
        // any additional handles ahead of it are accel devices.
        ggml_backend_dev_t last_dev =
            ggml_backend_get_device(plan.scheduler_list.back());
        REQUIRE(last_dev != nullptr);
        CHECK_EQ(classify_device(last_dev), BackendKind::Cpu);

        free_plan(plan);
    }

    // ---------------------------------------------------------------
    // 2. Explicit Metal — assert based on actual init result
    // ---------------------------------------------------------------
    //
    // A Metal device can be registered in the ggml device registry
    // but fail ggml_backend_dev_init() (e.g. headless CI, sandbox
    // restrictions). We cannot pre-judge from the registry; instead
    // call init_backends() and assert based on what it returns.
    {
        BackendPlan plan;
        transcribe_status st = init_backends(
            TRANSCRIBE_BACKEND_METAL, "test-metal", plan);

        if (st == TRANSCRIBE_OK) {
            CHECK_EQ(plan.primary_kind, BackendKind::Metal);
            CHECK(plan.primary != nullptr);
            CHECK(plan.scheduler_list.size() >= 2);
            free_plan(plan);
        } else {
            CHECK_EQ(st, TRANSCRIBE_ERR_BACKEND);
        }
    }

    // ---------------------------------------------------------------
    // 3. Explicit Vulkan — same pattern as Metal
    // ---------------------------------------------------------------
    {
        BackendPlan plan;
        transcribe_status st = init_backends(
            TRANSCRIBE_BACKEND_VULKAN, "test-vulkan", plan);

        if (st == TRANSCRIBE_OK) {
            CHECK_EQ(plan.primary_kind, BackendKind::Vulkan);
            CHECK(plan.primary != nullptr);
            CHECK(plan.scheduler_list.size() >= 2);
            free_plan(plan);
        } else {
            CHECK_EQ(st, TRANSCRIBE_ERR_BACKEND);
        }
    }

    // ---------------------------------------------------------------
    // 4. AUTO — always succeeds (falls back to CPU at minimum)
    // ---------------------------------------------------------------
    //
    // AUTO probes GPU/IGPU devices and falls back to CPU if none
    // initializes. A GPU/IGPU device can exist in the registry but
    // fail initialization, so we assert based on the returned
    // primary_kind rather than pre-judging from the registry.
    {
        BackendPlan plan;
        transcribe_status st = init_backends(
            TRANSCRIBE_BACKEND_AUTO, "test-auto", plan);
        REQUIRE(st == TRANSCRIBE_OK);
        CHECK(plan.primary != nullptr);
        CHECK(plan.primary_kind != BackendKind::Unknown);
        CHECK(plan.scheduler_list.size() >= 1);

        // Verify the primary_kind is self-consistent: classify the
        // device behind the primary backend handle and confirm it
        // matches what init_backends reported.
        ggml_backend_dev_t dev = ggml_backend_get_device(plan.primary);
        if (dev != nullptr) {
            CHECK_EQ(classify_device(dev), plan.primary_kind);
        }

        free_plan(plan);
    }

    // ---------------------------------------------------------------
    // 5. Invalid enum — returns TRANSCRIBE_ERR_INVALID_ARG
    // ---------------------------------------------------------------
    {
        BackendPlan plan;
        transcribe_status st = init_backends(
            static_cast<transcribe_backend_request>(999),
            "test-invalid", plan);
        CHECK_EQ(st, TRANSCRIBE_ERR_INVALID_ARG);
    }

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    if (g_failures > 0) {
        std::fprintf(stderr, "\n%d assertion(s) FAILED\n", g_failures);
    }

    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
