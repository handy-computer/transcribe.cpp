// backend_init_throw_unit.cpp - device-init exception containment.
//
//   AUTO            -> the probe walks past the throwing GPU/ACCEL devices
//                      and lands on CPU.
//   METAL/VULKAN/   -> a specific-GPU request whose every candidate throws
//   CUDA               returns TRANSCRIBE_ERR_BACKEND.
//   CPU / CPU_ACCEL -> CPU primary still succeeds.

#include "ggml-backend.h"
#include "transcribe-backend.h"
#include "transcribe-load-common.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

void set_env(const char * key, const char * value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

void unset_env(const char * key) {
#if defined(_WIN32)
    _putenv_s(key, "");  // MSVC CRT: empty value removes the variable
#else
    unsetenv(key);
#endif
}

// Free every backend the plan created. scheduler_list owns each backend
// exactly once (primary included); reverse order mirrors the model
// destructor's cleanup convention.
void free_plan(transcribe::BackendPlan & plan) {
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        ggml_backend_free(*it);
    }
    plan = transcribe::BackendPlan{};
}

void test_baseline_no_hook() {
    unset_env("TRANSCRIBE_TEST_DEV_INIT_THROW");
    transcribe::BackendPlan plan;
    const transcribe_status st = transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, 0, "test", plan);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(plan.primary != nullptr);
    CHECK(!plan.scheduler_list.empty());
    free_plan(plan);
}

void test_nonmatching_hook_is_inert() {
    set_env("TRANSCRIBE_TEST_DEV_INIT_THROW", "no-such-device-name-xyzzy");
    transcribe::BackendPlan plan;
    const transcribe_status st = transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, 0, "test", plan);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(plan.primary != nullptr);
    free_plan(plan);
    unset_env("TRANSCRIBE_TEST_DEV_INIT_THROW");
}

#if !defined(_WIN32)
void test_empty_hook_value_is_inert() {
    // POSIX-only: Windows deletes the variable when setting an empty value.
    unset_env("TRANSCRIBE_TEST_DEV_INIT_THROW");
    transcribe::BackendPlan baseline;
    CHECK(transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, 0, "test", baseline) == TRANSCRIBE_OK);

    set_env("TRANSCRIBE_TEST_DEV_INIT_THROW", "");
    transcribe::BackendPlan plan;
    const transcribe_status st = transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, 0, "test", plan);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(plan.primary != nullptr);
    CHECK(plan.primary_kind == baseline.primary_kind);
    unset_env("TRANSCRIBE_TEST_DEV_INIT_THROW");

    free_plan(plan);
    free_plan(baseline);
}
#endif

void test_auto_falls_back_to_cpu_when_every_device_throws() {
    set_env("TRANSCRIBE_TEST_DEV_INIT_THROW", "*");
    transcribe::BackendPlan plan;
    const transcribe_status st = transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, 0, "test", plan);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(plan.primary != nullptr);
    CHECK(plan.primary_kind == transcribe::BackendKind::Cpu);
    free_plan(plan);
    unset_env("TRANSCRIBE_TEST_DEV_INIT_THROW");
}

void test_specific_gpu_request_fails_cleanly_when_every_device_throws() {
    set_env("TRANSCRIBE_TEST_DEV_INIT_THROW", "*");
    const transcribe_backend_request kinds[] = {
        TRANSCRIBE_BACKEND_METAL,
        TRANSCRIBE_BACKEND_VULKAN,
        TRANSCRIBE_BACKEND_CUDA,
    };
    for (const auto kind : kinds) {
        transcribe::BackendPlan plan;
        const transcribe_status st = transcribe::load_common::init_backends(kind, 0, "test", plan);
        CHECK(st == TRANSCRIBE_ERR_BACKEND);
        CHECK(plan.primary == nullptr);
    }
    unset_env("TRANSCRIBE_TEST_DEV_INIT_THROW");
}

void test_cpu_request_unaffected_by_hook() {
    set_env("TRANSCRIBE_TEST_DEV_INIT_THROW", "*");
    for (const auto kind : { TRANSCRIBE_BACKEND_CPU, TRANSCRIBE_BACKEND_CPU_ACCEL }) {
        transcribe::BackendPlan plan;
        const transcribe_status st = transcribe::load_common::init_backends(kind, 0, "test", plan);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(plan.primary_kind == transcribe::BackendKind::Cpu);
        free_plan(plan);
    }
    unset_env("TRANSCRIBE_TEST_DEV_INIT_THROW");
}

void test_explicit_gpu_device_fails_cleanly_when_it_throws() {
    // Only meaningful when a GPU/IGPU device sits at index > 0.
    int       gpu_index = -1;
    const int n         = transcribe_backend_device_count();
    for (int i = 1; i < n; ++i) {
        struct transcribe_backend_device dev;
        transcribe_backend_device_init(&dev);
        if (transcribe_get_backend_device(i, &dev) != TRANSCRIBE_OK) {
            continue;
        }
        if (dev.device_type == TRANSCRIBE_DEVICE_TYPE_GPU || dev.device_type == TRANSCRIBE_DEVICE_TYPE_IGPU) {
            gpu_index = i;
            break;
        }
    }
    if (gpu_index < 0) {
        return;
    }
    set_env("TRANSCRIBE_TEST_DEV_INIT_THROW", "*");
    transcribe::BackendPlan plan;
    const transcribe_status st =
        transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, gpu_index, "test", plan);
    CHECK(st == TRANSCRIBE_ERR_BACKEND);
    CHECK(plan.primary == nullptr);
    unset_env("TRANSCRIBE_TEST_DEV_INIT_THROW");
}

}  // namespace

int main() {
    // No-op in static builds; loads backend modules in a dynamic-backends
    // build so the device registry is populated either way.
    transcribe_init_backends_default();

    test_baseline_no_hook();
    test_nonmatching_hook_is_inert();
#if !defined(_WIN32)
    test_empty_hook_value_is_inert();
#endif
    test_auto_falls_back_to_cpu_when_every_device_throws();
    test_specific_gpu_request_fails_cleanly_when_every_device_throws();
    test_cpu_request_unaffected_by_hook();
    test_explicit_gpu_device_fails_cleanly_when_it_throws();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
