// backend_gpu_denylist_gate_unit.cpp - AUTO device selection skips an
// integrated GPU on the hardware denylist (Handy issue #1884), while explicit
// selection still honors it.
//
// Drives the gate through the TRANSCRIBE_TEST_GPU_HW_IDS hook, which makes
// every GPU device report the named identity (and count as integrated), so
// the real selection path runs on whatever GPU the test machine has:
//
//   AUTO + denylisted identity      -> skip the GPU, land on CPU.
//   AUTO + eligible identity        -> selection unchanged from the baseline.
//   AUTO + TRANSCRIBE_NO_GPU_DENYLIST -> selection unchanged from the baseline.
//   explicit backend + denylisted   -> honored (warn only).
//   explicit device + denylisted    -> honored (warn only).
//
// Every case is a no-op on machines with no GPU device (CPU-only CI).

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
    _putenv_s(key, "");
#else
    unsetenv(key);
#endif
}

void clear_hooks() {
    unset_env("TRANSCRIBE_TEST_GPU_HW_IDS");
    unset_env("TRANSCRIBE_NO_GPU_DENYLIST");
}

void free_plan(transcribe::BackendPlan & plan) {
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        ggml_backend_free(*it);
    }
    plan = transcribe::BackendPlan{};
}

// UHD Graphics 620 (KBL GT2): the part from Handy #1884.
constexpr const char * kDenied = "8086:5917";
// Iris Xe (TGL GT2): eligible.
constexpr const char * kEligible = "8086:9a49";

// The first GPU/IGPU device the AUTO probe would consider, or nullptr.
ggml_backend_dev_t g_gpu = nullptr;
// What AUTO selects with no hooks at all, captured once in main(). On a machine
// whose GPU is genuinely gated (e.g. a pre-Apple7 Metal GPU) this is CPU.
transcribe::BackendKind g_baseline = transcribe::BackendKind::Unknown;

ggml_backend_dev_t find_gpu_device() {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto         t   = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return dev;
        }
    }
    return nullptr;
}

transcribe::BackendKind auto_kind() {
    transcribe::BackendPlan plan;
    const transcribe_status st = transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, nullptr, "test", plan);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(plan.primary != nullptr);
    const transcribe::BackendKind kind = plan.primary_kind;
    free_plan(plan);
    return kind;
}

// The explicit backend request matching the test machine's GPU, or AUTO when
// the kind has no dedicated request value (e.g. an unrecognized GPU).
transcribe_backend_request explicit_request_for(ggml_backend_dev_t dev) {
    switch (transcribe::classify_device(dev)) {
        case transcribe::BackendKind::Metal:
            return TRANSCRIBE_BACKEND_METAL;
        case transcribe::BackendKind::Vulkan:
            return TRANSCRIBE_BACKEND_VULKAN;
        case transcribe::BackendKind::Cuda:
            return TRANSCRIBE_BACKEND_CUDA;
        case transcribe::BackendKind::Rocm:
            return TRANSCRIBE_BACKEND_ROCM;
        default:
            return TRANSCRIBE_BACKEND_AUTO;
    }
}

// The hook feeds the identity into the same input the selector uses.
void test_hook_shapes_deny_input() {
    set_env("TRANSCRIBE_TEST_GPU_HW_IDS", "1002:164e:2");
    const transcribe::GpuDenyInput in = transcribe::gpu_deny_input(g_gpu);
    CHECK(in.dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU);
    CHECK(in.hw.vendor_id == 0x1002);
    CHECK(in.hw.device_id == 0x164e);
    CHECK(in.hw.shader_core_count == 2);
    CHECK(transcribe::gpu_auto_deny_reason(in) != nullptr);

    // A malformed hook is ignored, never trusted.
    set_env("TRANSCRIBE_TEST_GPU_HW_IDS", "garbage");
    const transcribe::GpuDenyInput bad = transcribe::gpu_deny_input(g_gpu);
    CHECK(bad.hw.vendor_id != 0x1002);
    clear_hooks();
}

// The fix: AUTO skips the denylisted GPU and lands on CPU.
void test_auto_skips_denylisted_gpu() {
    set_env("TRANSCRIBE_TEST_GPU_HW_IDS", kDenied);
    CHECK(auto_kind() == transcribe::BackendKind::Cpu);
    clear_hooks();
}

// An eligible identity changes nothing.
void test_auto_keeps_eligible_gpu() {
    set_env("TRANSCRIBE_TEST_GPU_HW_IDS", kEligible);
    CHECK(auto_kind() == g_baseline);
    clear_hooks();
}

// The kill switch restores pre-denylist behavior.
void test_kill_switch_restores_baseline() {
    set_env("TRANSCRIBE_TEST_GPU_HW_IDS", kDenied);
    set_env("TRANSCRIBE_NO_GPU_DENYLIST", "1");
    CHECK(auto_kind() == g_baseline);
    clear_hooks();
}

// An explicit backend request is an override: the device is honored.
void test_explicit_backend_is_honored() {
    const transcribe_backend_request req = explicit_request_for(g_gpu);
    if (req == TRANSCRIBE_BACKEND_AUTO) {
        return;
    }
    set_env("TRANSCRIBE_TEST_GPU_HW_IDS", kDenied);
    transcribe::BackendPlan plan;
    const transcribe_status st = transcribe::load_common::init_backends(req, nullptr, "test", plan);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(plan.primary_kind == transcribe::classify_device(g_gpu));
    free_plan(plan);
    clear_hooks();
}

// An explicit device handle is an override too.
void test_explicit_device_is_honored() {
    set_env("TRANSCRIBE_TEST_GPU_HW_IDS", kDenied);
    transcribe::BackendPlan plan;
    const transcribe_status st = transcribe::load_common::init_backends(
        TRANSCRIBE_BACKEND_AUTO, reinterpret_cast<transcribe_device_t>(g_gpu), "test", plan);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(plan.primary_kind == transcribe::classify_device(g_gpu));
    free_plan(plan);
    clear_hooks();
}

}  // namespace

int main() {
    transcribe_init_backends_default();
    clear_hooks();

    g_gpu = find_gpu_device();
    if (g_gpu == nullptr) {
        std::printf("ok (no GPU device; gate not exercised)\n");
        return EXIT_SUCCESS;
    }
    g_baseline = auto_kind();

    test_hook_shapes_deny_input();
    test_auto_skips_denylisted_gpu();
    test_auto_keeps_eligible_gpu();
    test_kill_switch_restores_baseline();
    test_explicit_backend_is_honored();
    test_explicit_device_is_honored();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("ok\n");
    return EXIT_SUCCESS;
}
