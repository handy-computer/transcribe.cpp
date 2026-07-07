// backend_metal_simdgroup_gate_unit.cpp - AUTO device selection skips a Metal
// device that lacks simdgroup matrix multiply.
//
// A pre-Apple7 Metal GPU (Intel integrated / AMD on Intel Macs) runs
// whisper-class matmul graphs through vector fallback kernels that silently
// produce garbage ("!!!!!") at ~0.01x realtime (Handy issue #1608). The loader
// gates it via metal_backend_lacks_simdgroup_mm; here we force that verdict
// with the TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM hook and pin the behavior:
//
//   AUTO           -> skip the gated Metal device and fall back to CPU.
//   explicit METAL -> honor the device the caller named (warn only).
//   hook unset / non-matching -> selection matches the real hardware verdict.
//
// The baselines can't assume "Metal always wins": on an actual pre-Apple7
// Intel Mac the real gate fires with the hook unset, so AUTO lands on CPU
// there. main() probes the hardware verdict once (hook cleared) and the
// baselines assert against it.
//
// Every case is a no-op on machines with no Metal device (e.g. Linux CI), so
// the test passes trivially where the gate can't apply.

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

void free_plan(transcribe::BackendPlan & plan) {
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        ggml_backend_free(*it);
    }
    plan = transcribe::BackendPlan{};
}

// The real hardware verdict for the first Metal device, probed once in
// main() with the hook cleared. On simdgroup-mm hardware (Apple Silicon)
// an unhooked AUTO must pick Metal; on gated hardware (pre-Apple7 GPUs)
// the real gate fires and AUTO must land on CPU — the same outcome the
// forced-hook case pins.
enum class MetalProbe { None, HasSimdgroupMM, LacksSimdgroupMM };

MetalProbe g_probe = MetalProbe::None;

MetalProbe probe_metal_device() {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (transcribe::classify_device(dev) != transcribe::BackendKind::Metal) {
            continue;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (be == nullptr) {
            // A Metal device that can't init never reaches the gate; the
            // loader's AUTO probe skips it the same way. Treat as absent.
            return MetalProbe::None;
        }
        const bool lacks = transcribe::load_common::metal_backend_lacks_simdgroup_mm(be, dev);
        ggml_backend_free(be);
        return lacks ? MetalProbe::LacksSimdgroupMM : MetalProbe::HasSimdgroupMM;
    }
    return MetalProbe::None;
}

transcribe::BackendKind expected_auto_kind() {
    return g_probe == MetalProbe::LacksSimdgroupMM ? transcribe::BackendKind::Cpu : transcribe::BackendKind::Metal;
}

// Baseline: with no hook, AUTO selection matches the hardware verdict.
void test_hook_unset_matches_hardware_verdict() {
    if (g_probe == MetalProbe::None) {
        return;
    }
    unset_env("TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM");
    transcribe::BackendPlan plan;
    CHECK(transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, 0, "test", plan) == TRANSCRIBE_OK);
    CHECK(plan.primary_kind == expected_auto_kind());
    free_plan(plan);
}

// A non-matching hook value leaves selection unchanged.
void test_nonmatching_hook_is_inert() {
    if (g_probe == MetalProbe::None) {
        return;
    }
    set_env("TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM", "no-such-device-name-xyzzy");
    transcribe::BackendPlan plan;
    CHECK(transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, 0, "test", plan) == TRANSCRIBE_OK);
    CHECK(plan.primary_kind == expected_auto_kind());
    free_plan(plan);
    unset_env("TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM");
}

// The fix: AUTO skips a simdgroup-mm-less Metal device and lands on CPU.
void test_auto_skips_gated_metal_and_falls_back_to_cpu() {
    if (g_probe == MetalProbe::None) {
        return;
    }
    set_env("TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM", "*");
    transcribe::BackendPlan plan;
    const transcribe_status st = transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_AUTO, 0, "test", plan);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(plan.primary != nullptr);
    CHECK(plan.primary_kind == transcribe::BackendKind::Cpu);
    free_plan(plan);
    unset_env("TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM");
}

// An explicit Metal request is an override: the gated device is honored
// (the loader warns) rather than turning into a hard failure.
void test_explicit_metal_is_honored_despite_gate() {
    if (g_probe == MetalProbe::None) {
        return;
    }
    set_env("TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM", "*");
    transcribe::BackendPlan plan;
    const transcribe_status st = transcribe::load_common::init_backends(TRANSCRIBE_BACKEND_METAL, 0, "test", plan);
    CHECK(st == TRANSCRIBE_OK);
    CHECK(plan.primary_kind == transcribe::BackendKind::Metal);
    free_plan(plan);
    unset_env("TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM");
}

}  // namespace

int main() {
    transcribe_init_backends_default();

    // The hook may leak in from the caller's environment; clear it so the
    // probe and the baselines see the real hardware verdict.
    unset_env("TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM");
    g_probe = probe_metal_device();

    test_hook_unset_matches_hardware_verdict();
    test_nonmatching_hook_is_inert();
    test_auto_skips_gated_metal_and_falls_back_to_cpu();
    test_explicit_metal_is_honored_despite_gate();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
