// backend_gpu_denylist_unit.cpp - unit tests for the AUTO GPU denylist policy.
//
// gpu_auto_deny_reason is a pure function of (device type, vendor id,
// device id, shader core count), so the policy is pinned here against
// literal PCI ids without any GPU present. The generated id table
// (src/transcribe-gpu-denylist-data.h) is exercised through it.
//
// The runtime plumbing (proc-address query, AUTO skip vs explicit honor)
// is covered by backend_gpu_denylist_gate_unit.

#include "transcribe-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int g_failures = 0;

constexpr uint32_t kIntel  = 0x8086;
constexpr uint32_t kAmd    = 0x1002;
constexpr uint32_t kNvidia = 0x10de;
constexpr uint32_t kApple  = 0x106b;

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

transcribe::GpuDenyInput make(enum ggml_backend_dev_type type, uint32_t vendor, uint32_t device, uint32_t cores = 0) {
    transcribe::GpuDenyInput in;
    in.dev_type             = type;
    in.hw.vendor_id         = vendor;
    in.hw.device_id         = device;
    in.hw.shader_core_count = cores;
    return in;
}

void expect_denied(const char * what, const transcribe::GpuDenyInput & in, const char * reason_substr) {
    const char * why = transcribe::gpu_auto_deny_reason(in);
    if (why == nullptr) {
        std::fprintf(stderr, "FAIL: %s (%04x:%04x, %u cores) should be denied\n", what, in.hw.vendor_id,
                     in.hw.device_id, in.hw.shader_core_count);
        ++g_failures;
        return;
    }
    if (std::strstr(why, reason_substr) == nullptr) {
        std::fprintf(stderr, "FAIL: %s denied with \"%s\", expected it to mention \"%s\"\n", what, why, reason_substr);
        ++g_failures;
    }
}

void expect_allowed(const char * what, const transcribe::GpuDenyInput & in) {
    const char * why = transcribe::gpu_auto_deny_reason(in);
    if (why != nullptr) {
        std::fprintf(stderr, "FAIL: %s (%04x:%04x, %u cores) should be allowed, got \"%s\"\n", what, in.hw.vendor_id,
                     in.hw.device_id, in.hw.shader_core_count, why);
        ++g_failures;
    }
}

}  // namespace

int main() {
    constexpr auto IGPU = GGML_BACKEND_DEVICE_TYPE_IGPU;
    constexpr auto GPU  = GGML_BACKEND_DEVICE_TYPE_GPU;
    constexpr auto CPU  = GGML_BACKEND_DEVICE_TYPE_CPU;

    // --- Intel: pre-Xe integrated parts are denied by device id -------------
    expect_denied("UHD Graphics 620 (KBL GT2, Handy #1884)", make(IGPU, kIntel, 0x5917), "pre-Xe");
    expect_denied("HD Graphics 4600 (HSW GT2)", make(IGPU, kIntel, 0x0416), "pre-Xe");
    expect_denied("HD Graphics 530 (SKL GT2)", make(IGPU, kIntel, 0x1912), "pre-Xe");
    expect_denied("UHD Graphics 630 (CFL GT2)", make(IGPU, kIntel, 0x3e9b), "pre-Xe");
    expect_denied("UHD Graphics 600 (GLK)", make(IGPU, kIntel, 0x3185), "pre-Xe");
    expect_denied("Iris Plus Graphics (ICL GT2, Gen 11)", make(IGPU, kIntel, 0x8a52), "pre-Xe");
    expect_denied("UHD Graphics (JSL, Gen 11)", make(IGPU, kIntel, 0x4e71), "pre-Xe");

    // --- Intel: Xe-LP and newer are eligible ---------------------------------
    expect_allowed("Iris Xe Graphics (TGL GT2)", make(IGPU, kIntel, 0x9a49));
    expect_allowed("UHD Graphics 770 (ADL-S GT1)", make(IGPU, kIntel, 0x4680));
    expect_allowed("UHD Graphics (ADL-N)", make(IGPU, kIntel, 0x46d0));
    expect_allowed("Arc Graphics (MTL)", make(IGPU, kIntel, 0x7d55));
    expect_allowed("Arc 140V (LNL)", make(IGPU, kIntel, 0x64a0));
    expect_allowed("Arc A770 (DG2, discrete)", make(GPU, kIntel, 0x56a0));

    // --- AMD: GCN 1-3 APUs denied by id; Vega and RDNA APUs eligible ---------
    expect_denied("Radeon R7 Graphics (Kaveri)", make(IGPU, kAmd, 0x1313), "GCN 1-3");
    expect_denied("Radeon R4/R5 Graphics (Kabini)", make(IGPU, kAmd, 0x9830), "GCN 1-3");
    expect_denied("Radeon R7 Graphics (Carrizo)", make(IGPU, kAmd, 0x9874), "GCN 1-3");
    expect_denied("Radeon R2/R5 Graphics (Stoney Ridge)", make(IGPU, kAmd, 0x98e4), "GCN 1-3");
    expect_allowed("Radeon Vega 8 (Raven Ridge)", make(IGPU, kAmd, 0x15dd, 8));
    expect_allowed("Radeon Graphics (Renoir, Vega 8)", make(IGPU, kAmd, 0x1636, 8));
    expect_allowed("Radeon 680M (Rembrandt, 12 CU)", make(IGPU, kAmd, 0x1681, 12));
    expect_allowed("Radeon 780M (Phoenix, 12 CU)", make(IGPU, kAmd, 0x15bf, 12));
    expect_allowed("Radeon 890M (Strix Point, 16 CU)", make(IGPU, kAmd, 0x150e, 16));

    // --- AMD: the 1-2 CU display stubs are denied by core count, no table ----
    expect_denied("Raphael iGPU (Ryzen 7000 desktop, 2 CU)", make(IGPU, kAmd, 0x164e, 2), "compute units");
    expect_denied("Granite Ridge iGPU (Ryzen 9000 desktop, 2 CU)", make(IGPU, kAmd, 0x13c0, 2), "compute units");
    expect_denied("Mendocino 610M (2 CU)", make(IGPU, kAmd, 0x1506, 2), "compute units");
    expect_allowed("Raven2 Vega 3 (3 CU) stays above the stub threshold", make(IGPU, kAmd, 0x15d8, 3));
    expect_allowed("unknown AMD iGPU with unknown core count", make(IGPU, kAmd, 0xffff, 0));

    // --- Scope: only integrated devices with a known vendor -----------------
    expect_allowed("a denylisted id reported as a discrete GPU", make(GPU, kIntel, 0x5917));
    expect_allowed("a denylisted id on a CPU-type device", make(CPU, kIntel, 0x5917));
    expect_allowed("NVIDIA integrated GPU", make(IGPU, kNvidia, 0x2b00, 48));
    expect_allowed("Apple via MoltenVK", make(IGPU, kApple, 0x0000, 0));
    expect_allowed("no hardware identity at all", make(IGPU, 0, 0));
    expect_allowed("Intel id outside 16-bit range", make(IGPU, kIntel, 0x10005917));

    // --- Kill switch reads the documented flag convention ------------------
    unset_env("TRANSCRIBE_NO_GPU_DENYLIST");
    if (transcribe::gpu_denylist_disabled()) {
        std::fprintf(stderr, "FAIL: denylist reported disabled with the env var unset\n");
        ++g_failures;
    }
    set_env("TRANSCRIBE_NO_GPU_DENYLIST", "1");
    if (!transcribe::gpu_denylist_disabled()) {
        std::fprintf(stderr, "FAIL: TRANSCRIBE_NO_GPU_DENYLIST=1 did not disable the denylist\n");
        ++g_failures;
    }
    set_env("TRANSCRIBE_NO_GPU_DENYLIST", "0");
    if (transcribe::gpu_denylist_disabled()) {
        std::fprintf(stderr, "FAIL: TRANSCRIBE_NO_GPU_DENYLIST=0 should leave the denylist enabled\n");
        ++g_failures;
    }
    unset_env("TRANSCRIBE_NO_GPU_DENYLIST");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("ok\n");
    return EXIT_SUCCESS;
}
