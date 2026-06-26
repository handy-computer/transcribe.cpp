// thread_default_unit.cpp - white-box tests for the CPU thread-count helpers
// in transcribe-batch-util (default_n_threads / configure_sched_n_threads).
//
// The bug these pin: the per-arch default used std::thread::hardware_concurrency(),
// which reports the host's total cores and ignores the process CPU affinity
// mask. Under a constrained scheduler (taskset / cpuset / a CI runner pinned to
// N vCPUs) that over-counts, and the resulting oversubscription makes ggml's
// spin-wait barriers livelock. default_n_threads() counts the CPUs the process
// may actually run on instead. So the load-bearing assertion here is the
// affinity one (Linux): pin to K CPUs, expect K.
//
// configure_sched_n_threads() is exercised on the null-scheduler path, which
// isolates its resolution contract (requested>0 passes through; requested<=0
// falls back to default_n_threads()) without standing up a real backend graph —
// the real per-backend application is covered by the e2e/example runs.

#include "transcribe-batch-util.h"

#include <cstdio>

#if defined(__linux__)
#include <sched.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX  // else <windows.h>'s min/max macros clobber std::min/std::max
#define NOMINMAX
#endif
#include <windows.h>
#endif

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

// default_n_threads(): cap and range, independent of platform.
void test_default_range_and_cap() {
    const int uncapped = transcribe::default_n_threads(/*cap=*/0);
    CHECK(uncapped >= 1);                       // always at least one thread

    const int capped8 = transcribe::default_n_threads(8);
    CHECK(capped8 >= 1);
    CHECK(capped8 <= 8);                        // cap is honored
    CHECK(capped8 == (uncapped < 8 ? uncapped : 8));  // cap = min(uncapped, 8)

    CHECK(transcribe::default_n_threads(1) == 1);     // cap of 1 pins to 1
    const int cap4 = transcribe::default_n_threads(4);
    CHECK(cap4 >= 1 && cap4 <= 4);
}

// configure_sched_n_threads(nullptr, ...): resolution contract, no backends.
void test_configure_resolution_null_sched() {
    // requested > 0 passes through verbatim.
    CHECK(transcribe::configure_sched_n_threads(nullptr, 3) == 3);
    CHECK(transcribe::configure_sched_n_threads(nullptr, 1) == 1);
    // requested <= 0 falls back to default_n_threads().
    const int def = transcribe::default_n_threads();
    CHECK(transcribe::configure_sched_n_threads(nullptr, 0) == def);
    CHECK(transcribe::configure_sched_n_threads(nullptr, -5) == def);
}

#if defined(__linux__)
// The actual bug: default_n_threads() must honor the process affinity mask,
// not the host core count. Pin to K CPUs and expect K. Skips gracefully if the
// environment forbids changing affinity (some sandboxes do).
void test_affinity_honored_linux() {
    cpu_set_t original;
    CPU_ZERO(&original);
    if (sched_getaffinity(0, sizeof(original), &original) != 0) {
        std::fprintf(stderr, "SKIP affinity test: sched_getaffinity failed\n");
        return;
    }
    const int avail = CPU_COUNT(&original);

    // Collect the CPU ids currently allowed, so we pin to real, permitted CPUs.
    int cpus[CPU_SETSIZE];
    int n_cpus = 0;
    for (int c = 0; c < CPU_SETSIZE && n_cpus < avail; ++c) {
        if (CPU_ISSET(c, &original)) cpus[n_cpus++] = c;
    }

    // Pin to exactly one CPU -> default_n_threads must be 1.
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpus[0], &one);
    if (sched_setaffinity(0, sizeof(one), &one) != 0) {
        std::fprintf(stderr, "SKIP affinity test: sched_setaffinity not permitted\n");
        return;
    }
    CHECK(transcribe::default_n_threads(8) == 1);
    CHECK(transcribe::default_n_threads(/*cap=*/0) == 1);  // uncapped still sees 1 usable

    // Pin to two CPUs -> expect 2 (only when the box actually has >= 2).
    if (n_cpus >= 2) {
        cpu_set_t two;
        CPU_ZERO(&two);
        CPU_SET(cpus[0], &two);
        CPU_SET(cpus[1], &two);
        if (sched_setaffinity(0, sizeof(two), &two) == 0) {
            CHECK(transcribe::default_n_threads(8) == 2);
        }
    }

    // Restore the process to its original affinity.
    CHECK(sched_setaffinity(0, sizeof(original), &original) == 0);
}
#elif defined(_WIN32)
// Windows analog: GetProcessAffinityMask must drive the default, not the host
// core count. Pin the process to K CPUs via SetProcessAffinityMask and expect K.
// Skips gracefully if affinity changes aren't permitted.
void test_affinity_honored_windows() {
    const HANDLE proc = GetCurrentProcess();
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    if (!GetProcessAffinityMask(proc, &proc_mask, &sys_mask) || proc_mask == 0) {
        std::fprintf(stderr, "SKIP affinity test: GetProcessAffinityMask failed\n");
        return;
    }
    const DWORD_PTR original = proc_mask;

    // Lowest set bit = one permitted CPU. Pin to it -> expect 1.
    const DWORD_PTR one = original & (~original + 1);
    if (!SetProcessAffinityMask(proc, one)) {
        std::fprintf(stderr, "SKIP affinity test: SetProcessAffinityMask not permitted\n");
        return;
    }
    CHECK(transcribe::default_n_threads(8) == 1);
    CHECK(transcribe::default_n_threads(/*cap=*/0) == 1);

    // Pin to two CPUs (lowest two set bits) -> expect 2, when available.
    const DWORD_PTR second = (original & ~one) & (~(original & ~one) + 1);
    if (second != 0) {
        if (SetProcessAffinityMask(proc, one | second)) {
            CHECK(transcribe::default_n_threads(8) == 2);
        }
    }

    // Restore the process to its original affinity.
    CHECK(SetProcessAffinityMask(proc, original) != 0);
}
#endif

}  // namespace

int main() {
    test_default_range_and_cap();
    test_configure_resolution_null_sched();
#if defined(__linux__)
    test_affinity_honored_linux();
#elif defined(_WIN32)
    test_affinity_honored_windows();
#endif

    if (g_failures != 0) {
        std::fprintf(stderr, "thread_default_unit: %d failure(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "thread_default_unit: OK\n");
    return 0;
}
