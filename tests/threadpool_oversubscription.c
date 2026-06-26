// threadpool_oversubscription.c - regression test for the ggml CPU (non-OpenMP)
// threadpool barrier deadlock under CPU oversubscription.
//
// No model, no external data, no inference engine — just a small graph computed
// repeatedly with more compute threads than available logical CPUs.
//
// The bug this pins: ggml_thread_cpu_relax() emits a real `pause` only for
// __x86_64__ (a GCC/Clang macro). MSVC defines _M_X64, not __x86_64__, so on
// MSVC the spin body was an empty no-op and ggml_barrier()'s waiters busy-spun
// without ever yielding the core. When n_threads > available logical CPUs, the
// spinning waiters starve a worker that has not yet reached the barrier, so the
// barrier never releases and ggml_graph_compute() hangs forever. The fix
// (ggml-cpu.c) adds an MSVC relax and a bounded-spin-then-yield in ggml_barrier.
//
// Only a real test on the custom (non-OpenMP) barrier: a GGML_OPENMP=ON build
// uses `#pragma omp barrier`, which has no such defect, so this passes trivially
// there. It is load-bearing in the GGML_OPENMP=OFF lanes (the wheels).
//
// To be reproducible on any multi-core dev box (not just a 2-vCPU CI runner),
// this pins the process to 2 logical CPUs so the default n_threads(=4)
// oversubscribes. A watchdog turns a hang into a test FAILURE (exit code 2)
// instead of an indefinite hang.

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
static DWORD WINAPI watchdog_main(LPVOID arg) {
    DWORD timeout_ms = (DWORD)(uintptr_t)arg;
    Sleep(timeout_ms);
    fprintf(stderr,
        "DEADLOCK: ggml_graph_compute did not finish within %lu ms "
        "(ggml_barrier livelock under CPU oversubscription)\n", timeout_ms);
    fflush(stderr);
    ExitProcess(2);
    return 0;
}
#endif

int main(int argc, char ** argv) {
    const int n_threads = argc > 1 ? atoi(argv[1]) : 4;    // > available CPUs below
    const int n_iters   = argc > 2 ? atoi(argv[2]) : 50;   // each does many barriers
    const int chain     = argc > 3 ? atoi(argv[3]) : 64;   // graph nodes per compute

#if defined(_WIN32)
    // Force oversubscription: restrict the process to 2 logical CPUs, so the
    // default n_threads=4 has more threads than cores (what a 2-vCPU CI box has
    // naturally). Without this the bug only shows on machines with few cores.
    if (!SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)0x3)) {
        fprintf(stderr, "warning: SetProcessAffinityMask failed (%lu)\n", GetLastError());
    }
    // 15s watchdog -> converts a hang into exit code 2 instead of hanging CI.
    CreateThread(NULL, 0, watchdog_main, (LPVOID)(uintptr_t)15000, 0, NULL);
#endif

    struct ggml_init_params params = {
        /*.mem_size   =*/ (size_t)512 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

    const int n = 1024;
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    for (int i = 0; i < n; i++) {
        ((float *) a->data)[i] = 1.0f;
        ((float *) b->data)[i] = 2.0f;
    }

    // Long chain of element-wise ops -> many graph nodes -> many ggml_barrier()
    // calls per compute, so the race is hit near-deterministically.
    struct ggml_tensor * cur = a;
    for (int i = 0; i < chain; i++) {
        cur = ggml_add(ctx, cur, b);
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    printf("n_threads=%d iters=%d chain=%d (process pinned to 2 logical CPUs)\n",
           n_threads, n_iters, chain);
    fflush(stdout);

    for (int it = 0; it < n_iters; it++) {
        enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ggml_graph_compute failed at iter %d (status %d)\n", it, st);
            return 1;
        }
        if ((it % 10) == 0) { printf("  iter %d ok\n", it); fflush(stdout); }
    }

    printf("PASS: %d computes completed, no barrier deadlock\n", n_iters);
    ggml_free(ctx);
    return 0;
}
