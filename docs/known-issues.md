# Known issues

Tracked bugs that are understood but not yet fully fixed, with enough detail to
reproduce and resume.

---

## W1 — ggml's non-OpenMP CPU threadpool barrier deadlocks under MSVC (Windows)

**Severity:** medium. Multi-threaded CPU compute on Windows/MSVC hangs unless
ggml is built with OpenMP. Worked around per consumer (see below); a proper
root-cause fix is deferred to a session with a real Windows machine.

**Symptom.** With `GGML_OPENMP=OFF`, a CPU whisper run on Windows/MSVC wedges:
all ggml-cpu worker threads spin forever in `ggml_barrier`
(`ggml/src/ggml-cpu/ggml-cpu.c`, the custom non-OpenMP `#else` path). Reproduces
on BOTH the static and the shared (wheel-posture) builds — it is a
near-deterministic race, hanging ~12/13 runs (≈always), occasionally winning.
A thread dump shows the main thread parked in one barrier site while the
secondary threads sit at a *different* barrier site, i.e. one thread is a full
barrier ahead of the others — an "impossible" state for a correct barrier, so a
thread is escaping a barrier early.

**Platform signature.** MSVC-only. The same code with `GGML_OPENMP=OFF` works on
Linux/macOS (GCC/Clang). The OpenMP `#pragma omp barrier` path
(`GGML_OPENMP=ON`) does not deadlock on any platform.

**What is NOT the cause (ruled out from static analysis).** The MSVC atomic shim
(`ggml-cpu.c`, `#if defined(_MSC_VER) && !defined(__clang__)`) implements every
atomic op as a full-barrier `Interlocked*` (`InterlockedCompareExchange` for
load, `InterlockedExchangeAdd` for fetch_add, `InterlockedExchange` for store),
so it is NOT a weak-memory ordering bug — the ordering is sequentially
consistent, stronger than the `memory_order_relaxed` the C says. The barrier
logic is the standard sense-reversing barrier and reads correct on paper. Why a
thread still escapes early is unknown without live inspection.

**Current handling (per consumer).**
- **Rust binding:** forces `GGML_OPENMP=ON` on Windows/MSVC (`bindings/rust/sys/
  build.rs`), honored by the `GGML_OPENMP` guard in the root `CMakeLists.txt`.
  MSVC auto-links `vcomp`; no `-fopenmp` reaches the link manifest. Validated
  green on the Windows CI legs. This is correct and necessary — the static Rust
  default posture deadlocks without it.
- **Python wheels:** stay OpenMP-free (numpy/torch OpenMP-coexistence hygiene)
  and default to the Vulkan backend, so normal use does not hit the CPU
  threadpool. CPU-backend compute on Windows is the documented limitation: it
  hangs. (Confirmed via a ctypes repro on the wheel posture; faulthandler shows
  the main thread parked in the native `transcribe_run`.)

**Related gap (ours, fixable independently).** Whisper never calls
`ggml_backend_set_n_threads` — unlike cohere/parakeet/gigaam/granite — so its
encoder/decoder graph runs at the ggml default of 4 threads regardless of the
session `n_threads`. Teaching whisper to set the backend thread count is worth
doing on its own, and it is the enabler for the single-threaded mitigation below
(barrier early-returns when `n_threads == 1`, so single-threaded CPU has no
barrier and cannot deadlock).

**Mitigation options (a decision for when this is picked up).**
- **(a) OpenMP for the wheels** — fixes it, but reintroduces the numpy/torch
  OpenMP-runtime coexistence hazard the wheels deliberately avoid.
- **(b) Single-threaded CPU on Windows-without-OpenMP** — make whisper (and any
  family missing it) set `ggml_backend_set_n_threads`, default to 1 thread on
  the OpenMP-free Windows posture. No barrier, no OpenMP runtime; perf cost
  (single-threaded CPU). Keeps wheel hygiene intact.
- **(c) Patch ggml's vendored barrier** — riskiest; needs the root cause first.

**Debugging plan (needs a Windows machine; do not burn CI cycles on this).**
1. Reproduce locally: build the SHARED lib `-DTRANSCRIBE_BUILD_SHARED=ON
   -DTRANSCRIBE_USE_OPENMP=OFF` (or static), point the ctypes binding at it via
   `TRANSCRIBE_LIBRARY`, and run a CPU whisper transcription
   (`Model(path, backend="cpu")`). The throwaway repro is in git history at the
   commit that removed `.github/diag/py_cpu_repro.py` / `diag-py-cpu.yml`.
2. Attach a debugger (WinDbg / Visual Studio) to the hung process; break in and
   inspect the threadpool: `tp->n_barrier`, `tp->n_barrier_passed`,
   `tp->n_graph & GGML_THREADPOOL_N_THREADS_MASK` (the live thread count), and
   each worker's `ith` and its stuck barrier site. That distinguishes a
   thread-count mismatch (n_graph wrong) from a generation skew (a thread
   captured a stale `n_passed`).
3. Check whether a single-threaded run (`n_threads == 1`, barrier disabled) is
   clean — confirms mitigation (b) is viable.
4. Diff our vendored ggml (0.9.8) against upstream around `ggml_barrier` /
   `ggml_graph_compute_thread` — a newer ggml may already fix the MSVC
   threadpool, in which case a targeted backport beats the workarounds.
