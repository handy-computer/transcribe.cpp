# Known issues

Tracked bugs that are understood but not yet fixed, with enough detail to
reproduce and fix. Surfaced (mostly) by the first cargo-driven MSVC build of
the tree during the Rust-bindings Windows bringup (2026-06-13).

---

## B — whisper in-flight cancel: native integer divide-by-zero (Windows/MSVC)

**Severity:** medium (crashes a *cancelled* whisper run on Windows; normal
(uncancelled) runs transcribe fine on Windows, and cancellation works on
linux/macos).

**Symptom.** Cancelling an in-flight whisper run (from another thread, mid-run)
crashes with an integer divide-by-zero — `STATUS_INTEGER_DIVIDE_BY_ZERO`
(`0xC0000094`) — on **Windows/MSVC**. Reproduces for both short-form (~22 s) and
long-form (~132 s) audio, so it is in the general abort/partial path, not the
long-form chunk loop.

**Why it's Windows-specific.** It is a genuine integer `÷0` (or the equivalent
`INT_MIN / -1`) in whisper's abort/partial path, surfaced only by a particular
combination:

- **Windows/MSVC only** — the *same* in-flight cancel passes on linux x86
  (Blacksmith `rust-build (linux-dylib)`) and is masked on arm64 (ARM integer
  `÷0` returns 0). So it is **not** generic-x86; it is MSVC codegen + the slow
  2-vcpu Windows runner reliably landing the abort in the danger window.
- The leading hypothesis is therefore an **uninitialized / conditionally-set
  integer divisor** that the early-return abort path skips initializing —
  clang/gcc happen to leave the stack slot nonzero, MSVC leaves it 0. UBSan
  (`-fsanitize=undefined`) does NOT catch uninitialized reads (that is MSan,
  Linux+clang-only, needing an instrumented libc++), which is why a local UBSan
  threaded-cancel sweep on arm64 reproduces nothing.
- The sanitized C++ lane (`cpp-tests-sanitized`) never catches it: it runs
  model-less, so its abort tests never load whisper.

**Where (localized, not line-pinpointed).** Whisper's run/abort path in
`src/arch/whisper/model.cpp` (the short-form batch path at ~3155 and the
long-form serial fallback at ~3601 both reproduce, so the `÷0` is on the shared
abort/partial path). Every integer division audited there is either guarded
(`mel_us / valid_count` with `valid_count = std::max(1, …)`) or a `double`
division (`no_speech_prob`, `avg_logprob`, `compression_ratio` — all guarded);
none is the offending `÷0`, which (with the MSVC-only signature) points at an
**uninitialized / conditionally-set integer divisor** the early-return abort
path skips initializing, not a missing guard. UBSan does not catch uninitialized
reads (that is MSan, Linux+clang-only, needs an instrumented libc++), so the
exact `file:line` needs the faulting address from an **MSVC** build.

**Reproduce.** On Windows/MSVC: the Rust `cancel` test reproduces it directly
(it is why that test is skipped on Windows). For a backtrace, build the C++
`transcribe_whisper_e2e_smoke` (with `TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON` +
RelWithDebInfo + a whisper GGUF in `TRANSCRIBE_WHISPER_MODEL`), set
`TRANSCRIBE_DIAG_CANCEL=1` (the env-gated threaded-cancel harness in
`tests/whisper_e2e_smoke.cpp`), and run it under `cdb` to catch `0xC0000094` and
print the stack. NOTE: that C++ test does not build on MSVC yet without working
around **F** below (define `TRANSCRIBE_BUILD` for the consumer, or fix F); the
earlier `diag-cancel` workflow (git history) is the scaffold.

**Workaround in place.** The Rust in-flight-cancel test
(`bindings/rust/transcribe-cpp/tests/cancel.rs::cross_thread_cancel_of_in_flight_run`)
is **skipped on Windows** (`cfg!(target_os = "windows")`); the binding's
cancellation contract (`CancelToken` → `Error::Aborted` + partial) is still
covered on linux/macos, and the uncancelled-run test runs everywhere.

---

## F — public header forces `dllimport` on Windows static consumers

**Severity:** low (only static **C++** consumers of the header on Windows; the
Rust and Python bindings are unaffected).

**Symptom.** A C++ translation unit that includes `include/transcribe.h` and
links the **static** `transcribe.lib` on MSVC fails to link:
`LNK2019: unresolved external symbol __imp_transcribe_*`.

**Cause.** `transcribe.h` (~line 174) defines `TRANSCRIBE_API` as
`__declspec(dllexport)` when `TRANSCRIBE_BUILD` is set (building the lib) and
`__declspec(dllimport)` otherwise (any consumer). There is no
`TRANSCRIBE_STATIC` escape hatch, so a consumer linking the static archive still
gets `dllimport` and the linker looks for `__imp_*` thunks that a static lib
doesn't provide. The Rust binding sidesteps this because its bindgen FFI has no
`__declspec`; native-ci never hit it because the C++ tests build only on
linux/macos.

**Fix sketch.** Add a `TRANSCRIBE_STATIC` guard:
`#if defined(_WIN32) && !defined(TRANSCRIBE_STATIC)` around the dllexport/import
block (else empty), and have the static `transcribe` target propagate it to
consumers: `if(NOT TRANSCRIBE_BUILD_SHARED) target_compile_definitions(transcribe
PUBLIC TRANSCRIBE_STATIC)`. (As a build-only workaround, defining
`TRANSCRIBE_BUILD` for the consumer also links, since both sides then use the
plain dllexport symbol.)
