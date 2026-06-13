# Known issues

Tracked bugs that are understood but not yet fixed, with enough detail to
reproduce and fix. Surfaced (mostly) by the first cargo-driven MSVC build of
the tree during the Rust-bindings Windows bringup (2026-06-13).

---

## B — whisper long-form cancel: native integer divide-by-zero (x86)

**Severity:** medium (crashes a *cancelled long-form* whisper run on x86; normal
runs and short-form cancels are unaffected).

**Symptom.** Cancelling an in-flight **long-form** whisper run (audio > ~30 s, so
the chunked/serial long-form path) crashes with an integer divide-by-zero:
`STATUS_INTEGER_DIVIDE_BY_ZERO` (`0xC0000094`) on Windows; `SIGFPE` on Linux
x86 if the abort lands in the danger window.

**Why it hides.** It is a genuine integer `÷0` (or the equivalent `INT_MIN / -1`)
in whisper's abort/partial path that is only reached when the abort interrupts a
specific point of the compute:

- **arm64 masks it** — ARM integer division by zero returns 0 instead of
  faulting, so macOS never crashes (and the value is benign downstream).
- **Timing-gated** — a fast machine clears the danger window before a ~40 ms
  cancel lands, so it does *not* reproduce on a fast x86 box (the Blacksmith
  Linux runner passed; the slow 2-vcpu Windows runner faults every time).
- The sanitized C++ lane (`cpp-tests-sanitized`, ASan+UBSan) never catches it
  because that lane runs model-less, so its abort tests are short-form only.
  Local UBSan builds reproduce *nothing* on arm64 even with the faithful
  threaded-cancel sweep (the danger window is never entered there).

**Where (localized, not line-pinpointed).** Whisper's long-form
`whisper_run` abort path in `src/arch/whisper/model.cpp` (the serial fallback at
~3601 calls the single-utterance long-form `whisper_run` at ~1300, whose seek
loop is at ~1800). Every integer division audited in that path is either guarded
(`mel_us / valid_count` with `valid_count = std::max(1, …)`) or a `double`
division (`no_speech_prob`, `avg_logprob`, `compression_ratio` — all guarded);
none is the flaky `÷0`. The signature (reached only on a specific aborted-window
interleaving, x86-only, MSVC-and-timing-dependent) most strongly fits an
**uninitialized / conditionally-set integer divisor** that the early-return
abort path skips initializing — clang/gcc leave the stack slot nonzero, the slow
MSVC build leaves it 0. UBSan does not catch uninitialized reads (that is MSan,
Linux+clang-only, and needs an instrumented libc++).

**Reproduce.** Set `TRANSCRIBE_DIAG_CANCEL=1` and run the
`transcribe_whisper_e2e_smoke` test (built with `TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON`
+ a whisper GGUF in `TRANSCRIBE_WHISPER_MODEL`) on a **slow x86** machine under a
debugger — the env-gated harness in `tests/whisper_e2e_smoke.cpp` does a threaded
wall-clock cancel of a long-form run swept over delays. A Windows `cdb` backtrace
or a Linux `gdb`/core backtrace (RelWithDebInfo) gives the exact `file:line`;
the fix is then almost certainly a one-line guard or initializer. The earlier
`diag-cancel` workflow (git history) drove exactly this.

**Workaround in place.** The Rust `cancel` conformance test
(`bindings/rust/transcribe-cpp/tests/cancel.rs`) tiles the clip only enough to be
mid-flight while staying **short-form** (< 30 s), so it exercises the binding's
cancellation contract (`CancelToken` → `Error::Aborted` + partial) without
entering the long-form path that trips this bug.

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
