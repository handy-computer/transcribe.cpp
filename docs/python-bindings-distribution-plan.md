# Python Bindings and Distribution Plan

Direction for first-party Python bindings and native package distribution for
`transcribe.cpp`. The binding layer itself is implemented; this document now
records the settled decisions, the design context for the distribution work
that remains, and the punch list to a public release.

The short version:

- Bindings live in-tree at `bindings/python`; the public C API
  (`include/transcribe/extensions.h`) is the source of truth.
- Low-level FFI is generated `ctypes` (libclang), committed, drift-gated.
- A small handwritten Pythonic layer owns the user experience.
- Distribution is a pure-Python API package (`transcribe-cpp`) plus
  self-contained native provider wheels per platform/backend.
- An sdist that builds from source is the universal fallback.
- Vulkan ships **by default** in the Linux and Windows wheels, as a ggml
  dynamic backend module that degrades cleanly to CPU when no Vulkan
  loader/driver exists. Metal ships by default on macOS arm64. CUDA is an
  opt-in provider built and tested in the same cycle.
- Merge bar: this branch merges release-ready. Everything in the punch list
  is merge-blocking unless explicitly marked otherwise.

## Status: what is already built

Done and verified (see the `bindings/python/tests/` pytest suite, which
exercises all of it against a locally built shared library):

- **Generated FFI layer.** `_generate/generate.py` parses the public headers
  with libclang and emits `_generated.py`: structs with compiler-computed
  layouts, enum/macro constants, function prototypes, and per-struct ABI
  metadata. Committed; `--check` is the CI drift gate.
- **ABI verification at import.** `_abi.py` checks every generated struct's
  size/alignment/per-field offsets against the captured C-compiler layout,
  then size/alignment against the loaded library via the
  `transcribe_abi_struct_size/_align` C API. Mismatch raises before any
  struct is constructed.
- **High-level API.** `Model`, `Session`, `run`, `run_batch`, streaming
  (`Stream`, commit policies, text views), typed family extensions for all
  five families, log callback with GIL-safe trampoline, event-based
  cancellation, exception hierarchy mapped from `transcribe_status`. Results
  are fully materialized (no borrowed pointers escape); ctypes releases the
  GIL on every foreign call (verified by test).
- **C/CMake groundwork.** `TRANSCRIBE_VERSION_*` macros as the single version
  source (parsed by CMake; surfaced by `transcribe_version()/_commit()`;
  stamped as shared-lib VERSION/SOVERSION). Build switches
  `TRANSCRIBE_BUILD_SHARED`, `TRANSCRIBE_USE_OPENMP`,
  `TRANSCRIBE_USE_SYSTEM_BLAS`. Shared builds register only the public-ABI
  smoke (white-box suite requires the static build).
- **PyPI.** `transcribe-cpp` is reserved (0.0.0 placeholder published).
  Import package is `transcribe_cpp`.

## Settled decisions

Recorded so they are not relitigated; rationale condensed.

- **Bind the C API, not C++ internals.** Generated `ctypes` rather than
  pybind11/nanobind (handwritten glue drifts, and a compiled extension forces
  a Python-version × platform × backend wheel matrix) and rather than CFFI
  (kept as fallback only). ctypes call overhead is irrelevant at this call
  granularity — one FFI call per transcription run, not per token.
- **Pure-Python API package + native provider packages.** PyPI wheel tags
  cannot express backend choices, so `transcribe-cpp` stays pure Python and
  loads a package-local native artifact supplied by a provider package. The
  **default provider on Linux/Windows bundles CPU + Vulkan** as ggml backend
  modules (Vulkan degrades cleanly to CPU when no loader/driver exists —
  verified, see design context); macOS's default provider is Metal. Bigger
  or specialized backends are separate providers installed via extras
  (`pip install "transcribe-cpp[cu12]"`), which are additive — CUDA installs
  alongside the default provider, not instead of it.
- **One default provider distribution: `transcribe-cpp-native`** (decided
  2026-06-10). Its wheels are platform-tagged and differ in content
  (manylinux/win x86_64 → CPU+Vulkan modules; macOS arm64 → Metal); the
  descriptor's `backends` field tells the truth per platform, and the
  scikit-build-core sdist lives under the same name so "no prebuilt wheel"
  falls back to a source build automatically. Specialized providers keep
  suffixed names (`transcribe-cpp-native-cu12`). The pyproject for this
  package sits at the **repo root** — an sdist cannot reach files outside its
  project directory, and the sdist must carry the whole C++ tree.
- **`transcribe-cpp` hard-depends on `transcribe-cpp-native==X.Y.Z.*`**
  (decided 2026-06-10): `pip install transcribe-cpp` transcribes out of the
  box, which is what the clean-machine gate demands. The `.*` pin is the
  pre-1.0 base-version contract at resolver level; the import-time
  version/header-hash check stays as the runtime backstop, and
  `check_version_sync.py` gates the pin against `transcribe.h`. Dev/CI uv
  environments neutralize the dependency via `tool.uv.override-dependencies`
  with a provably-false marker (they test locally built libraries, and the
  provider isn't on an index until release).
- **Provider discovery via entry points**, group `transcribe_cpp.native`,
  not import-name probing. The entry-point contract: provider package name,
  native artifact path/directory, native library version, generated
  public-header hash, supported backend kinds. The API package hard-fails
  before loading a mismatched provider, with an actionable error. Pip pins
  are not enough; the runtime check is the backstop.
- **Provider selection ≠ backend request.** Provider selection picks which
  native artifact loads into the process; `backend=` maps to
  `transcribe_backend_request` inside it. Selection policy: explicit
  argument → `TRANSCRIBE_NATIVE_PROVIDER` env var → best accelerated
  (CUDA/Metal, then Vulkan) → CPU. An installed provider that cannot satisfy
  a backend request raises from the request path — a different error from
  "no such provider installed."
- **Versioning.** Pre-1.0, the Python package and native library must match
  on the *base* version exactly (post-releases like `0.0.1.post1` must still
  load). No version skew support until the API grows a caller-size-aware
  init convention.
- **Wheel hygiene.** Official provider wheels: `GGML_NATIVE=OFF`,
  OpenMP off, non-Apple system BLAS off (Accelerate is fine on macOS — it is
  a system framework, nothing to vendor). Reason: wheel-repair tools vendor
  OpenMP/BLAS runtimes that collide with PyTorch/NumPy/MKL in the same
  process; `KMP_DUPLICATE_LIB_OK` is not a packaging strategy. Opt-in
  OpenMP/BLAS providers only after co-import tests prove coexistence.
- **Linux baseline `manylinux_2_28`** (amended 2026-06-10; was manylinux2014).
  The documented carve-out fired: no Vulkan toolchain can exist on glibc 2.17
  (LunarG's prebuilt glslc needs glibc 2.34 — measured; even source-built
  toolchains want newer CMake/compilers), and the manylinux2014 images are
  EOL. `manylinux_2_28` is the de-facto ecosystem floor (PyTorch ships it).
  The Vulkan toolchain (Headers/Loader/glslc) is built from pinned Khronos
  git tags inside the container — LunarG prunes old SDK downloads, so pinned
  LunarG URLs rot (observed: 1.4.313.2 Linux tarball already gone).
- **Metal ships by default on macOS arm64**, with
  `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON` (no sidecar shader files).
- **Native artifacts are named build tuples** (`linux-x86_64-cpu`,
  `macos-arm64-metal`, ...). There is no single portable
  `libtranscribe.so`; artifacts vary by OS, arch, libc, backend set, and CPU
  feature policy. Every wheel is self-contained for its tuple.
- **sdist every release.** Builds from source against vendored ggml; the
  universal fallback for platforms/backends without prebuilt wheels and the
  path for local CUDA/ROCm/CPU tuning. It does not replace wheels for
  adoption.
- **Licensing.** Every wheel vendoring native code includes third-party
  license texts (ggml, any bundled backend/runtime components); license
  inclusion is part of provider packaging tests.
- **Complete the Python API surface before broadening the provider matrix.**
  A mostly complete API on CPU+Metal is a better cross-language precedent
  than a broad backend matrix with a partial surface. (The API surface is now
  essentially complete, so the remaining work is all distribution.)

## Remaining design context

### The provider artifact is a backend-module directory (verified)

Vulkan-by-default on Linux/Windows forces the artifact-shape decision that
was previously deferred: the default provider is a **directory** of ggml
dynamic backend modules, not one library. The vendored ggml was inspected
(2026-06) and already supports everything required — this is the same shape
llama.cpp ships for its multi-backend release binaries:

- `GGML_BACKEND_DL=ON` builds every backend as a separate `MODULE` library
  (`ggml/src/CMakeLists.txt:247`, `ggml_add_backend_library`). The core
  `libggml` **never links `libvulkan`**; only the `libggml-vulkan` module
  does.
- The registry tolerates module-load failure: `load_backend()` logs and
  returns nullptr when `dlopen` fails (`ggml-backend-reg.cpp:202`). A
  machine with no Vulkan loader simply never registers the Vulkan backend —
  **no crash, no import failure**, CPU keeps working.
- Loader present but no driver/device is equally safe:
  `ggml_backend_vk_reg()` wraps instance init in catch-all handlers and
  returns nullptr on any failure (`ggml-vulkan.cpp:15930`); zero devices
  logs "No devices found" and returns early (`:5825`).
- `ggml_backend_load_all_from_path(dir)` is the exact primitive for a
  package-local artifact directory.
- CUDA implements the identical module protocol
  (`GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)`), so the CUDA provider
  reuses this mechanism unchanged.

Consequences:

- The default Linux/Windows provider wheel contains: `libtranscribe` +
  `libggml` (+ base libs) + a conservative `ggml-cpu` module + the
  `ggml-vulkan` module. Vulkan wins when usable; CPU is the silent floor.
- Packaging risks move into scope and must be proven by the validation
  track: `$ORIGIN`/`@loader_path`/`add_dll_directory` resolution inside the
  wheel; module search must never escape the package; wheel-repair tools
  cannot see string-dlopened modules, so they are explicitly included,
  repaired, and re-tested after repair.
- Build images need the Vulkan SDK (glslc) to compile the Vulkan module with
  embedded SPIR-V (manylinux container and the Windows lane).

Degradation validation track (required before release):

1. Clean Linux container, no loader: import works, CPU transcribes,
   `backend="vulkan"` raises a clean Python exception.
2. Container with loader but no ICD/driver: same outcome.
3. Real Vulkan machine (Linux and Windows): `backend="vulkan"` runs and the
   diagnostics name the device.

### CPU variants ("Strategy B") ride the same mechanism — fast follow

With `GGML_BACKEND_DL` in the release path, fat CPU is no longer a separate
high-risk track: `GGML_CPU_ALL_VARIANTS=ON` is a one-flag addition that
emits one CPU module per ISA tier with runtime feature scoring. It stays a
**fast follow** (not merge-blocking) only because its acceptance gate is
hardware: a no-AVX2 machine must run, an AVX-512 machine must select the
AVX-512 module. Until then the shipped CPU module is the conservative
baseline (every SIMD tier explicitly off — `GGML_NATIVE=OFF` alone does not
disable them). Strategy B+ (compiled-in baseline + dynamic override) remains
an upstream-ggml improvement track.

### Backend init / diagnostics C API (now required for release)

The binding must point ggml at the wheel-local module directory, so this
API moves from sketch to requirement — and it is shared groundwork for the
Rust/Swift/TS bindings:

```c
transcribe_status transcribe_init_backends(const char * artifact_dir);
```

Wraps `ggml_backend_load_all_from_path` for a package-local directory;
never searches an ambient system directory. Diagnostics must answer: which
artifact directory loaded, which backend modules/devices were discovered,
which backend serves a given request, and why a requested backend (e.g.
`vulkan`) is unavailable on this machine. Public-header change → regenerate
the FFI layer (the drift gate enforces this).

### CUDA: opt-in provider, built and tested this cycle

CUDA stays out of the default wheel for size reasons (fatbins + NVIDIA
runtime deps), not mechanism — it uses the same backend-module shape.
`transcribe-cpp-native-cu12`, installed via `transcribe-cpp[cu12]`. Do not
vendor the toolkit: depend on NVIDIA runtime wheels
(`nvidia-cuda-runtime-cu12`, ...) and resolve/load them explicitly — they
install under `site-packages/nvidia/*/lib`, which the platform loader does
not find unaided. Build CI now; a real NVIDIA runtime smoke is required
before the provider is called release-grade (the one item that may trail
the merge, with explicit sign-off). `cu13`/ROCm only when demand justifies.

## Punch list

In order. **The merge bar is release-ready**: everything through Milestone 6
is merge-blocking (one explicitly flagged exception); Milestone 7 is the
fast-follow set.

### Milestone 1 — Lock in what exists (CI)

- [x] GitHub Actions workflow: generator drift gate
      (`_generate/generate.py --check`) + version-sync check
      (`transcribe.h` ↔ `pyproject.toml` ↔ `__init__.__version__`).
      (`.github/workflows/python-bindings.yml`, `lint` job;
      `_generate/check_version_sync.py`.)
- [x] Static build lane: full C++ white-box test suite.
      (`cpp-tests` job, Linux + macOS.)
- [x] Shared build lane (`TRANSCRIBE_BUILD_SHARED=ON`): `api_smoke` as the
      exported-symbol canary + Python no-model tests (import, ABI layout,
      version gate). (`python-shared` job.)
- [x] Canary model wired: `handy-computer/whisper-tiny-gguf` (private for
      now, public later) → `whisper-tiny-Q5_K_M.gguf` (44 MB, MIT weights).
      CI downloads it when the `HF_TOKEN` repo secret exists (forks without
      it keep skipping cleanly): `python-shared` runs the offline
      transcription tests for real, and `provider-dl-vulkan` tier 2b
      transcribes end-to-end ON the lavapipe Vulkan device. The streaming
      canary (`handy-computer/moonshine-streaming-tiny-gguf` → Q8_0, 50 MB)
      is wired the same way, so the streaming tests run in CI too. The
      prompted (nemotron) streaming test stays local-only; its regression
      is pinned in CI by the sanitized C-level test.
- [x] Convert `tests/smoke.py` into a pytest suite with proper skips.

### Milestone 2 — Pre-publish hygiene (small; alongside M1)

- [x] Raise `requires-python` to ≥3.9 (3.8 is EOL).
- [x] Import-time version gate compares *base* versions so a `.postN`
      packaging fix still loads.
- [x] Test asserting `errors.py` status codes match the `_generated` enum
      values.

### Milestone 3 — Correctness floor (done 2026-06-10)

- [x] Streaming run-params retention fixed **in the dispatcher**, not per
      family: `transcribe_stream_begin` copies `language`/`target_language`
      into session-owned storage and hands the family hooks a params view
      pointing at it, with the run-slot `family` ext nulled (no family reads
      it on the stream path; a retained pointer to it would dangle per the
      ext copy-out contract). Zero per-family edits. The header documents
      the contract: everything passed to begin may be freed once it returns.
      Repro evidence: the language-hint pytest failed against the unfixed
      library with visibly corrupted memory (`"dn-US"`), and the C-level
      repro produced `AddressSanitizer: heap-use-after-free, READ of size 3`
      with the fix reverted. Both green post-fix.
- [x] Python `Stream` pins `(run_params, stream_params, ext)` until
      `reset()` — defense in depth on top of the C contract.
- [x] `Model.close()` closes live sessions (weakref-tracked) before freeing
      the model, so explicit close/context-exit is safe in any order.
- [x] Tests: streaming with `language=` on the prompted parakeet streaming
      model + moonshine (`test_streaming.py`); close-ordering/lifetime
      battery (`test_lifetime.py`); PCM edge-case battery (`test_pcm.py`,
      including optional numpy cases). Suite: 82 passed.
- [x] Sanitizer certification, adjusted for platform reality: the full
      white-box C++ suite (38 tests) runs clean under ASan+UBSan, including
      a new FFI-shaped regression in `stream_dispatch_unit.cpp` that
      heap-allocates params, frees+scribbles them right after begin, and
      feeds — the exact ctypes caller shape. A `cpp-tests-sanitized` CI lane
      (Linux) now runs this on every push. (Injecting ASan under a Python
      process is blocked on macOS — dyld strips `DYLD_INSERT_LIBRARIES`;
      the C-level lane certifies the same contracts. A sanitized *pytest*
      lane becomes worthwhile once the CI canary model exists.)
- [x] Bonus class fixed while certifying: loading caller-supplied enum
      fields through enum-typed lvalues is UB in C++ for out-of-range ints
      that C callers can legally pass. Every public boundary now raw-reads
      and validates enums first (`enum_field_raw`): run-params
      task/timestamps/pnc/itn, session kv_type, model-load backend,
      `init_backends`, ABI accessors, `transcribe_model_supports`, stream
      commit_policy. UBSan found three live instances; all fixed.

### Milestone 4 — Backend-module provider core (core done 2026-06-10)

- [x] `transcribe_init_backends(artifact_dir)` + device diagnostics:
      `transcribe_backend_device_count` / `transcribe_get_backend_device`
      (name, description, classified kind string) /
      `transcribe_backend_available(kind)`. Package-local by construction
      (ggml scans ONLY the given dir when one is passed); idempotent per
      canonical directory; zero-devices after load is a loud
      `ERR_BACKEND`. New ABI id 13; FFI regenerated (header hash moved, as
      it should for a real ABI change); covered in `api_smoke` and
      `tests/test_backends.py`. Python surface: `backends()`,
      `backend_available()`, import-time module loading from the artifact
      dir.
- [x] `TRANSCRIBE_GGML_BACKEND_DL` option (requires `TRANSCRIBE_BUILD_SHARED`)
      + presets `wheel-linux-cpu-vulkan` and `wheel-windows-cpu-vulkan`
      (conservative CPU module + Vulkan module). macOS arm64 stays a single
      Metal artifact. **Track finding:** the first DL build exposed direct
      link-time references to CPU-module symbols
      (`ggml_backend_cpu_init/_set_n_threads` in parakeet's host-joint
      decoder) — replaced with registry-based
      `ggml_backend_init_by_type` + `get_proc_address`, the pattern any
      backend-specific call must use from now on.
- [x] Loader/binding wiring: the loader records the artifact directory
      (provider `artifact_dir`, else the library's own dir), import calls
      `transcribe_init_backends` on it, Windows already uses
      `os.add_dll_directory`.
- [x] **Local end-to-end proof (macOS):** built the DL configuration,
      assembled a flat wheel-like directory (`libtranscribe` + ggml libs +
      `libggml-cpu.so` + `libggml-metal.so`, `@loader_path` rpaths via
      `install_name_tool`), hid the build tree so nothing could resolve
      outside it, and ran the full 90-test pytest suite through it — ggml
      logged both modules loading from the provider directory, devices
      classified `[('MTL0','metal'), ('CPU','cpu')]`, vulkan answered
      unavailable, and a real model transcribed on Metal.
- [x] CI `provider-dl-vulkan` lane, **green 2026-06-10**: builds the Linux
      CPU+Vulkan module provider with glslc, assembles the flat directory
      with `$ORIGIN` rpaths via patchelf, deletes the build tree, then
      proves tier 1 (Vulkan loader removed → `[('CPU','cpu')]`, vulkan
      answers False) and tier 2 (mesa lavapipe installed → the same
      artifacts report `[('Vulkan0','vulkan'), ('CPU','cpu')]`; needs
      `GGML_VK_VISIBLE_DEVICES=0` because lavapipe is a CPU-type Vulkan
      device that ggml's default dedicated-GPU filter excludes). Getting it
      green also caught: the `GGML_NATIVE`+`BACKEND_DL` x86 hard error (now
      forced off by `TRANSCRIBE_GGML_BACKEND_DL`) and gcc's internal
      linkage for `extern "C"` definitions inside anonymous namespaces
      (clang exports them — the new API had to move to file scope).
- [x] Real-GPU Vulkan validation, Linux (2026-06-10): Fedora / ThinkPad
      T14, AMD Renoir iGPU via RADV. Provider-directory build assembled on
      the machine, `libggml-vulkan.so` loaded from the directory, device
      discovered through the DEFAULT selection path (no
      `GGML_VK_VISIBLE_DEVICES` override — confirms integrated GPUs pass
      ggml's device filter), model placed on `Vulkan0`, jfk.wav transcribed
      correctly with `backend="vulkan"`.
- [ ] Remaining: real-GPU Vulkan validation on a physical **Windows**
      machine, and Vulkan SDK in the *cibuildwheel* images when the M5
      wheel lanes are built.

### Milestone 5 — Wheels and out-of-box proof

- [x] Provider discovery in `_library.py`: entry points in group
      `transcribe_cpp.native`, version/header-hash hard-fail per the
      contract above. (Descriptor fields: `name`, `library_path`/`artifact_dir`,
      `version`, `header_hash`, `backends`. Selection: explicit arg →
      `TRANSCRIBE_NATIVE_PROVIDER` → best accelerated → CPU. The ABI tag is
      `_generated.PUBLIC_HEADER_HASH`, a stable digest the generator emits;
      a provider echoes it back. Covered by `tests/test_provider_discovery.py`.)
- [x] End-to-end provider test (fake entry point → descriptor → selection →
      contract → load): `test_provider_discovery.py` now builds a real
      installed-distribution shape on disk (module + `.dist-info` advertising
      `transcribe_cpp.native`) and drives importlib.metadata discovery →
      `ep.load()` → contract → an actual dlopen of the suite's library; the
      ABI-mismatch twin proves refusal happens before dlopen.
- [x] Provider package skeleton (done 2026-06-10): `transcribe-cpp-native`
      builds from the repo-root `pyproject.toml` via **scikit-build-core**
      (one backend for wheels AND the from-source sdist). Pieces:
      `cmake/python-wheel-install.cmake` (SKBUILD-only; flat
      `transcribe_cpp_native/_native/` artifact dir, `COMPONENT wheel` so
      ggml's header/cmake-config installs stay out, `$ORIGIN`/`@loader_path`
      rpaths, VERSION/SOVERSION stripped — wheels can't carry symlinks — and
      `_contract.py` stamped from `PROJECT_VERSION` + the drift-gated
      `_generated.py` header hash + `GGML_AVAILABLE_BACKENDS`);
      `bindings/python-native/` (descriptor + entry point); SKBUILD posture
      block in the top-level CMakeLists (shared, no tests/examples, OpenMP/
      system-BLAS off by default, Metal shaders embedded). Package version is
      regex-parsed from `transcribe.h` at build time — same single source as
      CMake. `[tool.uv] package = false` keeps `uv run scripts/...` from ever
      triggering a surprise native build (verified).
      **Local end-to-end proof (macOS arm64):** `uv build --wheel` →
      `transcribe_cpp_native-0.0.1-py3-none-macosx_26_0_arm64.whl` (3.4 MB:
      libtranscribe + ggml/base/cpu/metal, `_contract.py` echoing hash
      `0007af60bfcecf7e`, backends `("metal","cpu")`, both LICENSE files);
      otool shows every cross-ref `@rpath` + `@loader_path` with system-only
      externals; clean venv + `pip install` both wheels → entry-point
      discovery selected the provider from site-packages, Metal device found,
      embedded metallib used, jfk.wav transcribed, **zero env vars**; full
      pytest suite (95 tests) green against the installed pair AND in
      dev-tree mode.
- [ ] cibuildwheel lanes: Linux x86_64 (cpu+vulkan), Windows x86_64
      (cpu+vulkan), macOS arm64 (Metal). Repair with
      auditwheel/delvewheel/delocate — string-dlopened modules must be
      explicitly included — and **test the repaired artifacts**, not the
      build outputs.
      **Written 2026-06-10, CI validation pending:**
      `.github/workflows/python-wheels.yml` + `[tool.cibuildwheel]` /
      `TRANSCRIBE_WHEEL_LANE` overrides in the root pyproject (lanes mirror
      the wheel-* presets; cross-referenced). cibuildwheel's test phase runs
      `scripts/ci/wheel_smoke.py` against the *repaired* wheel in a fresh
      venv: installs the API package (resolver-level pin check), pulls the
      canary GGUFs when HF_TOKEN exists, asserts provider identity +
      per-platform backend posture (Metal present on macOS; Vulkan quietly
      unavailable on GPU-less CI), runs the full pytest suite. Linux builds
      in manylinux_2_28 with the Vulkan toolchain source-built from pinned
      tags (`scripts/ci/manylinux-vulkan-toolchain.sh`, cached via a mounted
      volume); repair keeps the system loader out (`auditwheel --exclude
      libvulkan.so.1`, delvewheel `--exclude vulkan-1.dll`) per the verified
      degradation design. **Local macOS evidence:** the metal lane built via
      the override with `MACOSX_DEPLOYMENT_TARGET=11.0` →
      `py3-none-macosx_11_0_arm64`; `delocate-listdeps` shows system-only
      externals; delocate repair normalized install names and vendored
      nothing; the *repaired* wheel transcribed on Metal in a clean venv with
      zero env vars. (cibuildwheel itself refuses local non-CI runs without
      a python.org CPython — orchestration is proven on CI.) **Known risk:**
      the Windows lane is the first MSVC build of this codebase ever; zlib
      comes via vcpkg static (`find_package(ZLIB REQUIRED)`), and C++
      portability fixes may surface. Windows real-GPU validation stays
      deferred per M4.
- [x] sdist that compiles from source (scikit-build-core; local proof
      2026-06-10, CI job `sdist` re-proves on Linux every run). `uv build
      --sdist` → 6.1 MB compressed / 28 MB uncompressed, 2493 files; audit
      asserts no GGUF/model/dump/report artifacts and only `samples/jfk.wav`
      ships (gitignored trees are auto-excluded — verified `samples/wer`
      600 MB stayed out; tracked sample WAVs are pruned by `sdist.exclude` +
      `sdist.include` carve-back). Fresh venv `pip install <tarball>`
      compiled from source (cmake/ninja arrive as build deps), registered
      the provider entry point, discovered Metal+CPU, and passed the full
      95-test suite with real models via `scripts/ci/wheel_smoke.py`.
- [ ] Co-import smokes in wheel CI: `numpy` then `torch`, each followed by a
      real transcription.
      **Wired 2026-06-10** (`co-import` job in python-wheels.yml +
      `scripts/ci/co_import_smoke.py`): both import orders per framework
      (framework-first, and library-first with a transcription before AND
      after the framework import — late-loaded runtime clashes), all three
      platforms, installing from the built artifacts via
      `pip install --no-index --find-links wheelhouse transcribe-cpp` (which
      doubles as the resolver proof for the hard pin). **Local evidence:**
      the numpy smoke passed both orders against the repaired macOS wheel
      with real transcriptions. CI validation (incl. torch) pending.
- [ ] Clean-machine installs (no repo checkout, no env vars): fresh
      manylinux container, clean macOS account/machine, clean Windows
      machine — install the wheel, transcribe.
- [ ] TestPyPI rehearsal of the full set; install-from-index smoke; then
      merge and cut **0.1.0 on PyPI** from the merge commit.

### Milestone 6 — CUDA provider (this cycle)

- [ ] `transcribe-cpp-native-cu12` package: same backend-module mechanism,
      NVIDIA runtime-wheel dependencies, explicit load-path resolution for
      `site-packages/nvidia/*/lib`.
- [ ] Build CI lane.
- [ ] Real NVIDIA runtime smoke. *The one item that may trail the merge,
      with explicit sign-off — it requires GPU hardware/runners.*

### Milestone 7 — Fast follows (post-merge)

- [ ] Fat CPU variants: add `GGML_CPU_ALL_VARIANTS=ON` to the proven module
      directory; gate on no-AVX2 and AVX-512 hardware validation; then make
      it the default CPU module set.
- [ ] README/examples polish round two; `cu13`/ROCm if demand justifies;
      Strategy B+ upstream patch evaluation.

## Open questions

- Whether the CUDA runtime smoke gates the merge or trails it (needs NVIDIA
  hardware — human decision).
- Which physical machines validate Vulkan-on-Windows and the fat-CPU
  hardware matrix.
- Whether to pursue (and upstream) the Strategy B+ baseline-plus-variants
  ggml patch.
- Whether opt-in OpenMP/BLAS-enabled providers are worth shipping after
  co-import testing.
