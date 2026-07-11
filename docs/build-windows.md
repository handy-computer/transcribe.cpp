# Building transcribe.cpp on Windows

A start-to-finish guide for building `transcribe.cpp` and running a
transcription on a **clean Windows 11 machine** — one with no compiler,
no CMake, and no build tooling installed. It targets the CPU backend,
which is the simplest path and needs no GPU SDK.

Worked example at the end: **Parakeet TDT 0.6B v3, Q8_0** on `samples/jfk.wav`.

> Tested on Windows 11 Pro (26200) with Visual Studio Build Tools 2022
> (MSVC 19.44 / toolset 14.44), CMake 4.x, and the CPU backend. Other
> recent versions should work the same way.

---

## 0. What you need

Windows 11 includes **winget** (the App Installer's command-line tool) — run
it from any PowerShell window. Everything else, **including git**, is
installed below. The full toolchain:

| Tool | Why | Install |
| --- | --- | --- |
| winget | installs the rest | ships with Windows 11 (App Installer) |
| Git | clone the repo | step 1 |
| Visual Studio Build Tools 2022 (C++) | the `cl.exe` compiler + Windows SDK | step 1 |
| CMake | build system | step 1 |

> If `winget --version` errors, install or update **App Installer** from the
> Microsoft Store, then reopen PowerShell.

There is **no** external native dependency to install: the only compression
code (Whisper's compression-ratio metric) is the vendored **miniz**, built
straight into the library — no zlib, no vcpkg.

You do **not** need Visual Studio (the full IDE) — the Build Tools are enough.
You do **not** need the Vulkan or CUDA SDK for a CPU build.

---

## 1. Install git, the compiler, and CMake

Run in PowerShell. The Build Tools installer needs **elevation** — accept
the UAC prompt when it appears. Declining it aborts the install with MSI
error `1602` ("user cancelled").

```powershell
# Git
winget install --id Git.Git --accept-source-agreements --accept-package-agreements

# CMake
winget install --id Kitware.CMake --accept-source-agreements --accept-package-agreements

# MSVC C++ build tools (multi-GB download; the C++ workload is NOT default,
# so it must be requested explicitly via --override)
winget install --id Microsoft.VisualStudio.2022.BuildTools `
  --accept-source-agreements --accept-package-agreements `
  --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended"
```

> **Restart your terminal after these installs.** winget updates your `PATH`,
> but an already-open shell keeps its old environment. A new terminal will
> have `git` and `cmake` on `PATH`, and CMake will be able to find the MSVC
> compiler.

Verify in a fresh terminal:

```powershell
git --version
cmake --version
```

`cl.exe` will *not* be on your global `PATH`, and that's expected — CMake's
"Visual Studio 17 2022" generator locates MSVC automatically via `vswhere`,
so you don't need a Developer Command Prompt for this.

---

## 2. Configure and build

Clone the repo and configure — no toolchain file or external dependency is
needed:

```powershell
git clone https://github.com/handy-computer/transcribe.cpp
cd transcribe.cpp

cmake -B build

cmake --build build --target transcribe-cli --config Release
```

Notes:
- The Visual Studio generator is **multi-config**, so pass `--config Release`
  at build time (omitting it yields a Debug build).
- OpenMP is **not** required on Windows. ggml's native CPU threadpool is the
  default everywhere (see the OpenMP CENTRAL POLICY in the top-level
  `CMakeLists.txt`); the MSVC barrier bug that once forced OpenMP here is fixed,
  so the native pool is correct under MSVC and oversubscription. Build with the
  defaults; opt into OpenMP only with `-DTRANSCRIBE_USE_OPENMP=ON` if you have a
  specific reason.
- `no BLAS found — decoder uses scalar fallback` is fine; it only affects
  host-side decoder speed, not correctness.
- `uv not found on PATH` is harmless — it only skips an optional test.

The binary lands at `build\bin\Release\transcribe-cli.exe`. It is
self-contained — no extra DLLs to stage beyond the standard VC++ runtime.

---

## 3. Get a model

Pre-built GGUFs are hosted on Hugging Face under
[`handy-computer`](https://huggingface.co/handy-computer); each model doc in
[`docs/models/`](models/) links every quant. Example for Parakeet v3 Q8_0
(~705 MB):

```powershell
mkdir models\parakeet-tdt-0.6b-v3
curl.exe -L --fail `
  -o models\parakeet-tdt-0.6b-v3\parakeet-tdt-0.6b-v3-Q8_0.gguf `
  "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf"
```

---

## 4. Run

```powershell
build\bin\Release\transcribe-cli.exe `
  -m models\parakeet-tdt-0.6b-v3\parakeet-tdt-0.6b-v3-Q8_0.gguf `
  samples\jfk.wav
```

Expected output:

```
backend:    CPU
text: And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
realtime:   6x (1703.9 ms for 11.0 s)
```

Input must be **16 kHz mono WAV**. `samples\jfk.wav` already is. For other
audio, convert it first with ffmpeg (`winget install Gyan.FFmpeg`):

```powershell
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

---

## 5. Optional: the Vulkan GPU backend

The CPU build above is the simplest path. To run on a GPU, build the Vulkan
backend instead. It needs the **Vulkan SDK** (for `glslc`, headers, and
`vulkan-1.lib`); the Vulkan *runtime* loader (`vulkan-1.dll`) already ships
with most modern GPU drivers.

```powershell
winget install --id KhronosGroup.VulkanSDK --accept-source-agreements --accept-package-agreements
```

The SDK sets a machine-wide `VULKAN_SDK` environment variable — **open a new
terminal** so it's visible.

A plain `cmake -B build` from a normal checkout location works — no special
build root needed:

```powershell
cmake -B build -DTRANSCRIBE_VULKAN=ON
cmake --build build --target transcribe-cli --config Release
```

A successful configure prints `Found Vulkan: ... found components: glslc`
and `Including Vulkan backend`. Run it the same way; the CLI auto-selects the
GPU:

```powershell
build\bin\Release\transcribe-cli.exe `
  -m models\parakeet-tdt-0.6b-v3\parakeet-tdt-0.6b-v3-Q8_0.gguf `
  samples\jfk.wav
# backend: Vulkan0  (e.g. "Intel(R) Iris(R) Xe Graphics")
```

> ### Only for very deep checkouts: MAX_PATH
> ggml builds its `vulkan-shaders-gen` helper as a nested ExternalProject;
> transcribe.cpp flattens it to `<build>\e\src\` on Windows so normal paths
> stay under Windows' 260-char `MAX_PATH` limit. But if your build directory
> path is itself very long (roughly 120+ characters), the intermediate paths
> can still overflow, and the build dies early with a misleading
> `error MSB6003: The specified task executable "link.exe" could not be run`
> / `DirectoryNotFoundException` on a `.tlog` path — i.e. the compiler check
> reports "broken." It is **not** a compiler or Vulkan problem.
>
> The fix is to put the build directory at a short root instead, e.g.
> `cmake -B C:\bv -S . -DTRANSCRIBE_VULKAN=ON`. (Enabling Win32 long paths
> does not reliably help — MSBuild's file tracker doesn't fully honor them.)

> **Performance note — measure *warm*, not the first run.** A single
> `transcribe-cli` invocation pays a large one-time cost on Vulkan: the
> backend compiles its compute pipelines (SPIR-V → device shaders) and
> uploads weights to the GPU on first use. That can add several seconds to
> the first run and makes a one-shot timing look *slower* than CPU. Warm
> steady-state is the real number — and there Vulkan wins. Measured on this
> machine (Intel Iris Xe, Parakeet v3 Q8_0, `jfk.wav`, 1 warmup + 3 iters
> via `transcribe-bench`): **Vulkan ≈ 877 ms (12.5× realtime) vs CPU
> ≈ 1621 ms (6.8×)**. Use `transcribe-bench --backend vulkan|cpu --warmup 1
> --iters 3` (build with `-DTRANSCRIBE_BUILD_TOOLS=ON`) to compare on your
> own hardware rather than trusting a single cold run.

---

## Troubleshooting recap

| Symptom | Cause | Fix |
| --- | --- | --- |
| winget install fails with `1602` | UAC / elevation prompt was declined | rerun and accept the prompt |
| `git` / `cmake` not found after install | shell has stale `PATH` | open a new terminal |
| build is Debug / slow | VS generator is multi-config | add `--config Release` |
| Vulkan build: `MSB6003 ... link.exe could not be run` / `DirectoryNotFoundException` on a `.tlog` | `MAX_PATH` (260) exceeded — build dir path too deep even for the flattened `e\src` shader-gen layout | build from a short root, e.g. `cmake -B C:\bv ...` |
