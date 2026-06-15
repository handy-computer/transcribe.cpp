# transcribe-cpp-sys

Raw native FFI bindings for
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), a C/C++
speech-to-text library built on ggml.

> **Status: in development (0.0.1).** This crate exposes the unsafe, generated
> FFI surface. Most users want the safe wrapper,
> [`transcribe-cpp`](https://crates.io/crates/transcribe-cpp).

## What it does

`build.rs` compiles the vendored C++ tree from source via CMake (the crate
tarball carries the whole tree) and reconstructs the link line from the
installed `transcribe-link.json` manifest — no hardcoded per-platform link
lists. The committed bindgen output means **libclang is not needed** to build
this crate.

## Build prerequisites

A C++ toolchain, **CMake**, and **zlib**. zlib is system-provided on Linux
(`zlib1g-dev`) and macOS; on Windows install it with
`vcpkg install zlib:x64-windows-static-md` (static lib against the dynamic CRT,
matching Rust's default `/MD` on `x86_64-pc-windows-msvc`) and point
`CMAKE_PREFIX_PATH` at the vcpkg install tree. The static link is the default;
the `shared` feature links a shared library instead.

## Features

- `metal` (default on Apple), `vulkan`, `cuda`, `openmp` — each forwards to the
  matching `TRANSCRIBE_*` CMake option.
- `shared` — link a shared `libtranscribe` (`.so`/`.dylib`/`.dll`) loaded at
  runtime instead of statically baking it in. The default is a self-contained
  static link.
- `dynamic-backends` — additionally ship each compute backend (the per-ISA CPU
  tiers, Vulkan, CUDA, …) as a loadable module next to the library, selected at
  runtime by `transcribe_init_backends_default()` when the modules sit next to
  `libtranscribe`, or `transcribe_init_backends(dir)` for a custom provider
  directory. Implies `shared`.

## Build-flag escape hatch

The features above cover the common, tested configurations. Anything else CMake
accepts can be forwarded via the `TRANSCRIBE_CMAKE_ARGS` (or `CMAKE_ARGS`) env
var — e.g. `TRANSCRIBE_CMAKE_ARGS="-DGGML_VULKAN=ON" cargo build`. These are
split on whitespace with simple double-quote handling, applied after the
feature-derived defines (so a user `-D` wins), and unsupported/untested by
design: they exist so a Cargo feature is never a hard ceiling on what you can
configure. The link line is still reconstructed from the generated manifest, so
whatever you turn on links correctly.

## ABI drift

The generated FFI is committed and CI-checked against `include/transcribe.abihash`
(`cargo xtask bindgen --check`): a public-header ABI change turns the check red
until the bindings are regenerated. Per-field layout checks are waived because
bindgen takes layout from a real compiler at generation time.

- Crate: `transcribe-cpp-sys` (raw FFI; the safe API is `transcribe-cpp`)
- License: MIT
