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
the `dylib` feature links a shared library instead.

## Features

- `metal` (default on Apple), `vulkan`, `cuda`, `openmp` — each forwards to the
  matching `TRANSCRIBE_*` CMake option.
- `dylib` — link a shared `libtranscribe` (and the backend-module /
  `transcribe_init_backends` posture). The default is a self-contained static
  link.

## ABI drift

The generated FFI is committed and CI-checked against `include/transcribe.abihash`
(`cargo xtask bindgen --check`): a public-header ABI change turns the check red
until the bindings are regenerated. Per-field layout checks are waived because
bindgen takes layout from a real compiler at generation time.

- Crate: `transcribe-cpp-sys` (raw FFI; the safe API is `transcribe-cpp`)
- License: MIT
