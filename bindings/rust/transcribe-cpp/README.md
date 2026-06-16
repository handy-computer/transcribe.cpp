# transcribe-cpp

Safe, idiomatic Rust bindings for
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), a C/C++
speech-to-text library built on ggml.

> **Status: in development (0.0.1).** Core model, session, run, stream,
> cancellation, backend, and family-extension APIs are implemented and tested.

## Install

```sh
cargo add transcribe-cpp
```

The native library is compiled from source by the `transcribe-cpp-sys` crate, so
a first build needs a C++ toolchain, **CMake**, and **zlib** (system zlib on
Linux/macOS; on Windows, `vcpkg install zlib:x64-windows-static-md` and point
`CMAKE_PREFIX_PATH` at it). No prebuilt download and no environment variables on
the happy path.

## Quickstart

```rust
use transcribe_cpp::{Model, RunOptions};

let mut session = Model::load("model.gguf")?.session()?;
// pcm: 16 kHz mono f32 in [-1, 1]
let result = session.run(&pcm, &RunOptions::default())?;
println!("{}", result.text);
# Ok::<(), transcribe_cpp::Error>(())
```

Runnable examples:

```sh
cargo run --example transcribe-file
cargo run --example streaming
cargo run --example batch
cargo run --example backend-select
cargo run --example error-handling
```

The raw FFI layer is
[`transcribe-cpp-sys`](https://crates.io/crates/transcribe-cpp-sys); this crate
is the safe wrapper.

## Backends

Backends are selected with cargo features forwarded to `transcribe-cpp-sys`:
`metal` (default on Apple), `vulkan`, `cuda`, and `openmp`.

The default link is static and self-contained. Advanced packaging modes are
available through `shared` and `dynamic-backends`; see the `transcribe-cpp-sys`
README if you need runtime-loaded backend modules or custom
`TRANSCRIBE_CMAKE_ARGS`.

## Threading

- `Model` is `Send + Sync` and cheap to clone (`Arc`-backed); the native model
  is freed only after the last handle and every `Session` drop.
- `Session` is `Send` but not `Sync`; mutating calls take `&mut self`.
- In 0.x the C library allows at most one in-flight run across all sessions of a
  model; this crate enforces it with a per-model mutex, so concurrent calls
  queue rather than race. For real parallelism, use one `Model` per worker.

- License: MIT
