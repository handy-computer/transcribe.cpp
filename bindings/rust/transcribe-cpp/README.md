# transcribe-cpp

Safe, idiomatic Rust bindings for
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), a C/C++
speech-to-text library built on ggml.

> **Status: in development (0.0.1).** The full feature surface — model load,
> sessions, `run`/`run_batch`, owned results, error mapping, backend discovery,
> the version/ABI gate, streaming, the five family extensions, cancellation,
> tokenize, and log routing — is implemented and tested against the canary
> models. CI-executed examples and the publish flow (M6/M7) are still to come.

```rust
use transcribe_cpp::{Model, RunOptions};

let mut session = Model::load("model.gguf")?.session()?;
// pcm: 16 kHz mono f32 in [-1, 1]
let result = session.run(&pcm, &RunOptions::default())?;
println!("{}", result.text);
# Ok::<(), transcribe_cpp::Error>(())
```

The raw FFI layer is the
[`transcribe-cpp-sys`](https://crates.io/crates/transcribe-cpp-sys) crate; this
crate is the safe wrapper on top of it.

## Backends

The native library is built from source by `transcribe-cpp-sys` (see its
README). Backends are selected with cargo features — `metal` (default on Apple),
`vulkan`, `cuda`, `openmp` — forwarded to the underlying build. A static,
self-contained link is the default; `dylib` links a shared library.

## Threading

- `Model` is `Send + Sync` and cheap to clone (`Arc`-backed); the native model
  is freed only after the last handle and every `Session` drop.
- `Session` is `Send` but not `Sync`; mutating calls take `&mut self`.
- In 0.x the C library allows at most one in-flight run across all sessions of a
  model; this crate enforces it with a per-model mutex, so concurrent calls
  queue rather than race. For real parallelism, use one `Model` per worker.

## ABI verification

The per-field struct-layout check the ctypes binding performs is **waived**
here: bindgen takes every struct's layout from a real compiler at generation
time, so the generated FFI cannot disagree with the headers it was built
against. The load-time base-version lock (pre-1.0) is retained.

- License: MIT
