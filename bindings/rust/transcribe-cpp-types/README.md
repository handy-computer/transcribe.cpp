# transcribe-cpp-types

The plain-data types of [`transcribe-cpp`](https://crates.io/crates/transcribe-cpp),
the Rust binding for
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp): options,
results, enums, errors, abilities, stream updates and family extensions. No
handles, no FFI, nothing native, no build script.

`transcribe-cpp` depends on this crate at the exact same version and re-exports
every type at its root, so `transcribe_cpp::RunOptions` and
`transcribe_cpp_types::RunOptions` are the same type. Use this crate on its own
when a program needs to name those types without the engine
(`transcribe-cpp-sys`, CMake, ggml, Metal/Vulkan/CUDA) in its dependency graph,
for example a process that talks to transcribe.cpp workers over IPC in the
workers' own types:

```sh
cargo add transcribe-cpp-types --features serde
```

Use the same version as the `transcribe-cpp` your workers run: the two crates
are released together, and `transcribe-cpp` pins this crate exactly.

## Features

- `serde`: `Serialize` / `Deserialize` on every type. The wire format is the
  one `transcribe-cpp` has always used (`transcribe-cpp/serde` turns this
  feature on).

## Building values

A few types are `#[non_exhaustive]` and cannot be built with a struct literal
outside this crate. `ErrorReport::new(kind, message)` builds an error report;
the `*Info` / `*Result` types implement `Default`, so build one with
`default()` and set its public fields.

## License

MIT
