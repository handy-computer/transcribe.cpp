# Serde goldens from transcribe-cpp 0.5.0

`json/<Name>.json` is `serde_json::to_string` and `postcard/<Name>.hex` the
postcard bytes (hex) of the value `values.rs` (and, for `Error.*`, `errors.rs`)
builds under `<Name>`, **as written by transcribe-cpp 0.5.0 from crates.io**,
the last release before the plain-data types moved into this crate. They pin
the wire format: `tests/serde_golden.rs` here and `tests/serde.rs` in
transcribe-cpp fail if any type writes or reads differently.

Never regenerate them from this tree. To add a value, write it from the oldest
release that has the type, with a throwaway crate outside the workspace:

```toml
# Cargo.toml
[package]
name = "tcpp-golden-gen"
version = "0.0.0"
edition = "2021"

[dependencies]
transcribe-cpp = { version = "=0.5.0", default-features = false, features = ["serde"] }
serde = "1"
serde_json = "1"
postcard = { version = "1", default-features = false, features = ["alloc"] }

[workspace]
```

```rust
// src/main.rs, with src/values.rs and src/errors.rs linked to the files here.
// Usage: cargo run -- <this directory>
use std::path::PathBuf;
use transcribe_cpp as tc;

#[path = "values.rs"]
mod values;
#[path = "errors.rs"]
mod errors;

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect("out dir"));
    std::fs::create_dir_all(out.join("json")).unwrap();
    std::fs::create_dir_all(out.join("postcard")).unwrap();
    let mut all: Vec<(&str, String, Vec<u8>)> = values::goldens()
        .into_iter()
        .map(|g| (g.name, g.json, g.postcard))
        .collect();
    all.extend(errors::error_goldens());
    for (name, json, pc) in &all {
        let hex: String = pc.iter().map(|b| format!("{b:02x}")).collect();
        std::fs::write(out.join(format!("json/{name}.json")), format!("{json}\n")).unwrap();
        std::fs::write(out.join(format!("postcard/{name}.hex")), format!("{hex}\n")).unwrap();
    }
    println!("wrote {} goldens with transcribe-cpp {}", all.len(), tc::compiled_version());
}
```

`values.rs` sticks to API both 0.5.0 and this crate have: struct literals,
`Default` plus field assignment for the `#[non_exhaustive]` types with
`Default`, and JSON for the ones 0.5.0 gave no way to build.
