//! Build script for the safe `transcribe-cpp` crate.
//!
//! The native build and link live entirely in `transcribe-cpp-sys`; this script
//! exists only to carry two pieces of that crate's `links` metadata onto the
//! safe crate's OWN test/example/bin link line — `cargo:rustc-link-arg` does
//! NOT propagate from a dependency's build script to its dependents, so the
//! rpath the sys crate emits for itself never reaches here on its own.
//!
//! 1. **shared-lib rpath.** In the `shared` posture the consumer binary must
//!    find `libtranscribe` at runtime. On Unix that is an rpath to the sys
//!    crate's installed lib dir (`DEP_TRANSCRIBE_LIB_DIR`); Windows has no
//!    rpath (the DLL resolves via PATH / the exe dir) so nothing is emitted —
//!    a Windows shared consumer puts the lib dir on PATH instead.
//!
//! In the default static posture this branch is inert (no shared lib), so this
//! script is a near no-op.

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_TRANSCRIBE_LIB_DIR");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // `shared` is set whenever the lib is a separate file — both the plain
    // `shared` feature and `dynamic-backends` (which enables `shared` here).
    let shared = env::var_os("CARGO_FEATURE_SHARED").is_some();

    // Mirror the sys crate's runtime rpath onto THIS crate's binaries (tests,
    // examples), which the dependency's rustc-link-arg cannot reach.
    if shared && target_os != "windows" {
        if let Some(lib_dir) = env::var_os("DEP_TRANSCRIBE_LIB_DIR") {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                Path::new(&lib_dir).display()
            );
        }
    }
}
