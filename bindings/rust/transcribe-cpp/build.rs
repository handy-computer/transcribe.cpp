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
//! 2. **module dir.** A `GGML_BACKEND_DL` build compiles in no backends and
//!    needs `transcribe_init_backends(dir)`. When the sys build advertises a
//!    module dir (`DEP_TRANSCRIBE_MODULE_DIR`), forward it as a compile-time
//!    env so examples/tests can locate it via `option_env!`.
//!
//! In the default static posture both branches are inert (no shared lib, no
//! module dir), so this script is a near no-op.

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_TRANSCRIBE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DEP_TRANSCRIBE_MODULE_DIR");

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

    // Surface the backend-module directory (DL builds only) to example/test code
    // as TRANSCRIBE_MODULE_DIR. Absent in compiled-in builds → option_env! None.
    if let Some(module_dir) = env::var_os("DEP_TRANSCRIBE_MODULE_DIR") {
        println!(
            "cargo:rustc-env=TRANSCRIBE_MODULE_DIR={}",
            Path::new(&module_dir).display()
        );
    }
}
