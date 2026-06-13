//! Build script for `transcribe-cpp-sys`.
//!
//! Source build is the primary (and currently only) path: the `cmake` crate
//! drives the vendored C++ tree with `TRANSCRIBE_INSTALL=ON`, and the link
//! line is reconstructed from NOTHING but the installed
//! `lib/transcribe-link.json` manifest — the same artifact the `link_smoke`
//! CI lane compiles a toy C consumer against. No per-platform link lists are
//! hardcoded here (the whisper-rs drift class this avoids).
//!
//! Cargo features map directly to CMake options:
//!   `dylib`  -> TRANSCRIBE_BUILD_SHARED=ON (default: static)
//!   `metal`  -> TRANSCRIBE_METAL=ON   (Apple targets only; no-op elsewhere)
//!   `vulkan` -> TRANSCRIBE_VULKAN=ON
//!   `cuda`   -> TRANSCRIBE_CUDA=ON
//!   `openmp` -> TRANSCRIBE_USE_OPENMP=ON
//! Official-artifact hygiene flags (OpenMP/BLAS off) are deliberately NOT
//! forced here: a source build is the consumer's build (same philosophy as
//! the Python sdist).

use std::env;
use std::path::{Path, PathBuf};

fn feature(name: &str) -> bool {
    env::var_os(format!("CARGO_FEATURE_{name}")).is_some()
}

fn main() {
    // CARGO_MANIFEST_DIR for this crate is the repo root (the sys crate's
    // manifest lives there so the tarball can carry the whole C++ tree).
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    for p in [
        "CMakeLists.txt",
        "CMakePresets.json",
        "include",
        "src",
        "ggml",
        "cmake",
        "bindings/rust/sys/build.rs",
        "bindings/rust/sys/src",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(p).display());
    }

    let dylib = feature("DYLIB");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_apple = matches!(target_os.as_str(), "macos" | "ios");

    let mut cfg = cmake::Config::new(&root);
    cfg.profile("Release") // a transcription library always wants an optimized native core
        .define("TRANSCRIBE_INSTALL", "ON")
        .define("TRANSCRIBE_BUILD_TESTS", "OFF")
        .define("TRANSCRIBE_BUILD_EXAMPLES", "OFF")
        .define("TRANSCRIBE_BUILD_TOOLS", "OFF")
        .define("TRANSCRIBE_BUILD_SHARED", if dylib { "ON" } else { "OFF" });

    // Metal is the Apple default; the feature is a no-op on other targets
    // (CMake already defaults TRANSCRIBE_METAL OFF there).
    if feature("METAL") && is_apple {
        cfg.define("TRANSCRIBE_METAL", "ON");
        // Self-contained installed tree: embed the metallib instead of a
        // sidecar default.metallib next to the lib (matches the shipped macOS
        // wheel posture; what the shared-infra link-smoke uses).
        cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
    }
    if feature("VULKAN") {
        cfg.define("TRANSCRIBE_VULKAN", "ON");
    }
    if feature("CUDA") {
        cfg.define("TRANSCRIBE_CUDA", "ON");
    }
    // Force OpenMP OFF unless explicitly opted in. TRANSCRIBE_USE_OPENMP
    // defaults ON and auto-detects, but its `-fopenmp` shows up only as a
    // manifest link_flag → a `cargo:rustc-link-arg` that does NOT propagate to
    // downstream binaries, so a static consumer link fails with undefined
    // GOMP_*/omp_* symbols. A self-contained static build is the default;
    // `--features openmp` opts in (and then owns providing the OpenMP runtime).
    cfg.define(
        "TRANSCRIBE_USE_OPENMP",
        if feature("OPENMP") { "ON" } else { "OFF" },
    );
    // ggml's non-OpenMP CPU threadpool barrier (ggml_barrier's custom spin path)
    // DEADLOCKS under MSVC codegen on Windows: every multi-threaded CPU run wedges
    // its workers in the barrier spin (it works on every other compiler). The
    // OpenMP `#pragma omp barrier` path is the only working CPU threading there,
    // so build ggml with OpenMP on Windows even though our TRANSCRIBE_USE_OPENMP
    // knob stays off. GGML-internal only: TRANSCRIBE_USE_OPENMP=OFF keeps the link
    // manifest free of the GNU-only `-fopenmp` flag and leaves the Parakeet
    // host-decoder TU alone; MSVC auto-links vcomp via the /openmp pragma in the
    // ggml objects (vcomp140.dll ships in the VC++ runtime the binary needs
    // anyway). CMakeLists honors this explicit GGML_OPENMP and skips its force-off.
    if target_os == "windows" {
        cfg.define("GGML_OPENMP", "ON");
    }

    // Builds + installs into OUT_DIR; the returned path IS the install prefix.
    let prefix = cfg.build();

    let manifest = find_manifest(&prefix)
        .unwrap_or_else(|| panic!("transcribe-link.json not found under {}", prefix.display()));
    emit_link_lines(&prefix, &manifest);
}

/// GNUInstallDirs picks `lib` or `lib64`; find the manifest under either.
fn find_manifest(prefix: &Path) -> Option<PathBuf> {
    for libdir in ["lib", "lib64"] {
        let p = prefix.join(libdir).join("transcribe-link.json");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn emit_link_lines(prefix: &Path, manifest_path: &Path) {
    let text = std::fs::read_to_string(manifest_path).expect("read transcribe-link.json");
    let json: serde_json::Value = serde_json::from_str(&text).expect("parse transcribe-link.json");

    let strs = |key: &str| -> Vec<String> {
        json[key]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };

    let shared = json["shared"].as_bool().unwrap_or(false);
    let lib_dir = prefix.join(json["lib_dir"].as_str().unwrap_or("lib"));
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Archives (static) or the single shared lib. The manifest order is
    // single-pass-GNU-ld safe (each archive's undefined refs resolve in a
    // later one: transcribe -> ggml -> backends -> ggml-base).
    let kind = if shared { "dylib" } else { "static" };
    for name in strs("libraries") {
        println!("cargo:rustc-link-lib={kind}={name}");
    }

    // Absolute library paths (e.g. a find_package(BLAS) result): link the file.
    for path in strs("library_paths") {
        println!("cargo:rustc-link-arg={path}");
    }
    // System libraries the C++/backend archives drag in.
    for name in strs("system_libs") {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    // Apple frameworks (Metal/Foundation/Accelerate...).
    for name in strs("frameworks") {
        println!("cargo:rustc-link-lib=framework={name}");
    }
    // Extra link flags (e.g. -fopenmp).
    for flag in strs("link_flags") {
        println!("cargo:rustc-link-arg={flag}");
    }
    // Shared posture: the installed libs carry $ORIGIN/@loader_path rpaths, but
    // the consumer binary still needs to find lib_dir itself. Unix uses an
    // rpath; Windows has no rpath (the DLL resolves via PATH / the exe dir), so
    // nothing is emitted there. NOTE: rustc-link-arg does NOT propagate to
    // downstream crates, so this rpath only reaches THIS crate's own artifacts
    // (the -sys smoke tests). The safe crate re-emits it for its tests/examples
    // — see bindings/rust/transcribe-cpp/build.rs.
    let is_windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if shared && !is_windows {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
    // Windows has no rpath: a shared-posture binary resolves transcribe.dll +
    // the ggml DLLs from its OWN directory. Stage the installed DLLs next to
    // every artifact cargo produces so tests/examples/bins run in place.
    if shared && is_windows {
        stage_windows_dlls(prefix);
    }

    // Metadata for the safe crate's build script (available as
    // DEP_TRANSCRIBE_* because this crate sets `links = "transcribe"`).
    println!(
        "cargo:include_dir={}",
        prefix
            .join(json["include_dir"].as_str().unwrap_or("include"))
            .display()
    );
    println!("cargo:lib_dir={}", lib_dir.display());
    if json["backend_dl"].as_bool().unwrap_or(false) {
        if let Some(md) = json["module_dir"].as_str() {
            println!("cargo:module_dir={}", prefix.join(md).display());
        }
    }
}

/// Copy the installed runtime DLLs (`<prefix>/bin/*.dll` — transcribe.dll plus
/// the ggml DLLs) next to every artifact cargo will build, so a shared-posture
/// consumer's tests/examples/bins find them with no rpath (Windows has none)
/// and no PATH fiddling. Windows + `dylib` only; a no-op anywhere else.
///
/// Windows resolves a process's DLLs from the EXE's own directory first, and
/// cargo emits tests into `deps/`, examples into `examples/`, and bins into the
/// profile root — so all three get the DLLs. This runs from the -sys build
/// script (it owns the native build), which is always in the dep graph, so it
/// covers the safe crate's artifacts too. Best-effort: copy failures warn
/// rather than fail the build.
fn stage_windows_dlls(prefix: &Path) {
    let bin_dir = prefix.join("bin");
    let dlls: Vec<PathBuf> = match std::fs::read_dir(&bin_dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("dll"))
            .collect(),
        Err(_) => Vec::new(),
    };
    if dlls.is_empty() {
        println!(
            "cargo:warning=transcribe-cpp-sys: no DLLs under {} to stage for the dylib posture",
            bin_dir.display()
        );
        return;
    }
    // OUT_DIR = <target>/<profile>/build/<crate>-<hash>/out; the profile root is
    // four ancestors up. tests -> deps/, examples -> examples/, bins -> root.
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    for sub in ["", "deps", "examples"] {
        let dest = if sub.is_empty() {
            profile_dir.to_path_buf()
        } else {
            profile_dir.join(sub)
        };
        let _ = std::fs::create_dir_all(&dest);
        for dll in &dlls {
            if let Some(name) = dll.file_name() {
                let _ = std::fs::copy(dll, dest.join(name));
            }
        }
    }
    // Re-stage when the built DLLs change.
    println!("cargo:rerun-if-changed={}", bin_dir.display());
}
