//! Backend degradation + dynamic-posture contract.
//!
//! These are no-model tests, so they run in BOTH CI postures — the static
//! default AND the `dylib` (shared-link) leg of the posture matrix
//! (`rust-ci.yml`). In the dylib leg they are also the end-to-end proof that a
//! shared-linked consumer loads and calls into `libtranscribe` at runtime
//! (resolved via the rpath the safe crate's build.rs emits).
//!
//! They cover the REQUEST-PATH half of the degradation contract
//! (requirements §5): CPU is always the floor, an unavailable backend probes
//! cleanly (the hook that turns an explicit `Backend::Vulkan` into a clear
//! error rather than a crashed model load), and `init_backends` — the
//! provider/DL entry point — is callable without tearing down the floor. The
//! Vulkan-loader-absent three-tier (no loader / loader-no-device / real GPU) is
//! certified once at the C level (native-ci `provider-dl-vulkan`) and inherited
//! per requirements §4.

use transcribe_cpp::{backend_available, device_count, devices, init_backends, Backend};

#[test]
fn cpu_is_the_floor() {
    // Whatever the posture, a CPU device is always registered and reported.
    assert!(device_count() >= 1, "no compute devices registered");
    assert!(backend_available(Backend::Cpu), "CPU backend unavailable");
    assert!(
        devices().iter().any(|d| d.kind == "cpu"),
        "no cpu-kind device in {:?}",
        devices()
    );
}

#[test]
fn unavailable_backend_probes_cleanly() {
    // In a CPU/Metal build Vulkan and CUDA are not compiled in. The probe must
    // answer (true/false) without crashing — this is what lets a host turn an
    // explicit Backend::Vulkan request into a clear error instead of a failed
    // model load. We assert it does not panic and is internally consistent;
    // CPU is the one backend we can assert positively on every runner.
    let _vulkan = backend_available(Backend::Vulkan);
    let _cuda = backend_available(Backend::Cuda);
    assert!(backend_available(Backend::Cpu));
}

#[test]
fn init_backends_keeps_the_cpu_floor() {
    // The dylib/provider entry point. In a compiled-in build (static, or the
    // shared non-DL dylib posture) it is an idempotent no-op; the contract is
    // that calling it on a real directory neither panics nor tears down the
    // already-registered CPU floor. Mirrors link_smoke.c's init_backends call.
    let dir = std::env::temp_dir().join("transcribe-rs-empty-modules");
    std::fs::create_dir_all(&dir).ok();

    let before = device_count();
    let _ = init_backends(&dir); // posture-dependent result; must not panic
    assert!(
        device_count() >= before.max(1),
        "device count regressed after init_backends ({before} -> {})",
        device_count()
    );
    assert!(backend_available(Backend::Cpu), "CPU floor lost");
}
