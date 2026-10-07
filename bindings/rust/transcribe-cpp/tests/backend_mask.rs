//! Allowed-backend mask. The mask is fixed per process, so every assertion
//! that touches the native registry lives in ONE test function in this file
//! (its own test binary, hence its own process).

use transcribe_cpp::{
    allowed_backends, backend_available, devices, init_backends_with, Backend, BackendMask,
    DeviceType, Error,
};

#[test]
fn for_backend_is_minimal() {
    assert_eq!(BackendMask::for_backend(Backend::Auto), BackendMask::ALL);
    assert_eq!(BackendMask::for_backend(Backend::Cpu), BackendMask::CPU);
    assert_eq!(
        BackendMask::for_backend(Backend::CpuAccel),
        BackendMask::CPU
    );
    assert_eq!(
        BackendMask::for_backend(Backend::Vulkan),
        BackendMask::VULKAN | BackendMask::CPU
    );
    assert!(BackendMask::ALL.contains(BackendMask::METAL | BackendMask::OTHER));
    assert!(!BackendMask::CPU.contains(BackendMask::VULKAN));
}

#[test]
fn cpu_only_worker() {
    let cpu = BackendMask::for_backend(Backend::Cpu);
    init_backends_with(None::<&std::path::Path>, cpu).expect("cpu-only init");
    assert_eq!(allowed_backends(), BackendMask::CPU);

    let devs = devices();
    assert!(!devs.is_empty());
    assert!(devs
        .iter()
        .all(|d| matches!(d.device_type, DeviceType::Cpu | DeviceType::Accel)));
    assert!(backend_available(Backend::Cpu));
    assert!(!backend_available(Backend::Vulkan));
    assert!(!backend_available(Backend::Metal));

    // Fixed for the process: the same mask is fine, a different one is not.
    init_backends_with(None::<&std::path::Path>, cpu).expect("same mask again");
    assert!(matches!(
        init_backends_with(None::<&std::path::Path>, BackendMask::ALL),
        Err(Error::Backend { .. })
    ));
}
