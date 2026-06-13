//! No-model tests: version/ABI introspection, the version gate, device
//! discovery, and error mapping. These always run (including on forks).

mod common;

use transcribe_cpp::{
    abi_struct_size, backend_available, compiled_version, device_count, devices, header_hash,
    version, AbiStruct, Backend, Error, Model,
};

#[test]
fn version_gate_agrees() {
    // The pre-1.0 base-version lock: the linked library and the generated
    // bindings must report the same MAJOR.MINOR.PATCH.
    assert_eq!(version(), compiled_version());
    assert!(!version().is_empty());
}

#[test]
fn header_hash_is_pinned() {
    // sha256/16 over the FFI surface — 16 hex chars.
    let h = header_hash();
    assert_eq!(h.len(), 16, "hash: {h}");
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn abi_struct_sizes_are_live() {
    // Real calls into the native library returning runtime layout values.
    for which in [
        AbiStruct::RunParams,
        AbiStruct::Capabilities,
        AbiStruct::Segment,
        AbiStruct::SessionLimits,
    ] {
        assert!(abi_struct_size(which) > 0, "{which:?} reported size 0");
    }
}

#[test]
fn at_least_a_cpu_device() {
    // A static build always has the CPU backend compiled in and registered.
    assert!(device_count() >= 1);
    assert!(backend_available(Backend::Cpu));
    let devices = devices();
    assert!(devices.iter().any(|d| d.kind == "cpu"), "{devices:?}");
}

#[test]
fn missing_file_is_not_found() {
    let err = Model::load("/no/such/model.gguf").unwrap_err();
    assert!(
        matches!(err, Error::ModelFileNotFound(_)),
        "got {err:?} (status {})",
        err.raw_status()
    );
}

#[test]
fn junk_file_is_model_load_error() {
    let dir = std::env::temp_dir();
    let path = dir.join("transcribe-rs-junk.gguf");
    std::fs::write(&path, b"not a gguf file at all").unwrap();
    let err = Model::load(&path).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert!(
        matches!(err, Error::ModelLoad(_)),
        "got {err:?} (status {})",
        err.raw_status()
    );
}

#[test]
fn handles_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}
    assert_send_sync::<Model>();
    assert_send::<transcribe_cpp::Session>();
    // Session is intentionally NOT Sync (single-threaded use).
}
