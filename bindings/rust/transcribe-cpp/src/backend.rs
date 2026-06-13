//! Backend module loading and compute-device discovery.
//!
//! In a static build the compiled-in backends are already registered and
//! [`init_backends`] is a harmless no-op. In a dynamic (`dylib` feature) build
//! the compute backends are loadable modules; a host points the library at the
//! provider directory ONCE, before the first model load. See the C header's
//! backend-module section for the degradation contract.

use std::ffi::CString;
use std::path::Path;

use transcribe_cpp_sys as sys;

use crate::error::{check, Result};
use crate::result::owned_str;
use crate::types::Backend;

/// One registered compute device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    /// ggml device name, e.g. "Metal".
    pub name: String,
    /// Human-readable description, e.g. "Apple M4 Max".
    pub description: String,
    /// Classified kind: "cpu", "accel", "metal", "vulkan", "cuda", "sycl",
    /// "gpu", or "unknown".
    pub kind: String,
}

/// Load backend modules from `dir` (dynamic builds) and register their
/// devices. A no-op in static builds. Call once, before the first model load.
///
/// Errors with [`crate::Error::Backend`] if, after loading, the process has no
/// registered compute device (a dynamic build pointed at a directory with no
/// usable modules). This call is idempotent per directory and NOT retryable in
/// the same process.
pub fn init_backends(dir: impl AsRef<Path>) -> Result<()> {
    let dir = dir.as_ref();
    let c_dir = CString::new(dir.to_string_lossy().as_bytes())?;
    let status = unsafe { sys::transcribe_init_backends(c_dir.as_ptr()) };
    check(status, "init_backends")
}

/// The number of compute devices currently registered.
pub fn device_count() -> usize {
    let n = unsafe { sys::transcribe_backend_device_count() };
    n.max(0) as usize
}

/// Every registered compute device.
pub fn devices() -> Vec<Device> {
    let mut out = Vec::with_capacity(device_count());
    for i in 0..device_count() as i32 {
        let mut raw: sys::transcribe_backend_device = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_backend_device_init(&mut raw) };
        let status = unsafe { sys::transcribe_get_backend_device(i, &mut raw) };
        if status == sys::transcribe_status::TRANSCRIBE_OK {
            out.push(Device {
                name: owned_str(raw.name),
                description: owned_str(raw.description),
                kind: owned_str(raw.kind),
            });
        }
    }
    out
}

/// Whether a backend request can be satisfied by some registered device. This
/// is the probe to turn `Backend::Vulkan` on a machine without Vulkan into a
/// clear error instead of a failed model load.
pub fn backend_available(backend: Backend) -> bool {
    unsafe { sys::transcribe_backend_available(backend.to_raw()) }
}
