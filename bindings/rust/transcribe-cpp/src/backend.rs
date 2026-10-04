//! Backend module loading and compute-device discovery.
//!
//! In a static (or plain `shared`) build the compiled-in backends are already
//! registered and [`init_backends`] is a harmless no-op. In a `dynamic-backends`
//! build the compute backends are loadable modules; a host points the library at
//! the provider directory ONCE, before the first model load — use
//! [`init_backends_default`] when the modules sit next to libtranscribe, or
//! [`init_backends`] with an explicit bundled provider directory. See the C
//! header's backend-module section for the degradation contract.

use std::ffi::CString;
use std::path::Path;

use transcribe_cpp_sys as sys;

use crate::error::{check, Result};
use crate::result::{owned_opt_str, owned_str};
use crate::types::Backend;

/// ggml's vendor-agnostic class for a compute device, orthogonal to
/// [`Device::kind`] (which carries the vendor). Backends report this
/// classification themselves, so use it as a runtime hint rather than a
/// portable hardware-memory taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DeviceType {
    /// CPU using system memory.
    Cpu,
    /// Backend-reported GPU.
    Gpu,
    /// Backend-reported integrated GPU.
    Igpu,
    /// Host-memory accelerator (BLAS/AMX/...).
    Accel,
    /// A device-type value this binding does not recognize — reported by a
    /// runtime newer than the header this binding was generated against.
    /// Distinguish such devices by [`Device::device_id`] / [`Device::name`],
    /// not by this axis.
    Unknown,
}

impl DeviceType {
    fn from_raw(raw: sys::transcribe_device_type) -> Self {
        use sys::transcribe_device_type as T;
        match raw {
            T::TRANSCRIBE_DEVICE_TYPE_CPU => DeviceType::Cpu,
            T::TRANSCRIBE_DEVICE_TYPE_GPU => DeviceType::Gpu,
            T::TRANSCRIBE_DEVICE_TYPE_IGPU => DeviceType::Igpu,
            T::TRANSCRIBE_DEVICE_TYPE_ACCEL => DeviceType::Accel,
            _ => DeviceType::Unknown,
        }
    }
}

/// One registered compute device.
#[derive(Debug, Clone)]
pub struct Device {
    /// ggml device name, e.g. "Metal".
    pub name: String,
    /// Human-readable description, e.g. "Apple M4 Max".
    pub description: String,
    /// Classified vendor kind: "cpu", "accel", "metal", "vulkan", "cuda",
    /// "rocm", "sycl", "gpu", or "unknown".
    pub kind: String,
    /// The CPU/GPU/IGPU/ACCEL axis, orthogonal to [`Device::kind`].
    pub device_type: DeviceType,
    /// Stable hardware id when the backend reports one (PCI bus id for PCI
    /// devices), or `None` (e.g. Metal).
    pub device_id: Option<String>,
    /// Reported device memory capacity in bytes, or 0 if unreported.
    pub memory_total: u64,
    /// Available device memory in bytes — a snapshot at the time this was
    /// queried, or 0 if unreported. Re-query (via [`devices`] or
    /// [`crate::Model::device`]) to refresh it; the value is backend-defined
    /// and not comparable across device kinds.
    pub memory_free: u64,
    /// Registry index when this device came from [`devices`]. This is useful
    /// for display only; pass the [`Device`] itself to [`ModelOptions`] for
    /// exact selection. Registry indices are not stable across processes.
    pub index: Option<usize>,
    pub(crate) handle: sys::transcribe_device_t,
}

// Device handles are immutable process-lifetime registry entries. Backend
// registration must finish before enumeration, so sharing a handle is safe.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl Eq for Device {}

impl Device {
    /// Build a [`Device`] from the raw FFI struct filled by the library.
    /// `index` is the registry index when the device came from enumeration,
    /// or `None` when it came from a loaded model.
    pub(crate) fn from_raw(
        raw: &sys::transcribe_device_info,
        handle: sys::transcribe_device_t,
        index: Option<usize>,
    ) -> Device {
        Device {
            name: owned_str(raw.name),
            description: owned_str(raw.description),
            kind: owned_str(raw.kind),
            device_type: DeviceType::from_raw(raw.device_type),
            device_id: owned_opt_str(raw.device_id),
            memory_total: raw.memory_total,
            memory_free: raw.memory_free,
            index,
            handle,
        }
    }
}

/// Load backend modules from `dir` (dynamic builds) and register their
/// devices. A no-op in static builds. Call once, before the first model load.
///
/// Errors with [`crate::Error::Backend`] if, after loading, the process has no
/// registered compute device (a dynamic build pointed at a directory with no
/// usable modules). This call is idempotent per directory and NOT retryable in
/// the same process.
///
/// This mutates the native process-global device registry. Complete every
/// backend-init call before other threads enumerate devices, query backend
/// availability, or load models; the native registry does not support racing
/// registration against those operations.
pub fn init_backends(dir: impl AsRef<Path>) -> Result<()> {
    let dir = dir.as_ref();
    // Pass the path bytes through faithfully (Unix) / reject non-UTF-8 (Windows),
    // matching model loading — never lossily mangle a path with to_string_lossy.
    let c_dir = CString::new(crate::model::path_bytes(dir)?)?;
    let status = unsafe { sys::transcribe_init_backends(c_dir.as_ptr()) };
    check(status, "init_backends")
}

/// Load the backend modules the way the build expects, with no caller-supplied
/// path.
///
/// - **Compiled-in builds** (the default static build, or a plain `shared`
///   build): a no-op returning `Ok(())` — the backends are already registered.
/// - **`dynamic-backends` builds**: resolves the directory containing the
///   loaded libtranscribe and scans only that directory. Ship the backend
///   modules next to libtranscribe for this helper to find them.
///
/// If your app uses a different layout, call [`init_backends`] with that
/// resolved module directory instead. Like [`init_backends`], this is
/// idempotent and must run once before the first model load. It must also
/// complete before other threads enumerate devices or query backend
/// availability; see [`init_backends`] for the registry-ordering contract.
pub fn init_backends_default() -> Result<()> {
    let status = unsafe { sys::transcribe_init_backends_default() };
    check(status, "init_backends_default")
}

/// Backend kinds allowed to register in this process. A backend outside the
/// mask never runs any code, so a broken GPU driver can be kept out of a
/// worker entirely. CPU is always allowed; `TRANSCRIBE_BACKENDS` can only
/// narrow the mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BackendMask(u32);

impl BackendMask {
    /// CPU plus host-memory accelerators (BLAS). Always implied.
    pub const CPU: BackendMask = BackendMask(sys::TRANSCRIBE_BACKEND_MASK_CPU);
    /// Apple Metal.
    pub const METAL: BackendMask = BackendMask(sys::TRANSCRIBE_BACKEND_MASK_METAL);
    /// Vulkan.
    pub const VULKAN: BackendMask = BackendMask(sys::TRANSCRIBE_BACKEND_MASK_VULKAN);
    /// NVIDIA CUDA.
    pub const CUDA: BackendMask = BackendMask(sys::TRANSCRIBE_BACKEND_MASK_CUDA);
    /// AMD ROCm / HIP.
    pub const ROCM: BackendMask = BackendMask(sys::TRANSCRIBE_BACKEND_MASK_ROCM);
    /// Every backend without a dedicated bit (SYCL, OpenCL, RPC, ...).
    pub const OTHER: BackendMask = BackendMask(sys::TRANSCRIBE_BACKEND_MASK_OTHER);
    /// Everything. The default.
    pub const ALL: BackendMask = BackendMask(sys::TRANSCRIBE_BACKEND_MASK_ALL);

    /// The smallest mask that can serve a model-load `backend` request
    /// ([`Backend::Auto`] needs everything).
    pub const fn for_backend(backend: Backend) -> BackendMask {
        match backend {
            Backend::Auto => BackendMask::ALL,
            Backend::Cpu | Backend::CpuAccel => BackendMask::CPU,
            Backend::Metal => BackendMask::METAL.union(BackendMask::CPU),
            Backend::Vulkan => BackendMask::VULKAN.union(BackendMask::CPU),
            Backend::Cuda => BackendMask::CUDA.union(BackendMask::CPU),
            Backend::Rocm => BackendMask::ROCM.union(BackendMask::CPU),
        }
    }

    /// The raw `TRANSCRIBE_BACKEND_MASK_*` bits.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// A mask from raw `TRANSCRIBE_BACKEND_MASK_*` bits.
    pub const fn from_bits(bits: u32) -> BackendMask {
        BackendMask(bits)
    }

    /// Both masks' backends.
    pub const fn union(self, other: BackendMask) -> BackendMask {
        BackendMask(self.0 | other.0)
    }

    /// Whether every backend in `other` is in `self`.
    pub const fn contains(self, other: BackendMask) -> bool {
        self.0 & other.0 == other.0
    }
}

impl Default for BackendMask {
    fn default() -> Self {
        BackendMask::ALL
    }
}

impl std::ops::BitOr for BackendMask {
    type Output = BackendMask;
    fn bitor(self, rhs: BackendMask) -> BackendMask {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for BackendMask {
    fn bitor_assign(&mut self, rhs: BackendMask) {
        *self = self.union(rhs);
    }
}

/// [`init_backends`] / [`init_backends_default`] (`dir` = `None`) with an
/// allowed-backend mask. The mask is fixed at first backend registration;
/// call this first, once per process. A later call with a different
/// effective mask returns [`crate::Error::Backend`].
///
/// ```no_run
/// use transcribe_cpp::{init_backends_with, Backend, BackendMask};
/// // A CPU fallback worker: never let the GPU driver load.
/// init_backends_with(None::<&std::path::Path>, BackendMask::for_backend(Backend::Cpu))?;
/// # Ok::<(), transcribe_cpp::Error>(())
/// ```
pub fn init_backends_with(dir: Option<impl AsRef<Path>>, allowed: BackendMask) -> Result<()> {
    let c_dir = match dir {
        Some(dir) => Some(CString::new(crate::model::path_bytes(dir.as_ref())?)?),
        None => None,
    };
    let mut params: sys::transcribe_backend_init_params = unsafe { std::mem::zeroed() };
    unsafe { sys::transcribe_backend_init_params_init(&mut params) };
    params.artifact_dir = c_dir.as_ref().map_or(std::ptr::null(), |d| d.as_ptr());
    params.allowed_backends = allowed.bits();
    let status = unsafe { sys::transcribe_init_backends_ex(&params) };
    check(status, "init_backends_with")
}

/// The effective mask: [`init_backends_with`]'s (ALL until then) narrowed
/// by `TRANSCRIBE_BACKENDS`.
pub fn allowed_backends() -> BackendMask {
    BackendMask(unsafe { sys::transcribe_allowed_backends() })
}

/// The number of compute devices currently registered.
///
/// Do not race this query with [`init_backends`] or [`init_backends_default`].
pub fn device_count() -> usize {
    let n = unsafe { sys::transcribe_device_count() };
    n.max(0) as usize
}

/// Every registered compute device.
///
/// Do not race enumeration with [`init_backends`] or
/// [`init_backends_default`]. Finish backend registration before sharing
/// devices across threads.
pub fn devices() -> Vec<Device> {
    let mut out = Vec::with_capacity(device_count());
    for i in 0..device_count() as i32 {
        let handle = unsafe { sys::transcribe_device_get(i) };
        if handle.is_null() {
            continue;
        }
        let mut raw: sys::transcribe_device_info = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_device_info_init(&mut raw) };
        let status = unsafe { sys::transcribe_device_get_info(handle, &mut raw) };
        if status == sys::transcribe_status::TRANSCRIBE_OK {
            out.push(Device::from_raw(&raw, handle, Some(i as usize)));
        }
    }
    out
}

/// Whether a backend request can be satisfied by some registered device. This
/// is the probe to turn `Backend::Vulkan` on a machine without Vulkan into a
/// clear error instead of a failed model load. Do not race this query with
/// [`init_backends`] or [`init_backends_default`].
pub fn backend_available(backend: Backend) -> bool {
    unsafe { sys::transcribe_backend_available(backend.to_raw()) }
}
