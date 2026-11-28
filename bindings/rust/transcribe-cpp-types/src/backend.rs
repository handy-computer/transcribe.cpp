//! Backend and device plain data: [`DeviceType`] and [`BackendMask`]. The
//! `transcribe_cpp::Device` handle and backend loading stay in transcribe-cpp.

use crate::types::Backend;

/// ggml's vendor-agnostic class for a compute device, orthogonal to
/// `transcribe_cpp::Device::kind` (which carries the vendor). Backends report
/// this classification themselves, so use it as a runtime hint rather than a
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
    /// Distinguish such devices by `transcribe_cpp::Device::device_uuid` /
    /// `transcribe_cpp::Device::device_id` / `transcribe_cpp::Device::name`,
    /// not by this axis.
    Unknown,
}

/// Backend kinds allowed to register in this process. A backend outside the
/// mask never runs any code, so a broken GPU driver can be kept out of a
/// worker entirely. CPU is always allowed; `TRANSCRIBE_BACKENDS` can only
/// narrow the mask.
///
/// The constants are the C header's `TRANSCRIBE_BACKEND_MASK_*` values;
/// transcribe-cpp tests them against it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BackendMask(u32);

impl BackendMask {
    /// CPU plus host-memory accelerators (BLAS). Always implied.
    pub const CPU: BackendMask = BackendMask(1);
    /// Apple Metal.
    pub const METAL: BackendMask = BackendMask(1 << 1);
    /// Vulkan.
    pub const VULKAN: BackendMask = BackendMask(1 << 2);
    /// NVIDIA CUDA.
    pub const CUDA: BackendMask = BackendMask(1 << 3);
    /// AMD ROCm / HIP.
    pub const ROCM: BackendMask = BackendMask(1 << 4);
    /// Every backend without a dedicated bit (SYCL, OpenCL, RPC, ...).
    pub const OTHER: BackendMask = BackendMask(1 << 31);
    /// Everything. The default.
    pub const ALL: BackendMask = BackendMask(u32::MAX);

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
