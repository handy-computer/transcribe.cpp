import CTranscribe

/// A backend request. `auto` always succeeds (CPU is the final fallback);
/// `metal`/`vulkan`/`cuda` require that backend to be present in the build.
public enum Backend: Sendable, Equatable {
    case auto
    case cpu
    case cpuAccel
    case metal
    case vulkan
    case cuda

    var cValue: transcribe_backend_request {
        switch self {
        case .auto: return TRANSCRIBE_BACKEND_AUTO
        case .cpu: return TRANSCRIBE_BACKEND_CPU
        case .cpuAccel: return TRANSCRIBE_BACKEND_CPU_ACCEL
        case .metal: return TRANSCRIBE_BACKEND_METAL
        case .vulkan: return TRANSCRIBE_BACKEND_VULKAN
        case .cuda: return TRANSCRIBE_BACKEND_CUDA
        }
    }
}

/// The vendor-agnostic class of a compute device, orthogonal to `Device.kind`
/// (which carries the vendor). Distinguishes a discrete GPU from an integrated
/// one, and a host-memory accelerator from the CPU.
public enum DeviceType: Sendable, Equatable {
    case cpu
    case gpu
    case igpu
    case accel

    init(_ c: transcribe_device_type) {
        switch c.rawValue {
        case TRANSCRIBE_DEVICE_TYPE_CPU.rawValue: self = .cpu
        case TRANSCRIBE_DEVICE_TYPE_IGPU.rawValue: self = .igpu
        case TRANSCRIBE_DEVICE_TYPE_ACCEL.rawValue: self = .accel
        // includes TRANSCRIBE_DEVICE_TYPE_GPU and any unknown value
        default: self = .gpu
        }
    }
}

/// A registered compute device.
public struct Device: Sendable, Equatable {
    /// ggml device name, e.g. "Metal".
    public let name: String
    /// Human-readable description, e.g. "Apple M4 Max".
    public let description: String
    /// Classified vendor kind string, e.g. "cpu", "metal", "vulkan", "cuda".
    public let kind: String
    /// The CPU/GPU/IGPU/ACCEL axis, orthogonal to `kind`.
    public let deviceType: DeviceType
    /// Stable hardware id (PCI bus id) when the backend reports one, else nil
    /// (e.g. Metal).
    public let deviceId: String?
    /// Reported device memory capacity in bytes, or 0 if unreported.
    public let memoryTotal: UInt64
    /// Available device memory in bytes — a snapshot at query time, or 0 if
    /// unreported. Re-query (`TranscribeCpp.devices()` or `Model.device`) to
    /// refresh; backend-defined and not comparable across device kinds.
    public let memoryFree: UInt64

    /// Build from the raw C struct the library filled.
    init(_ raw: transcribe_backend_device) {
        name = raw.name.map { String(cString: $0) } ?? ""
        description = raw.description.map { String(cString: $0) } ?? ""
        kind = raw.kind.map { String(cString: $0) } ?? ""
        deviceType = DeviceType(raw.device_type)
        deviceId = raw.device_id.map { String(cString: $0) }
        memoryTotal = raw.memory_total
        memoryFree = raw.memory_free
    }
}

/// A public ABI struct, for the no-model layout-liveness check. Mirrors the
/// `transcribe_abi_struct` enum (only the members the bindings introspect).
public enum AbiStruct: Sendable {
    case modelLoadParams
    case sessionParams
    case runParams
    case streamParams
    case capabilities
    case timings
    case segment
    case word
    case token
    case streamUpdate
    case streamText
    case sessionLimits
    case ext
    case backendDevice

    var cValue: transcribe_abi_struct {
        switch self {
        case .modelLoadParams: return TRANSCRIBE_ABI_MODEL_LOAD_PARAMS
        case .sessionParams: return TRANSCRIBE_ABI_SESSION_PARAMS
        case .runParams: return TRANSCRIBE_ABI_RUN_PARAMS
        case .streamParams: return TRANSCRIBE_ABI_STREAM_PARAMS
        case .capabilities: return TRANSCRIBE_ABI_CAPABILITIES
        case .timings: return TRANSCRIBE_ABI_TIMINGS
        case .segment: return TRANSCRIBE_ABI_SEGMENT
        case .word: return TRANSCRIBE_ABI_WORD
        case .token: return TRANSCRIBE_ABI_TOKEN
        case .streamUpdate: return TRANSCRIBE_ABI_STREAM_UPDATE
        case .streamText: return TRANSCRIBE_ABI_STREAM_TEXT
        case .sessionLimits: return TRANSCRIBE_ABI_SESSION_LIMITS
        case .ext: return TRANSCRIBE_ABI_EXT
        case .backendDevice: return TRANSCRIBE_ABI_BACKEND_DEVICE
        }
    }
}
