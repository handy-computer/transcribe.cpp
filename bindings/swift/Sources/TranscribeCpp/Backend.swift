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

/// A registered compute device.
public struct Device: Sendable, Equatable {
    /// ggml device name, e.g. "Metal".
    public let name: String
    /// Human-readable description, e.g. "Apple M4 Max".
    public let description: String
    /// Classified kind string, e.g. "cpu", "metal", "vulkan", "cuda".
    public let kind: String
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
