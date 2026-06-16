import CTranscribe
import Foundation

public enum LogLevel: Sendable {
    case none, info, warn, error, debug, continuation

    init(_ c: transcribe_log_level) {
        switch c {
        case TRANSCRIBE_LOG_LEVEL_INFO: self = .info
        case TRANSCRIBE_LOG_LEVEL_WARN: self = .warn
        case TRANSCRIBE_LOG_LEVEL_ERROR: self = .error
        case TRANSCRIBE_LOG_LEVEL_DEBUG: self = .debug
        case TRANSCRIBE_LOG_LEVEL_CONT: self = .continuation
        default: self = .none
        }
    }
}

/// Holds the global Swift log handler. The native sink is process-global and
/// may fire from any thread (including ggml worker threads), so access is
/// lock-guarded.
private final class LogState: @unchecked Sendable {
    static let shared = LogState()
    private let lock = NSLock()
    private var handler: (@Sendable (LogLevel, String) -> Void)?

    func set(_ h: (@Sendable (LogLevel, String) -> Void)?) {
        lock.lock(); handler = h; lock.unlock()
    }
    func current() -> (@Sendable (LogLevel, String) -> Void)? {
        lock.lock(); defer { lock.unlock() }
        return handler
    }
}

private func logTrampoline(
    _ level: transcribe_log_level, _ msg: UnsafePointer<CChar>?, _ userData: UnsafeMutableRawPointer?
) {
    let text = msg.map { String(cString: $0) } ?? ""
    LogState.shared.current()?(LogLevel(level), text)
}

extension Transcribe {
    /// Route native log messages (library + ggml diagnostics) to `handler`.
    /// Per the C contract, call once at process startup, before loading models
    /// or creating sessions. The handler may be invoked from any thread.
    public static func setLogHandler(_ handler: @escaping @Sendable (LogLevel, String) -> Void) {
        LogState.shared.set(handler)
        transcribe_log_set(logTrampoline, nil)
    }

    /// Disable logging entirely (library and ggml messages are dropped).
    public static func disableLogging() {
        LogState.shared.set(nil)
        transcribe_log_set(nil, nil)
    }
}
