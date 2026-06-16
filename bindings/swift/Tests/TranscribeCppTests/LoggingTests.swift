import Foundation
import XCTest

@testable import TranscribeCpp

/// Log routing (requirements: native log callback surfaced idiomatically).
/// No-model: `initBackends` emits a device summary through the sink.
final class LoggingTests: XCTestCase {
    private final class LogBox: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: [(LogLevel, String)] = []
        func append(_ level: LogLevel, _ message: String) {
            lock.lock(); stored.append((level, message)); lock.unlock()
        }
        var count: Int { lock.lock(); defer { lock.unlock() }; return stored.count }
    }

    func testLogHandlerReceivesNativeMessages() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let session = try Model(path: path).session()
        _ = try session.run(pcm)
        let box = LogBox()
        Transcribe.setLogHandler { level, message in box.append(level, message) }
        defer { Transcribe.disableLogging() }
        // printTimings publishes through the sink at INFO (same path Python's
        // log-routing test uses).
        session.printTimings()
        XCTAssertGreaterThan(box.count, 0, "expected printTimings output to route to the handler")
    }

    func testDisableLoggingIsSafe() {
        Transcribe.setLogHandler { _, _ in }
        Transcribe.disableLogging()
        // Re-disabling and operating afterwards must not crash.
        Transcribe.disableLogging()
        XCTAssertGreaterThanOrEqual(Transcribe.devices().count, 1)
    }
}
