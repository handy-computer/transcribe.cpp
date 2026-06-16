import Foundation
import XCTest

@testable import TranscribeCpp

/// No-model tier (requirements §4): import/link, version gate, ABI liveness,
/// and backend discovery. These run always — no canary GGUFs required — and are
/// the Swift analogs of Rust's `no_model.rs` and Python's `test_abi.py` /
/// `test_backends.py`.
final class NoModelTests: XCTestCase {
    func testNativeVersionPresent() {
        XCTAssertFalse(Transcribe.version().isEmpty)
    }

    func testVersionGateAgrees() throws {
        // The linked library and the binding agree on the base version.
        try Transcribe.ensureCompatible()
        XCTAssertEqual(baseVersion(Transcribe.version()), baseVersion(Transcribe.compiledVersion))
    }

    func testAbiStructSizesAreLive() {
        // A real layout is non-zero; a garbage/empty one would be 0.
        for s in [AbiStruct.runParams, .capabilities, .segment, .sessionLimits] {
            XCTAssertGreaterThan(Transcribe.abiStructSize(s), 0, "\(s)")
        }
    }

    func testHeaderHashIsPinned() {
        // 16 hex chars (sha256/16). The value-vs-header drift is gated in CI by
        // scripts/ci/swift_abihash_check.py; here we just assert the shape.
        let hash = Transcribe.headerHash()
        XCTAssertEqual(hash.count, 16)
        XCTAssertTrue(hash.allSatisfy { $0.isHexDigit })
    }

    func testStatusStringIsActionable() {
        XCTAssertEqual(Transcribe.statusString(0), "ok")
        XCTAssertFalse(Transcribe.statusString(3).isEmpty)  // ERR_FILE_NOT_FOUND
    }

    func testAtLeastOneDevice() {
        XCTAssertGreaterThanOrEqual(Transcribe.devices().count, 1)
    }

    func testCpuIsAlwaysAvailable() {
        XCTAssertTrue(Transcribe.backendAvailable(.cpu))
        XCTAssertTrue(Transcribe.backendAvailable(.auto))
    }

    func testCpuDeviceIsRegistered() {
        XCTAssertTrue(Transcribe.devices().contains { $0.kind == "cpu" })
    }

    // Error-mapping integration (no canary needed): the two load-failure
    // classes map to distinct cases. Mirrors Rust's no_model error tests and
    // Python's test_errors.py integration cases.
    func testMissingFileIsModelFileNotFound() {
        XCTAssertThrowsError(try Model(path: "/no/such/transcribe-model.gguf")) { error in
            guard case TranscribeError.modelFileNotFound = error else {
                return XCTFail("expected .modelFileNotFound, got \(error)")
            }
        }
    }

    // The run-mode enum is `TranscriptionTask` (renamed off `Task` so it does
    // not shadow Swift's concurrency `Task`). Lock the public name + `task:`
    // option here so an accidental rename is caught without a model.
    func testTranscriptionTaskOptionRoundTrips() {
        let translate = RunOptions(task: .translate)
        guard case .translate = translate.task else {
            return XCTFail("task option did not round-trip to .translate")
        }
        let task: TranscriptionTask = .transcribe
        guard case .transcribe = task else { return XCTFail("TranscriptionTask.transcribe") }
    }

    func testJunkFileIsModelLoadError() throws {
        let junk = FileManager.default.temporaryDirectory
            .appendingPathComponent("junk-\(UUID().uuidString).gguf")
        try Data("not a gguf".utf8).write(to: junk)
        defer { try? FileManager.default.removeItem(at: junk) }
        XCTAssertThrowsError(try Model(path: junk.path)) { error in
            guard case TranscribeError.modelLoad = error else {
                return XCTFail("expected .modelLoad, got \(error)")
            }
        }
    }
}
