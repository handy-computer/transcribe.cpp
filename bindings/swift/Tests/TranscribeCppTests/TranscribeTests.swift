import XCTest

@testable import TranscribeCpp

/// Model-gated tier (requirements §4): real transcription against the canary.
/// Mirrors Rust's `transcribe.rs` and Python's `test_transcribe.py`. Each test
/// skips cleanly when the canary fixtures are absent.
final class TranscribeTests: XCTestCase {
    func testTranscribesJfkWithText() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        let transcript = try model.session().run(pcm)
        XCTAssertTrue(transcript.text.lowercased().contains("country"), transcript.text)
    }

    func testRequestedTimestampsPopulateSegments() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        let transcript = try model.session().run(pcm, options: RunOptions(timestamps: .segment))
        XCTAssertFalse(transcript.segments.isEmpty)
        XCTAssertNotEqual(transcript.timestampKind, .none)
    }

    func testFinerThanSupportedTimestampsIsUnsupported() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        guard let finer = finerThanSupported(model.capabilities.maxTimestampKind) else {
            throw XCTSkip("model already supports the finest timestamps")
        }
        let session = try model.session()
        XCTAssertThrowsError(try session.run(pcm, options: RunOptions(timestamps: finer))) { error in
            guard case TranscribeError.unsupported = error else {
                return XCTFail("expected .unsupported, got \(error)")
            }
        }
    }

    func testEmptyPcmIsInvalidArgument() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let session = try Model(path: path).session()
        XCTAssertThrowsError(try session.run([])) { error in
            guard case TranscribeError.invalidArgument = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
        }
    }

    func testRunBatchTwoUtterances() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        let results = try model.session().runBatch([pcm, pcm])
        XCTAssertEqual(results.count, 2)
        for result in results {
            let transcript = try result.get()
            XCTAssertTrue(transcript.text.lowercased().contains("country"), transcript.text)
        }
    }

    func testCapabilitiesAndIdentity() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let model = try Model(path: path)
        XCTAssertFalse(model.arch.isEmpty)
        XCTAssertFalse(model.backend.isEmpty)
        XCTAssertGreaterThan(model.capabilities.nativeSampleRate, 0)
    }

    func testPncChangesCanaryPrompt() throws {
        guard let path = Fixtures.pncModelPath(), let audio = Fixtures.audioPath() else {
            throw XCTSkip("PNC model/audio unavailable")
        }
        let model = try Model(path: path, options: ModelOptions(backend: .cpu))
        XCTAssertTrue(model.supports(.pnc))
        let session = try model.session()
        let pcm = try Fixtures.loadWav(audio)
        let run = { (pnc: Pnc) in
            try session.run(pcm, options: RunOptions(pnc: pnc, language: "en")).text
        }
        let defaultText = try run(.default)
        let enabled = try run(.on)
        let disabled = try run(.off)
        XCTAssertEqual(defaultText, enabled)
        XCTAssertNotEqual(disabled, enabled)
        XCTAssertEqual(disabled, disabled.lowercased())
    }

    func testItnChangesSenseVoiceTextNormalization() throws {
        guard let path = Fixtures.itnModelPath(), let audio = Fixtures.audioPath() else {
            throw XCTSkip("ITN model/audio unavailable")
        }
        let model = try Model(path: path, options: ModelOptions(backend: .cpu))
        XCTAssertTrue(model.supports(.itn))
        let session = try model.session()
        let pcm = try Fixtures.loadWav(audio)
        let run = { (itn: Itn) in
            try session.run(pcm, options: RunOptions(itn: itn, language: "en"))
        }
        let defaultResult = try run(.default)
        let disabled = try run(.off)
        let enabled = try run(.on)
        XCTAssertEqual(defaultResult.text, disabled.text)
        XCTAssertEqual(defaultResult.rawText, disabled.rawText)
        XCTAssertNotEqual(enabled.text, disabled.text)
        XCTAssertTrue(disabled.rawText.contains("<|woitn|>"))
        XCTAssertTrue(enabled.rawText.contains("<|withitn|>"))
    }

    func testSessionLimitsAreSane() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let limits = try Model(path: path).session().limits
        XCTAssertGreaterThanOrEqual(limits.effectiveNCtx, 0)
        XCTAssertGreaterThanOrEqual(limits.maxKvBytes, 0)
    }

    func testOneModelManySessions() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        for _ in 0..<2 {
            let transcript = try model.session().run(pcm)
            XCTAssertTrue(transcript.text.lowercased().contains("country"))
        }
    }

    func testCloseOrderingSessionOutlivesModelReference() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        // Drop the local Model reference; the Session's strong ref must keep the
        // native model alive (close-ordering safety under ARC).
        let session: Session = try {
            let model = try Model(path: path)
            return try model.session()
        }()
        let transcript = try session.run(pcm)
        XCTAssertTrue(transcript.text.lowercased().contains("country"))
    }

    func testSharedModelAcrossThreadsSerializes() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        let group = DispatchGroup()
        let lock = NSLock()
        var hits = 0
        for _ in 0..<2 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                // Each thread uses its own session; the per-model lock serializes
                // the actual compute.
                if let transcript = try? model.session().run(pcm),
                    transcript.text.lowercased().contains("country") {
                    lock.lock(); hits += 1; lock.unlock()
                }
            }
        }
        group.wait()
        XCTAssertEqual(hits, 2)
    }

    func testAsyncRun() async throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        let transcript = try await model.session().run(pcm)
        XCTAssertTrue(transcript.text.lowercased().contains("country"), transcript.text)
    }

    func testAsyncRunBatch() async throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        let results = try await model.session().runBatch([pcm, pcm])
        XCTAssertEqual(results.count, 2)
        for result in results {
            XCTAssertTrue(try result.get().text.lowercased().contains("country"))
        }
    }
}

/// The granularity strictly finer than `kind`, or nil if already finest.
/// Ordering for the run-params ceiling: none < segment < word < token.
private func finerThanSupported(_ kind: TimestampKind) -> TimestampKind? {
    switch kind {
    case .none: return .segment
    case .segment: return .word
    case .word: return .token
    case .token, .auto: return nil
    }
}
