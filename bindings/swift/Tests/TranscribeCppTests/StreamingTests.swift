import XCTest

@testable import TranscribeCpp

/// Streaming tier (moonshine-streaming canary). Mirrors Rust's `streaming.rs`
/// and Python's `test_streaming.py`.
final class StreamingTests: XCTestCase {
    func testStreamsJfkCommittedText() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let session = try Model(path: path).session()
        let stream = try session.stream()
        try Fixtures.drive(stream, pcm: pcm)
        XCTAssertTrue(stream.text.full.lowercased().contains("country"), stream.text.full)
        let snapshot = stream.snapshot
        XCTAssertFalse(snapshot.text.isEmpty)
        if let language = snapshot.language { XCTAssertFalse(language.isEmpty) }
        XCTAssertFalse(snapshot.segments.isEmpty)
        _ = snapshot.words
        _ = snapshot.tokens
    }

    func testOnFinalizePolicyCommitsAtFinalize() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let session = try Model(path: path).session()
        let stream = try session.stream(.init(), StreamOptions(commitPolicy: .onFinalize))
        // Feed everything WITHOUT finalizing: committed stays empty under ON_FINALIZE.
        var i = 0
        while i < pcm.count {
            let end = min(i + 1600, pcm.count)
            _ = try stream.feed(Array(pcm[i..<end]))
            i = end
        }
        XCTAssertTrue(stream.text.committed.isEmpty, "committed before finalize: \(stream.text.committed)")
        try stream.finalize()
        XCTAssertFalse(stream.text.committed.isEmpty)
    }

    func testStreamRevisionAdvances() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let session = try Model(path: path).session()
        let stream = try session.stream()
        try Fixtures.drive(stream, pcm: pcm)
        XCTAssertGreaterThan(stream.revision, 0)
        XCTAssertNil(stream.lastStatus, "a healthy stream has no failure status")
    }

    func testStreamResetReturnsToIdleAndIsReusable() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let session = try Model(path: path).session()
        let stream = try session.stream()
        try Fixtures.drive(stream, pcm: pcm)
        XCTAssertEqual(stream.reset(), .idle)
        // Reusable: a new stream can begin on the same session.
        let again = try session.stream()
        _ = try again.feed(Array(pcm.prefix(1600)))
        try again.finalize()
    }

    func testNonStreamingModelIsRejected() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let session = try Model(path: path).session()  // whisper: not a streaming model
        XCTAssertThrowsError(try session.stream()) { error in
            guard case TranscribeError.notImplemented = error else {
                return XCTFail("expected .notImplemented, got \(error)")
            }
        }
    }

    /// Regression: the language hint must be copied at begin (not aliased to a
    /// freed Swift string). A run that streams cleanly with a hint proves the
    /// param-retention contract on the Swift side.
    func testStreamingWithLanguageHint() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let session = try Model(path: path).session()
        let stream = try session.stream(RunOptions(language: "en"))
        try Fixtures.drive(stream, pcm: pcm)
        XCTAssertFalse(stream.text.full.isEmpty)
    }

    // MARK: - Compute lease (C contract: one in-flight run/stream per model)

    /// 2b: a `Stream` dropped without `finalize()`/`reset()` must reset the
    /// session and release the model's compute lease in `deinit` — otherwise
    /// the session is wedged ACTIVE forever and the model stays `.busy`.
    func testDroppedActiveStreamReleasesLeaseAndResetsSession() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let model = try Model(path: path)
        let session = try model.session()
        do {
            let stream = try session.stream()
            _ = try stream.feed(Array(pcm.prefix(1600)))  // ACTIVE, holds the lease
            XCTAssertTrue(model.streamActive)
            // stream is dropped at the end of this scope WITHOUT finalize/reset
        }
        XCTAssertFalse(model.streamActive, "a dropped active stream must release the model lease")
        // The session is reusable: a fresh stream begins (would throw if the
        // session were still ACTIVE).
        let again = try session.stream()
        XCTAssertEqual(again.state, .active)
        again.reset()
    }

    /// The lease tracks "a stream is ACTIVE", not "a Stream handle exists":
    /// after finalize() or reset() another session may proceed WITHOUT waiting
    /// for the first Stream to drop. Mirrors Rust's
    /// `compute_lease_frees_at_finalize_and_reset`.
    func testFinalizeAndResetReleaseLeaseBeforeDrop() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let model = try Model(path: path)
        let s1 = try model.session()
        let s2 = try model.session()
        let chunk = Array(pcm.prefix(1600))

        // finalize() frees the lease even while stream1 is still in scope.
        do {
            let stream1 = try s1.stream()
            _ = try stream1.feed(chunk)
            try stream1.finalize()
            XCTAssertFalse(model.streamActive)
            let stream2 = try s2.stream()  // would throw .busy if the lease leaked
            stream2.reset()
            _ = stream1  // keep stream1 alive past s2.stream() to prove finalize freed it
        }
        // reset() frees the lease even while stream1 is still in scope.
        do {
            let stream1 = try s1.stream()
            _ = try stream1.feed(chunk)
            stream1.reset()
            let stream2 = try s2.stream()
            stream2.reset()
            _ = stream1
        }
    }

    /// While a stream is active, run / runBatch / stream on any session of the
    /// model throw `.busy`.
    func testActiveStreamRefusesSiblingCompute() throws {
        let (path, pcm) = try Fixtures.streamingModelAndAudio()
        let model = try Model(path: path)
        let s1 = try model.session()
        let s2 = try model.session()
        let active = try s1.stream()
        _ = try active.feed(Array(pcm.prefix(1600)))

        let cases: [(String, () throws -> Void)] = [
            ("a stream is active on this model; finish or drop it before run()", { _ = try s2.run(pcm) }),
            ("a stream is active on this model; finish or drop it before runBatch()",
             { _ = try s2.runBatch([pcm]) }),
            ("a stream is already active on this model", { _ = try s2.stream() }),
        ]
        for (message, call) in cases {
            XCTAssertThrowsError(try call()) { error in
                guard case TranscribeError.busy(let got) = error else {
                    return XCTFail("expected .busy, got \(error)")
                }
                XCTAssertEqual(got, message)
            }
        }
        active.reset()
    }
}
