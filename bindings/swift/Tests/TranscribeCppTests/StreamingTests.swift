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
}
