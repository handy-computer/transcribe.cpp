import XCTest

@testable import TranscribeCpp

/// Happy-path model tests for the family stream extensions that the core
/// `ExtensionTests` couldn't reach (no canary): parakeet cache-aware, parakeet
/// buffered, and voxtral realtime. These gate on their own GGUF and XCTSkip
/// when absent (they run locally; the CI canary set doesn't include them).
/// Together with whisper-run and moonshine-streaming, this exercises EVERY
/// shipped extension kind end-to-end: the typed struct is materialized, handed
/// across the FFI, and accepted by a model that actually consumes it.
final class FamilyStreamTests: XCTestCase {
    // MARK: parakeet cache-aware (PARAKEET_STREAM)

    func testParakeetCacheAwareAcceptanceDiscriminates() throws {
        guard let path = Fixtures.parakeetStreamModelPath() else {
            throw XCTSkip("no parakeet cache-aware canary")
        }
        let model = try Model(path: path)
        // The header's documented discrimination: cache-aware accepts STREAM,
        // rejects BUFFERED.
        XCTAssertTrue(model.accepts(.parakeetStream(ParakeetStreamOptions())))
        XCTAssertFalse(model.accepts(.parakeetBuffered(ParakeetBufferedStreamOptions())))
    }

    func testParakeetCacheAwareStreamsWithExtension() throws {
        guard let path = Fixtures.parakeetStreamModelPath(), let audio = Fixtures.audioPath() else {
            throw XCTSkip("no parakeet cache-aware canary")
        }
        let pcm = try Fixtures.loadWav(audio)
        let session = try Model(path: path).session()
        let stream = try session.stream(
            .init(), StreamOptions(family: .parakeetStream(ParakeetStreamOptions(attContextRight: -1))))
        try Fixtures.drive(stream, pcm: pcm)
        XCTAssertTrue(stream.text.full.lowercased().contains("country"), stream.text.full)
    }

    // MARK: parakeet buffered (PARAKEET_BUFFERED_STREAM)

    func testParakeetBufferedStreamsWithExtension() throws {
        guard let path = Fixtures.parakeetBufferedModelPath(), let audio = Fixtures.audioPath() else {
            throw XCTSkip("no parakeet buffered canary")
        }
        let model = try Model(path: path)
        XCTAssertTrue(model.accepts(.parakeetBuffered(ParakeetBufferedStreamOptions())))
        let pcm = try Fixtures.loadWav(audio)
        // Defaults (left/chunk/right = -1) resolve to the model's menu default
        // (L=5600/C=1040/R=1040). An explicit override must be an 80 ms multiple
        // AND land on a tuple in the model's training menu, else stream_begin
        // returns INVALID_ARG — so the safe, path-proving choice is the default.
        let stream = try model.session().stream(
            .init(), StreamOptions(family: .parakeetBuffered(ParakeetBufferedStreamOptions())))
        try Fixtures.drive(stream, pcm: pcm)
        XCTAssertTrue(stream.text.full.lowercased().contains("country"), stream.text.full)
    }

    // MARK: voxtral realtime (VOXTRAL_REALTIME_STREAM)

    func testVoxtralRealtimeAcceptsAndStreams() throws {
        guard let path = Fixtures.voxtralRealtimeModelPath(), let audio = Fixtures.audioPath() else {
            throw XCTSkip("no voxtral realtime canary")
        }
        let model = try Model(path: path)
        XCTAssertTrue(model.accepts(.voxtralRealtime(VoxtralRealtimeStreamOptions())))
        let pcm = try Fixtures.loadWav(audio)
        let stream = try model.session().stream(
            .init(),
            StreamOptions(family: .voxtralRealtime(VoxtralRealtimeStreamOptions(numDelayTokens: 4))))
        try Fixtures.drive(stream, pcm: pcm)
        XCTAssertFalse(stream.text.full.isEmpty, "voxtral produced no text")
    }
}
