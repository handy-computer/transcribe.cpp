import XCTest

@testable import TranscribeCpp

/// Family extensions + utilities. Mirrors Rust's `extensions.rs`. Note: the
/// "stream extension on the run slot" rejection the C API enforces at runtime
/// is enforced at COMPILE time here — `RunOptions.family` only accepts a
/// `RunExtension`, `StreamOptions.family` only a `StreamExtension` — so that
/// case is covered by the type system, not a test.
final class ExtensionTests: XCTestCase {
    func testWhisperAcceptsRunExtension() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let model = try Model(path: path)
        XCTAssertTrue(model.accepts(.whisper(WhisperRunOptions())))
    }

    func testWhisperRunWithInitialPrompt() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        let options = RunOptions(
            family: .whisper(WhisperRunOptions(initialPrompt: "Ask not", temperature: 0.0)))
        let transcript = try model.session().run(pcm, options: options)
        XCTAssertTrue(transcript.text.lowercased().contains("country"), transcript.text)
    }

    func testWrongFamilyExtensionIsRejected() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let model = try Model(path: path)
        // whisper has no streaming surface, so it accepts no stream extension.
        XCTAssertFalse(model.accepts(.parakeetStream(ParakeetStreamOptions())))
        XCTAssertFalse(model.accepts(.moonshineStreaming(MoonshineStreamingOptions())))
    }

    func testTokenizeRoundTripsNonEmpty() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let model = try Model(path: path)
        let tokens = try model.tokenize("ask not what your country can do for you")
        XCTAssertFalse(tokens.isEmpty)
        // A longer string tokenizes to at least as many tokens as a prefix.
        let prefix = try model.tokenize("ask not")
        XCTAssertGreaterThanOrEqual(tokens.count, prefix.count)
    }
}
