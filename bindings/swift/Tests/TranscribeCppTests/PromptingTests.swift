import CTranscribe
import XCTest

@testable import TranscribeCpp

/// Generic prompting (vocabulary / prompt / prefix). The marshalling tests run
/// without a model; the rest use the whisper and streaming canaries. Mirrors
/// bindings/python/tests/test_prompting.py.
final class PromptingTests: XCTestCase {
    func testEnumValues() {
        XCTAssertEqual(TranscriptionTask.instruct.cValue.rawValue, 2)
        XCTAssertEqual(Feature.vocabulary.cValue.rawValue, 7)
        XCTAssertEqual(Feature.contextPrompt.cValue.rawValue, 8)
        XCTAssertEqual(Feature.instruct.cValue.rawValue, 9)
        XCTAssertEqual(Feature.transcriptPrefix.cValue.rawValue, 10)
    }

    func testRunParamsCarryPromptingFields() {
        let options = RunOptions(
            task: .instruct, vocabulary: ["GGUF", "ggml"], prompt: "Summarize.", prefix: "And so")
        options.withCParams { p in
            XCTAssertEqual(p.pointee.n_vocabulary, 2)
            XCTAssertEqual(String(cString: p.pointee.vocabulary![0]!), "GGUF")
            XCTAssertEqual(String(cString: p.pointee.vocabulary![1]!), "ggml")
            XCTAssertEqual(String(cString: p.pointee.prompt!), "Summarize.")
            XCTAssertEqual(String(cString: p.pointee.prefix!), "And so")
        }
        XCTAssertThrowsError(try RunOptions(prompt: "a\0b").checkCStrings())
        XCTAssertThrowsError(try RunOptions(vocabulary: ["a\0b"]).checkCStrings())
        RunOptions().withCParams { p in
            XCTAssertNil(p.pointee.vocabulary)
            XCTAssertEqual(p.pointee.n_vocabulary, 0)
            XCTAssertNil(p.pointee.prompt)
            XCTAssertNil(p.pointee.prefix)
        }
    }

    func testVocabularyAndPromptReachRunAndBatch() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let session = try Model(path: path).session()
        let options = RunOptions(vocabulary: ["Kennedy", "Americans", "GGUF"], prompt: "A speech.")
        XCTAssertTrue(try session.run(pcm, options: options).text.lowercased().contains("country"))
        let batch = try session.runBatch([pcm, pcm], options: options)
        XCTAssertEqual(batch.count, 2)
        for item in batch { XCTAssertNoThrow(try item.get()) }
        XCTAssertThrowsError(try session.run(pcm, options: RunOptions(prompt: "hi <|endoftext|>")))
        XCTAssertThrowsError(try session.runBatch([pcm], options: RunOptions(prefix: "And so")))
    }

    func testPrefixTextIsTheContinuation() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let model = try Model(path: path)
        guard model.supports(.transcriptPrefix) else { throw XCTSkip("no transcript prefix") }
        let prefix = "And so my fellow Americans"
        let t = try model.session().run(pcm, options: RunOptions(timestamps: .none, prefix: prefix))
        XCTAssertTrue(t.rawText.trimmingCharacters(in: .whitespaces).hasPrefix(prefix), t.rawText)
        XCTAssertFalse(t.text.contains("fellow Americans"), t.text)
        XCTAssertTrue(t.text.lowercased().contains("ask not"), t.text)
    }

    func testStreamAcceptsVocabularyRejectsInstruct() throws {
        guard let path = Fixtures.streamingModelPath() else { throw XCTSkip("no streaming canary") }
        let session = try Model(path: path).session()
        XCTAssertThrowsError(try session.stream(RunOptions(task: .instruct, prompt: "Summarize.")))
        XCTAssertNoThrow(try session.stream(RunOptions(vocabulary: ["Kennedy"], prompt: "A speech.")))
    }
}
