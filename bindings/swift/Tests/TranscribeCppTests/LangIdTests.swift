import XCTest

@testable import TranscribeCpp

/// LANGID role (`LangIdSession`): contract checks on the toy ecapa_tdnn
/// fixture, top-1 on the real VoxLingua107 model.
final class LangIdTests: XCTestCase {
    private func noise(_ n: Int, seed: UInt32 = 7) -> [Float] {
        var s = seed | 1
        return (0..<n).map { _ in
            s = s &* 1664525 &+ 1013904223
            return Float((s >> 8) & 0xFFFFFF) / 16777216.0 - 0.5
        }
    }

    private func cpuModel(_ path: String) throws -> Model {
        try Model(path: path, options: ModelOptions(backend: .cpu))
    }

    func testToyRolesInfoLabels() throws {
        let model = try cpuModel(try Fixtures.langIdToyModelPath())
        XCTAssertEqual(model.roles, .langId)
        XCTAssertEqual(try model.langIdInfo, LangIdInfo(sampleRate: 16000, nLabels: 5, minAudioMs: 500, maxAudioMs: 30000))
        let labels = try model.langIdLabels
        XCTAssertEqual(labels[2].code, "cc")
        XCTAssertEqual(labels[2].name, "Charlie")
        XCTAssertEqual(model.langIdLabelIndex("xx"), 0)
        XCTAssertNil(model.langIdLabelIndex("zz"))
        XCTAssertThrowsError(try model.session()) { error in
            guard case TranscribeError.unsupportedRole = error else { return XCTFail("\(error)") }
        }
    }

    func testToyRunContract() throws {
        let model = try cpuModel(try Fixtures.langIdToyModelPath())
        let lid = try model.langIdSession(threads: 1)
        let pcm = noise(16000)

        let r = try lid.run(pcm)
        XCTAssertEqual(r.candidates.count, 5)
        XCTAssertEqual(r.allowedMass, 1.0)
        XCTAssertEqual(r.code, r.candidates[0].code)
        XCTAssertEqual(r.candidates.map(\.p).reduce(0, +), 1.0, accuracy: 1e-5)

        let restricted = try lid.run(pcm, options: LangIdOptions(allowed: ["bb", "dd"]))
        XCTAssertEqual(restricted.candidates.count, 2)
        XCTAssertEqual(Set(restricted.candidates.map(\.code)), ["bb", "dd"])
        XCTAssertLessThan(restricted.allowedMass, 1.0)

        XCTAssertThrowsError(try lid.run(pcm, options: LangIdOptions(allowed: []))) { error in
            guard case TranscribeError.invalidArgument = error else { return XCTFail("\(error)") }
        }
        XCTAssertThrowsError(try lid.run(pcm, options: LangIdOptions(allowed: ["zz"]))) { error in
            guard case TranscribeError.unsupported = error else { return XCTFail("\(error)") }
        }
        XCTAssertThrowsError(try lid.run(Array(pcm[..<6400]))) { error in
            guard case TranscribeError.inputTooShort = error else { return XCTFail("\(error)") }
        }
        XCTAssertGreaterThan(lid.timings.encodeMs, 0)
    }

    func testToyInteriorNulIsRejected() throws {
        // C would cut "bb\0zz" to "bb"; the code must not silently narrow.
        let model = try cpuModel(try Fixtures.langIdToyModelPath())
        XCTAssertNil(model.langIdLabelIndex("bb\0zz"))
        let lid = try model.langIdSession()
        XCTAssertThrowsError(try lid.run(noise(16000), options: LangIdOptions(allowed: ["bb\0zz"]))) { error in
            guard case TranscribeError.invalidArgument = error else { return XCTFail("\(error)") }
        }
    }

    func testToyCancelAborts() throws {
        let model = try cpuModel(try Fixtures.langIdToyModelPath())
        let lid = try model.langIdSession()
        let token = CancellationToken()
        lid.setCancellationToken(token)
        token.cancel()
        XCTAssertThrowsError(try lid.run(noise(16000))) { error in
            guard case TranscribeError.aborted = error else { return XCTFail("\(error)") }
        }
        lid.clearCancellationToken()
        XCTAssertNoThrow(try lid.run(noise(16000)))
    }

    func testRealFleursTop1() throws {
        let model = try cpuModel(try Fixtures.langIdModelPath())
        XCTAssertEqual(try model.langIdInfo.nLabels, 107)
        XCTAssertEqual(try model.langIdInfo.maxAudioMs, 30000)
        XCTAssertEqual(model.langIdLabelIndex("he"), model.langIdLabelIndex("iw"))
        let lid = try model.langIdSession()
        for code in ["en", "de", "fr", "es", "ja", "zh", "ru", "id"] {
            let pcm = try Fixtures.loadWav(
                Fixtures.repoRoot().appendingPathComponent("samples/fleurs-\(code).wav").path)
            let r = try lid.run(pcm)
            XCTAssertEqual(r.candidates.count, 107)
            XCTAssertEqual(r.code, code)
            XCTAssertGreaterThanOrEqual(r.candidates[0].p, 0.5)
        }
    }
}
