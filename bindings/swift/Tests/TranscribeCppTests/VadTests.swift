import XCTest

@testable import TranscribeCpp

/// VAD role on the toy silero_vad fixture: run, and stream parity with run.
final class VadTests: XCTestCase {
    func testToyRunAndStreamParity() throws {
        let model = try Model(path: try Fixtures.vadToyModelPath(), options: ModelOptions(backend: .cpu))
        XCTAssertEqual(model.roles, .vad)
        XCTAssertEqual(try model.vadInfo, VadInfo(sampleRate: 16000, frameSamples: 512))
        var s: UInt32 = 7
        let pcm: [Float] = (0..<(16000 + 100)).map { _ in
            s = s &* 1664525 &+ 1013904223
            return Float((s >> 8) & 0xFFFFFF) / 16777216.0 - 0.5
        }
        let vad = try model.vadSession(threads: 1)
        let r = try vad.run(pcm)
        XCTAssertEqual(r.probs.count, 32)

        var streamed: [Float] = []
        var i = 0
        while i < pcm.count {
            let end = min(i + 1600, pcm.count)
            streamed += try vad.streamFeed(Array(pcm[i..<end])).probs
            i = end
        }
        streamed += try vad.streamFlush().probs
        XCTAssertEqual(streamed, r.probs)
    }
}
