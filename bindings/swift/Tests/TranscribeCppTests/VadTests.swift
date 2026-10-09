import XCTest

@testable import TranscribeCpp

/// VAD role on the toy silero_vad fixture: run, stream parity with run, and
/// the stream frame counter.
final class VadTests: XCTestCase {
    private static func noise(_ n: Int) -> [Float] {
        var s: UInt32 = 7
        return (0..<n).map { _ in
            s = s &* 1664525 &+ 1013904223
            return Float((s >> 8) & 0xFFFFFF) / 16777216.0 - 0.5
        }
    }

    func testToyRunAndStreamParity() throws {
        let model = try Model(path: try Fixtures.vadToyModelPath(), options: ModelOptions(backend: .cpu))
        XCTAssertEqual(model.roles, .vad)
        XCTAssertEqual(try model.vadInfo, VadInfo(sampleRate: 16000, frameSamples: 512))
        let pcm = Self.noise(16000 + 100)
        let vad = try model.vadSession(threads: 1)
        let r = try vad.run(pcm)
        XCTAssertEqual(r.probs.count, 32)

        var streamed: [Float] = []
        var frame: Int64 = 0
        var i = 0
        while i < pcm.count {
            let end = min(i + 1600, pcm.count)
            let fed = try vad.streamFeed(Array(pcm[i..<end]))
            XCTAssertEqual(fed.firstFrame, frame)
            frame += Int64(fed.probs.count)
            streamed += fed.probs
            i = end
        }
        let flushed = try vad.streamFlush()
        XCTAssertEqual(flushed.firstFrame, frame)
        streamed += flushed.probs
        XCTAssertEqual(streamed, r.probs)

        // Flush and reset both restart the stream at frame 0.
        let chunk = Array(pcm[0..<1600])
        XCTAssertEqual(try vad.streamFeed(chunk).firstFrame, 0)
        XCTAssertEqual(try vad.streamFeed(chunk).firstFrame, 3)
        vad.streamReset()
        XCTAssertEqual(try vad.streamFeed(chunk).firstFrame, 0)
    }

    func testVadIsBusyWhileAStreamIsActive() throws {
        let model = try Model(path: try Fixtures.vadToyModelPath(), options: ModelOptions(backend: .cpu))
        let vad = try model.vadSession(threads: 1)
        let pcm = Self.noise(1600)
        // VAD cannot open an ASR stream, so set the lease one would hold directly.
        model.withCompute { model.streamActive = true }
        defer { model.withCompute { model.streamActive = false } }
        let calls: [(String, () throws -> VadResult)] = [
            ("vad run()", { try vad.run(pcm) }),
            ("vad streamFeed()", { try vad.streamFeed(pcm) }),
            ("vad streamFlush()", { try vad.streamFlush() }),
        ]
        for (name, call) in calls {
            XCTAssertThrowsError(try call()) { error in
                guard case TranscribeError.busy(let message) = error else {
                    return XCTFail("expected .busy, got \(error)")
                }
                XCTAssertEqual(
                    message, "a stream is active on this model; finish or drop it before \(name)")
            }
        }
    }
}
