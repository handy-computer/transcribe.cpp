import XCTest

@testable import TranscribeCpp

/// VAD role (`VadSession`, `VadIterator`): contract checks on the toy
/// silero_vad fixture; the iterator on synthetic probabilities (no model).
final class VadTests: XCTestCase {
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

    func testToyRolesInfo() throws {
        let model = try cpuModel(try Fixtures.vadToyModelPath())
        XCTAssertEqual(model.roles, .vad)
        XCTAssertEqual(try model.vadInfo, VadInfo(sampleRate: 16000, frameSamples: 512))
        XCTAssertThrowsError(try model.session()) { error in
            guard case TranscribeError.unsupportedRole = error else { return XCTFail("\(error)") }
        }
    }

    func testToyRunAndStreamParity() throws {
        let model = try cpuModel(try Fixtures.vadToyModelPath())
        let vad = try model.vadSession(threads: 1)
        let pcm = noise(40000 + 123)  // not a whole number of frames

        let r = try vad.run(pcm)
        XCTAssertEqual(r.probs.count, (pcm.count + 511) / 512)
        XCTAssertEqual(r.firstFrame, 0)
        XCTAssertTrue(r.probs.allSatisfy { $0 >= 0 && $0 <= 1 })
        for s in r.segments {
            XCTAssertLessThan(s.startSample, s.endSample)
            XCTAssertLessThanOrEqual(s.endSample, Int64(pcm.count))
        }
        XCTAssertGreaterThan(vad.timings.encodeMs, 0)

        // On CPU, streaming in any chunking equals the offline run, bit for bit.
        var streamed: [Float] = []
        var i = 0
        while i < pcm.count {
            let end = min(i + 1600, pcm.count)
            let f = try vad.streamFeed(Array(pcm[i..<end]))
            XCTAssertEqual(f.firstFrame, Int64(streamed.count))
            XCTAssertTrue(f.segments.isEmpty)
            streamed += f.probs
            i = end
        }
        streamed += try vad.streamFlush().probs
        XCTAssertEqual(streamed, r.probs)

        // Reset drops the stream: the next feed starts at frame 0.
        _ = try vad.streamFeed(Array(pcm[..<1000]))
        vad.streamReset()
        XCTAssertEqual(try vad.streamFeed(Array(pcm[..<1024])).firstFrame, 0)
    }

    func testToyInvalidInput() throws {
        let model = try cpuModel(try Fixtures.vadToyModelPath())
        let vad = try model.vadSession()
        for bad in [VadOptions(threshold: 1.5), VadOptions(threshold: 0.3, negThreshold: 0.4),
                    VadOptions(minSpeechMs: -1)] {
            XCTAssertThrowsError(try vad.run(noise(16000), options: bad)) { error in
                guard case TranscribeError.invalidArgument = error else { return XCTFail("\(error)") }
            }
        }
        XCTAssertThrowsError(try vad.run([])) { error in
            guard case TranscribeError.invalidArgument = error else { return XCTFail("\(error)") }
        }
        XCTAssertThrowsError(try vad.streamFeed([0, .nan])) { error in
            guard case TranscribeError.invalidArgument = error else { return XCTFail("\(error)") }
        }
    }

    func testToyCancelAborts() throws {
        let model = try cpuModel(try Fixtures.vadToyModelPath())
        let vad = try model.vadSession()
        let token = CancellationToken()
        vad.setCancellationToken(token)
        token.cancel()
        XCTAssertThrowsError(try vad.run(noise(16000))) { error in
            guard case TranscribeError.aborted = error else { return XCTFail("\(error)") }
        }
        vad.clearCancellationToken()
        XCTAssertNoThrow(try vad.run(noise(16000)))
    }

    func testIteratorEvents() throws {
        // 512-sample frames, 30 ms (480 samples) pad, 100 ms (1600) min silence.
        let it = try VadIterator(frameSamples: 512)
        XCTAssertEqual(try it.feed([0.9]), [VadEvent(kind: .start, sample: 0)])
        XCTAssertTrue(it.triggered)
        XCTAssertEqual(try it.feed([0, 0, 0, 0, 0]), [VadEvent(kind: .end, sample: 992)])
        XCTAssertFalse(it.triggered)
        XCTAssertEqual(it.currentSample, 6 * 512)

        // A bad probability fails without touching the state.
        XCTAssertThrowsError(try it.feed([0.9, 1.5])) { error in
            guard case TranscribeError.invalidArgument = error else { return XCTFail("\(error)") }
        }
        XCTAssertEqual(it.currentSample, 6 * 512)

        it.reset()
        XCTAssertEqual(it.currentSample, 0)

        // 0.3 is silence under the default negThreshold (0.35), not under 0.2.
        let probs: [Float] = [0.9] + Array(repeating: 0.3, count: 6)
        XCTAssertEqual(try VadIterator(frameSamples: 512).feed(probs).map(\.kind), [.start, .end])
        let low = try VadIterator(frameSamples: 512, options: VadIteratorOptions(negThreshold: 0.2))
        XCTAssertEqual(try low.feed(probs).map(\.kind), [.start])
        XCTAssertTrue(low.triggered)
        XCTAssertThrowsError(try VadIterator(frameSamples: 0)) { error in
            guard case TranscribeError.invalidArgument = error else { return XCTFail("\(error)") }
        }
    }
}
