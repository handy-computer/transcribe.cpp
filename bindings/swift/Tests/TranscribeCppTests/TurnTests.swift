import XCTest

@testable import TranscribeCpp

/// TURN role on the toy smart_turn (clip) and sava (streaming, with speech)
/// fixtures: info, run, stream frame bookkeeping, batch parity, errors, Busy.
final class TurnTests: XCTestCase {
    private static func noise(_ n: Int, seed: UInt32 = 7) -> [Float] {
        var s: UInt32 = seed
        return (0..<n).map { i in
            s = s &* 1664525 &+ 1013904223
            return (Float((s >> 8) & 0xFFFFFF) / 16777216.0 - 0.5) * 0.05
                + 0.3 * sin(0.05 * Float(i))
        }
    }

    private func smartTurn() throws -> Model {
        try Model(path: try Fixtures.toyModelPath("arch_smart_turn.gguf"), options: ModelOptions(backend: .cpu))
    }

    private func sava() throws -> Model {
        try Model(path: try Fixtures.toyModelPath("arch_sava.gguf"), options: ModelOptions(backend: .cpu))
    }

    private func expectError(
        _ call: @autoclosure () throws -> Any, file: StaticString = #filePath, line: UInt = #line,
        _ match: (TranscribeError) -> Bool
    ) {
        XCTAssertThrowsError(try call(), file: file, line: line) { error in
            guard let e = error as? TranscribeError, match(e) else {
                return XCTFail("unexpected error \(error)", file: file, line: line)
            }
        }
    }

    /// Feed `pcm` in `chunk`-sample pieces, then flush; checks `firstFrame` /
    /// `speechFirstFrame` against what was collected so far.
    private func stream(_ t: TurnSession, _ pcm: [Float], chunk: Int) throws -> (turn: [Float], speech: [Float]) {
        var turn: [Float] = []
        var speech: [Float] = []
        func collect(_ r: TurnResult) {
            XCTAssertEqual(r.firstFrame, Int64(turn.count))
            XCTAssertEqual(r.speechFirstFrame, Int64(speech.count))
            XCTAssertEqual(r.p, 0)
            XCTAssertFalse(r.ended)
            turn += r.probs
            speech += r.speechProbs
            XCTAssertLessThanOrEqual(speech.count, turn.count)
        }
        var i = 0
        while i < pcm.count {
            let end = min(i + chunk, pcm.count)
            collect(try t.streamFeed(Array(pcm[i..<end])))
            i = end
        }
        collect(try t.streamFlush())
        return (turn, speech)
    }

    func testSmartTurnInfoAndRun() throws {
        let model = try smartTurn()
        XCTAssertEqual(model.roles, .turn)
        XCTAssertEqual(
            try model.turnInfo,
            TurnInfo(sampleRate: 16000, maxAudioMs: 200, frameSamples: 0, hasSpeech: false, threshold: 0.5))
        let t = try model.turnSession(threads: 1)
        let pcm = Self.noise(3200)
        let r = try t.run(pcm)
        XCTAssertTrue(r.p > 0 && r.p < 1)
        XCTAssertEqual(r.ended, r.p >= 0.5)
        XCTAssertEqual(r.probs, [])
        XCTAssertEqual(r.speechProbs, [])
        XCTAssertEqual(r.firstFrame, 0)
        XCTAssertEqual(try t.run(pcm), r)  // deterministic

        // Threshold override: 0 always ends, 1 never (p < 1); negative = default.
        XCTAssertTrue(try t.run(pcm, threshold: 0).ended)
        XCTAssertFalse(try t.run(pcm, threshold: 1).ended)
        XCTAssertEqual(try t.run(pcm, threshold: -1), r)

        // A clip model does not stream.
        expectError(try t.streamFeed(Array(pcm[0..<640]))) {
            if case .notImplemented = $0 { return true } else { return false }
        }
        expectError(try t.streamFlush()) {
            if case .notImplemented = $0 { return true } else { return false }
        }
        expectError(try TurnSession.streamFeedBatch([(t, Array(pcm[0..<640]))])) {
            if case .notImplemented = $0 { return true } else { return false }
        }
        // Not a VAD model.
        expectError(try model.vadInfo) {
            if case .unsupportedRole = $0 { return true } else { return false }
        }
    }

    func testSavaRunAndStream() throws {
        let model = try sava()
        XCTAssertEqual(model.roles, [.vad, .turn])
        XCTAssertEqual(
            try model.turnInfo,
            TurnInfo(sampleRate: 16000, maxAudioMs: 0, frameSamples: 640, hasSpeech: true, threshold: 0.9))
        let t = try model.turnSession(threads: 1)
        let pcm = Self.noise(40 * 640 + 123)
        let nFrames = 41
        let r = try t.run(pcm)
        XCTAssertEqual(r.probs.count, nFrames)
        XCTAssertEqual(r.speechProbs.count, nFrames)
        XCTAssertEqual(r.firstFrame, 0)
        XCTAssertEqual(r.speechFirstFrame, 0)
        XCTAssertEqual(r.p, r.probs.last)
        XCTAssertEqual(r.ended, Double(r.p) >= 0.9)
        XCTAssertTrue(try t.run(pcm, threshold: 0).ended)
        XCTAssertFalse(try t.run(pcm, threshold: 1).ended)

        // Stream == run, bit for bit, for any chunking.
        for chunk in [320, 640, 1000, 2000] {
            let s = try stream(t, pcm, chunk: chunk)
            XCTAssertEqual(s.turn, r.probs, "chunk \(chunk)")
            XCTAssertEqual(s.speech, r.speechProbs, "chunk \(chunk)")
        }

        // An empty feed is fine; flush and reset both restart at frame 0.
        let chunk = Array(pcm[0..<1280])
        XCTAssertEqual(try t.streamFeed([]).probs, [])
        XCTAssertEqual(try t.streamFeed(chunk).firstFrame, 0)
        XCTAssertEqual(try t.streamFeed(chunk).firstFrame, 2)
        t.streamReset()
        XCTAssertEqual(try t.streamFeed(chunk).firstFrame, 0)
        _ = try t.streamFlush()
        XCTAssertEqual(try t.streamFeed(chunk).firstFrame, 0)
    }

    func testBatchEqualsSeparateFeeds() throws {
        let model = try sava()
        let pcms = (0..<3).map { Self.noise(10 * 640 + 77 * $0, seed: UInt32(3 + $0)) }
        let batched = try (0..<3).map { _ in try model.turnSession(threads: 1) }
        let alone = try (0..<3).map { _ in try model.turnSession(threads: 1) }
        let sizes = [0, 13, 640, 777, 1283, 2000]
        var off = [0, 0, 0]
        var k = 0
        while off.enumerated().contains(where: { $0.element < pcms[$0.offset].count }) {
            // Rotate the order so the first session (whose thread count is used) changes.
            let order = (0..<3).map { ($0 + k) % 3 }
            var inputs: [(TurnSession, [Float])] = []
            for i in order {
                let n = min(sizes[(i * 7 + k) % sizes.count], pcms[i].count - off[i])
                inputs.append((batched[i], Array(pcms[i][off[i]..<off[i] + n])))
            }
            let results = try TurnSession.streamFeedBatch(inputs)
            XCTAssertEqual(results.count, 3)
            for (j, i) in order.enumerated() {
                let n = inputs[j].1.count
                XCTAssertEqual(results[j], try alone[i].streamFeed(inputs[j].1))
                off[i] += n
            }
            k += 1
        }
        for i in 0..<3 {
            XCTAssertEqual(try batched[i].streamFlush(), try alone[i].streamFlush())
        }
    }

    func testInvalidInput() throws {
        let model = try sava()
        let t = try model.turnSession(threads: 1)
        let pcm = Self.noise(1280)
        func isInvalid(_ e: TranscribeError) -> Bool {
            if case .invalidArgument = e { return true } else { return false }
        }
        expectError(try t.run([]), isInvalid)
        expectError(try t.run([0, .nan, 0]), isInvalid)
        expectError(try t.run(pcm, threshold: 1.5), isInvalid)
        expectError(try t.streamFeed([0, .infinity]), isInvalid)

        // Invalid input leaves the stream where it was.
        XCTAssertEqual(try t.streamFeed(pcm).firstFrame, 0)
        expectError(try t.streamFeed([.nan]), isInvalid)
        XCTAssertEqual(try t.streamFeed(pcm).firstFrame, 2)

        let other = try model.turnSession(threads: 1)
        let otherModel = try sava()
        let foreign = try otherModel.turnSession(threads: 1)
        expectError(try TurnSession.streamFeedBatch([]), isInvalid)
        expectError(try TurnSession.streamFeedBatch([(t, pcm), (t, pcm)]), isInvalid)
        expectError(try TurnSession.streamFeedBatch([(t, pcm), (foreign, pcm)]), isInvalid)
        expectError(try TurnSession.streamFeedBatch([(t, pcm), (other, [.nan])]), isInvalid)
        // ...and every stream unchanged by a rejected batch.
        XCTAssertEqual(try t.streamFeed(pcm).firstFrame, 4)
        XCTAssertEqual(try other.streamFeed(pcm).firstFrame, 0)

        // A model without the TURN role.
        let vadOnly = try Model(path: try Fixtures.vadToyModelPath(), options: ModelOptions(backend: .cpu))
        expectError(try vadOnly.turnInfo) {
            if case .unsupportedRole = $0 { return true } else { return false }
        }
        expectError(try vadOnly.turnSession()) {
            if case .unsupportedRole = $0 { return true } else { return false }
        }
    }

    func testCancellation() throws {
        let model = try sava()
        let t = try model.turnSession(threads: 1)
        let pcm = Self.noise(1280)
        let token = CancellationToken()
        t.setCancellationToken(token)
        token.cancel()
        func isAborted(_ e: TranscribeError) -> Bool {
            if case .aborted = e { return true } else { return false }
        }
        expectError(try t.run(pcm), isAborted)
        expectError(try t.streamFeed(pcm), isAborted)
        expectError(try TurnSession.streamFeedBatch([(t, pcm)]), isAborted)
        t.clearCancellationToken()
        XCTAssertEqual(try t.streamFeed(pcm).firstFrame, 0)  // the abort reset the stream

        // Cancelling a later (not the first) listed session aborts the batch
        // and resets every listed stream.
        let u = try model.turnSession(threads: 1)
        XCTAssertEqual(try t.streamFeed(pcm).firstFrame, 2)
        XCTAssertEqual(try u.streamFeed(pcm).firstFrame, 0)
        let later = CancellationToken()
        u.setCancellationToken(later)
        later.cancel()
        expectError(try TurnSession.streamFeedBatch([(t, pcm), (u, pcm)]), isAborted)
        u.clearCancellationToken()
        XCTAssertEqual(try t.streamFeed(pcm).firstFrame, 0)
        XCTAssertEqual(try u.streamFeed(pcm).firstFrame, 0)
    }

    func testTurnIsBusyWhileAStreamIsActive() throws {
        let model = try sava()
        let t = try model.turnSession(threads: 1)
        let pcm = Self.noise(1280)
        // A turn model cannot open an ASR stream, so set the lease one would hold directly.
        model.withCompute { model.streamActive = true }
        defer { model.withCompute { model.streamActive = false } }
        let calls: [(String, () throws -> Any)] = [
            ("turn run()", { try t.run(pcm) }),
            ("turn streamFeed()", { try t.streamFeed(pcm) }),
            ("turn streamFeedBatch()", { try TurnSession.streamFeedBatch([(t, pcm)]) }),
            ("turn streamFlush()", { try t.streamFlush() }),
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

    // Real models (skip when absent).

    func testRealSmartTurn() throws {
        let model = try Model(path: try Fixtures.smartTurnModelPath())
        let info = try model.turnInfo
        XCTAssertEqual(info.frameSamples, 0)
        XCTAssertEqual(info.maxAudioMs, 8000)
        guard let audio = Fixtures.audioPath() else { throw XCTSkip("no canary audio") }
        let r = try model.turnSession().run(try Fixtures.loadWav(audio))
        XCTAssertTrue(r.p >= 0 && r.p <= 1)
        XCTAssertEqual(r.ended, Double(r.p) >= info.threshold)
    }

    func testRealSava() throws {
        let model = try Model(path: try Fixtures.savaModelPath(), options: ModelOptions(backend: .cpu))
        let info = try model.turnInfo
        XCTAssertEqual(info.frameSamples, 640)
        XCTAssertTrue(info.hasSpeech)
        guard let audio = Fixtures.audioPath() else { throw XCTSkip("no canary audio") }
        let pcm = Array(try Fixtures.loadWav(audio).prefix(16000 * 4))
        let t = try model.turnSession()
        let r = try t.run(pcm)
        let s = try stream(t, pcm, chunk: 320)
        XCTAssertEqual(s.turn, r.probs)
        XCTAssertEqual(s.speech, r.speechProbs)
    }
}
