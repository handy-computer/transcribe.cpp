import XCTest

@testable import TranscribeCpp

/// DIARIZE role (`DiarizeSession`).
final class DiarizeTests: XCTestCase {
    /// `p` is NaN on diarize rows, so compare the turns, not `SpeakerSegment`s.
    private func turns(_ rows: [SpeakerSegment]) -> [[Int64]] {
        rows.map { [$0.t0Ms, $0.t1Ms, Int64($0.speakerId)] }
    }

    private func assertUnsupportedRole(_ calls: [(String, () throws -> Void)]) {
        for (what, call) in calls {
            XCTAssertThrowsError(try call(), what) { error in
                guard case TranscribeError.unsupportedRole = error else {
                    return XCTFail("\(what): expected .unsupportedRole, got \(error)")
                }
            }
        }
    }

    func testAsrOnlyModelRefusesDiarize() throws {
        guard let path = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        let model = try Model(path: path)
        XCTAssertEqual(model.roles, .asr)
        XCTAssertFalse(model.accepts(DiarizeExtension.sortformer(SortformerDiarizeOptions())))
        assertUnsupportedRole([
            ("diarizeInfo", { _ = try model.diarizeInfo }),
            ("diarizeSession", { _ = try model.diarizeSession() }),
        ])
    }

    func testSortformerDiarizes() throws {
        let (path, pcm) = try Fixtures.sortformerModelAndAudio()
        // The golden turns are CPU numerics.
        let model = try Model(path: path, options: ModelOptions(backend: .cpu))
        XCTAssertEqual(model.roles, [.diarize])
        XCTAssertEqual(try model.diarizeInfo, DiarizeInfo(sampleRate: 16000, maxSpeakers: 4))
        XCTAssertTrue(model.accepts(DiarizeExtension.sortformer(SortformerDiarizeOptions())))
        assertUnsupportedRole([
            ("capabilities", { _ = try model.capabilities }),
            ("session", { _ = try model.session() }),
        ])

        // Segment values are pinned by the C test; here, check the rows are sane
        // and that the preset reaches the native run (.lowLatency moves turns).
        let session = try model.diarizeSession()
        let base = try session.run(pcm)
        XCTAssertFalse(base.isEmpty)
        for row in base {
            XCTAssertTrue((1...4).contains(row.speakerId), "\(row)")
            XCTAssertLessThan(row.t0Ms, row.t1Ms, "\(row)")
        }
        XCTAssertGreaterThan(session.timings.encodeMs, 0)
        let low = try session.run(
            pcm, options: DiarizeOptions(family: .sortformer(SortformerDiarizeOptions(preset: .lowLatency))))
        XCTAssertNotEqual(turns(low), turns(base), ".lowLatency must reach the native run")
    }

    func testCancelledRunAbortsAndRecovers() throws {
        let (path, pcm) = try Fixtures.sortformerModelAndAudio()
        let session = try Model(path: path).diarizeSession()
        let token = CancellationToken()
        token.cancel()
        session.setCancellationToken(token)
        XCTAssertThrowsError(try session.run(pcm)) { error in
            guard case TranscribeError.aborted = error else {
                return XCTFail("expected .aborted, got \(error)")
            }
        }
        session.clearCancellationToken()
        XCTAssertFalse(try session.run(pcm).isEmpty)
    }

    /// A diarize run shares the model's stream lease with ASR sessions.
    func testDiarizeRunIsBusyWhileAStreamIsActive() throws {
        let (path, pcm) = try Fixtures.sortformerModelAndAudio()
        let model = try Model(path: path)
        let session = try model.diarizeSession()
        // Sortformer cannot stream, so set the lease a stream would hold directly.
        model.withCompute { model.streamActive = true }
        defer { model.withCompute { model.streamActive = false } }
        XCTAssertThrowsError(try session.run(pcm)) { error in
            guard case TranscribeError.busy(let message) = error else {
                return XCTFail("expected .busy, got \(error)")
            }
            XCTAssertEqual(
                message, "a stream is active on this model; finish or drop it before diarize run()")
        }
    }
}
