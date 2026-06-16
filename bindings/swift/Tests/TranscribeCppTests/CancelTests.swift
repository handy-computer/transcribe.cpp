import XCTest

@testable import TranscribeCpp

/// Cancellation (requirements: abort surfaced idiomatically). Mirrors Rust's
/// `cancel.rs`.
final class CancelTests: XCTestCase {
    func testUncancelledRunIsNotAborted() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let session = try Model(path: path).session()
        let token = CancellationToken()
        session.setCancellationToken(token)  // installed but never cancelled
        let transcript = try session.run(pcm)
        XCTAssertFalse(session.wasAborted)
        XCTAssertTrue(transcript.text.lowercased().contains("country"))
    }

    func testPreCancelledRunAborts() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        // A long clip so the decode polls the abort callback well before it
        // could finish; pre-cancelling makes the first poll abort.
        let long = Array(repeating: pcm, count: 4).flatMap { $0 }
        let session = try Model(path: path).session()
        let token = CancellationToken()
        token.cancel()
        session.setCancellationToken(token)
        XCTAssertThrowsError(try session.run(long)) { error in
            guard case TranscribeError.aborted = error else {
                return XCTFail("expected .aborted, got \(error)")
            }
        }
        XCTAssertTrue(session.wasAborted)
    }

    func testCrossThreadCancelOfInFlightRun() throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let long = Array(repeating: pcm, count: 12).flatMap { $0 }
        let session = try Model(path: path).session()
        let token = CancellationToken()
        session.setCancellationToken(token)
        // Fire the cancel shortly after the run begins. On a fast machine the
        // run may finish first — both outcomes are valid; we only require that
        // an abort, if it happens, is reported coherently.
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.03) { token.cancel() }
        do {
            _ = try session.run(long)
            XCTAssertFalse(session.wasAborted)
        } catch {
            guard case TranscribeError.aborted(_, let partial) = error else {
                return XCTFail("expected .aborted, got \(error)")
            }
            XCTAssertTrue(session.wasAborted)
            _ = partial  // partial transcript may be present
        }
    }
}
