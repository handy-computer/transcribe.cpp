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

    // MARK: - Swift task cancellation bridge (async)

    /// The `async` run bridges Swift structured concurrency: cancelling the
    /// surrounding `Task` must abort the in-flight run (throwing `.aborted`)
    /// when the caller has NOT installed their own token. A long clip ensures
    /// the native abort poll trips well before the run could finish.
    func testAsyncRunBridgesTaskCancellation() async throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let long = Array(repeating: pcm, count: 6).flatMap { $0 }
        let session = try Model(path: path).session()  // no caller token -> bridge active
        // Bare `Task` must resolve to Swift's concurrency type: the run-mode enum
        // is `TranscriptionTask`, so it no longer shadows `_Concurrency.Task`.
        let task = Task { try await session.run(long) }
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("a cancelled Task must abort the bridged async run")
        } catch let error as TranscribeError {
            guard case .aborted = error else {
                return XCTFail("expected .aborted from a cancelled task, got \(error)")
            }
        }
        XCTAssertTrue(session.wasAborted)
    }

    /// The bridge must never clobber a caller-installed token: there is a single
    /// native abort slot, so a token the caller installed takes precedence and
    /// survives an async run unchanged. (Guards the "only when no caller token"
    /// rule — passes pre- and post-bridge; fails if the bridge installs blindly.)
    func testAsyncRunPreservesCallerInstalledToken() async throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let session = try Model(path: path).session()
        let myToken = CancellationToken()
        session.setCancellationToken(myToken)
        _ = try await session.run(pcm)
        XCTAssertTrue(session.cancelToken === myToken, "bridge must not replace the caller's token")
    }

    /// `runBatch` shares the same bridge as `run`. A whole-batch abort surfaces
    /// as per-slot `.aborted` failures (the binding does not throw on batch-level
    /// ABORTED, per include/transcribe.h) and `wasAborted` records it.
    func testAsyncRunBatchBridgesTaskCancellation() async throws {
        let (path, pcm) = try Fixtures.modelAndAudio()
        let long = Array(repeating: pcm, count: 6).flatMap { $0 }
        let session = try Model(path: path).session()
        let task = Task { try await session.runBatch([long]) }
        task.cancel()
        let results = (try? await task.value) ?? []
        XCTAssertTrue(session.wasAborted, "Task.cancel must abort the bridged async runBatch")
        if let first = results.first, case .failure(let err) = first {
            guard case TranscribeError.aborted = err else {
                return XCTFail("expected a per-slot .aborted, got \(err)")
            }
        }
    }
}
