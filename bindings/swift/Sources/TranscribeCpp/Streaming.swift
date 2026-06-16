import CTranscribe

public enum StreamState: Sendable {
    case idle, active, finished, failed
    init(_ c: transcribe_stream_state) {
        switch c {
        case TRANSCRIBE_STREAM_ACTIVE: self = .active
        case TRANSCRIBE_STREAM_FINISHED: self = .finished
        case TRANSCRIBE_STREAM_FAILED: self = .failed
        default: self = .idle
        }
    }
}

public enum CommitPolicy: Sendable {
    case auto, onFinalize, stablePrefix
    var cValue: transcribe_stream_commit_policy {
        switch self {
        case .auto: return TRANSCRIBE_STREAM_COMMIT_AUTO
        case .onFinalize: return TRANSCRIBE_STREAM_COMMIT_ON_FINALIZE
        case .stablePrefix: return TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX
        }
    }
}

public struct StreamOptions: Sendable {
    public var commitPolicy: CommitPolicy
    /// Consecutive agreeing hypotheses before a prefix commits; 0 = default (3).
    public var stablePrefixAgreementN: UInt32
    public var family: StreamExtension?

    public init(
        commitPolicy: CommitPolicy = .auto,
        stablePrefixAgreementN: UInt32 = 0,
        family: StreamExtension? = nil
    ) {
        self.commitPolicy = commitPolicy
        self.stablePrefixAgreementN = stablePrefixAgreementN
        self.family = family
    }

    func withCParams<R>(_ body: (UnsafePointer<transcribe_stream_params>) throws -> R) rethrows -> R {
        var params = transcribe_stream_params()
        transcribe_stream_params_init(&params)
        params.commit_policy = commitPolicy.cValue
        params.stable_prefix_agreement_n = stablePrefixAgreementN
        return try withStreamExtension(family) { ext in
            params.family = ext
            return try withUnsafePointer(to: &params) { try body($0) }
        }
    }
}

/// UI-stable streaming text. `committed` is append-only and flicker-free;
/// `tentative` is the volatile suffix; `full` is the authoritative raw
/// hypothesis. All copied at the FFI boundary.
public struct StreamText: Sendable, Equatable {
    public let full: String
    public let committed: String
    public let tentative: String
    /// Flicker-free display string: `committed + tentative`.
    public var display: String { committed + tentative }

    init(_ c: transcribe_stream_text) {
        full = c.full_text.map { String(cString: $0) } ?? ""
        committed = c.committed_text.map { String(cString: $0) } ?? ""
        tentative = c.tentative_text.map { String(cString: $0) } ?? ""
    }
}

/// Per-call change metadata from `feed` / `finalize`.
public struct StreamUpdate: Sendable, Equatable {
    public let resultChanged: Bool
    public let isFinal: Bool
    public let revision: Int32
    public let inputReceivedMs: Int64
    public let audioCommittedMs: Int64
    public let bufferedMs: Int64
    public let committedChanged: Bool
    public let tentativeChanged: Bool

    init(_ c: transcribe_stream_update) {
        resultChanged = c.result_changed
        isFinal = c.is_final
        revision = c.revision
        inputReceivedMs = c.input_received_ms
        audioCommittedMs = c.audio_committed_ms
        bufferedMs = c.buffered_ms
        committedChanged = c.committed_changed
        tentativeChanged = c.tentative_changed
    }
}

/// An active streaming run on a `Session`. Streaming is a mode on the session,
/// so a `Stream` holds its session (keeping it alive) and drives its state.
/// Like the session, it is single-threaded; `feed`/`finalize` serialize on the
/// model lock (the compute path).
public final class Stream {
    private let session: Session

    init(_ session: Session) { self.session = session }

    /// Feed a PCM frame (16 kHz mono float32). Returns per-call change metadata.
    public func feed(_ frame: [Float]) throws -> StreamUpdate {
        session.model.runLock.lock()
        defer { session.model.runLock.unlock() }
        var update = transcribe_stream_update()
        transcribe_stream_update_init(&update)
        let status = frame.withUnsafeBufferPointer {
            transcribe_stream_feed(session.ptr, $0.baseAddress, Int32($0.count), &update)
        }
        try TranscribeError.check(status, context: "stream_feed")
        return StreamUpdate(update)
    }

    /// Signal end of input; flushes buffered audio and emits remaining text.
    @discardableResult
    public func finalize() throws -> StreamUpdate {
        session.model.runLock.lock()
        defer { session.model.runLock.unlock() }
        var update = transcribe_stream_update()
        transcribe_stream_update_init(&update)
        let status = transcribe_stream_finalize(session.ptr, &update)
        try TranscribeError.check(status, context: "stream_finalize")
        return StreamUpdate(update)
    }

    /// Abandon the stream and return the session to idle.
    @discardableResult
    public func reset() -> StreamState {
        transcribe_stream_reset(session.ptr)
        return state
    }

    public var state: StreamState { StreamState(transcribe_stream_get_state(session.ptr)) }
    public var revision: Int32 { transcribe_stream_revision(session.ptr) }
    public var lastStatus: Int32 {
        Int32(bitPattern: transcribe_stream_last_status(session.ptr).rawValue)
    }

    /// The UI-stable text snapshot (committed / tentative / full).
    public var text: StreamText {
        var t = transcribe_stream_text()
        transcribe_stream_text_init(&t)
        _ = transcribe_stream_get_text(session.ptr, &t)
        return StreamText(t)
    }
}

extension Session {
    /// Begin a streaming run. The session must support streaming (else
    /// `.notImplemented`) and be idle/finished/failed (not already streaming).
    public func stream(
        _ runOptions: RunOptions = .init(), _ streamOptions: StreamOptions = .init()
    ) throws -> Stream {
        model.runLock.lock()
        defer { model.runLock.unlock() }
        let status = runOptions.withCParams { runParams in
            streamOptions.withCParams { streamParams in
                transcribe_stream_begin(ptr, runParams, streamParams)
            }
        }
        try TranscribeError.check(status, context: "stream_begin")
        return Stream(self)
    }
}
