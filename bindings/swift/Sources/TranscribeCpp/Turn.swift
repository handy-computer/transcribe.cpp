import CTranscribe
import Foundation

/// Static facts about a turn (end-of-turn detection) model.
public struct TurnInfo: Sendable, Equatable {
    /// Input PCM rate (16000).
    public let sampleRate: Int32
    /// The longest audio `run` looks at; longer audio is scored on its last
    /// `maxAudioMs`. 0 = no limit.
    public let maxAudioMs: Int32
    /// Samples per stream frame; 0 = the model cannot stream (use `run`).
    public let frameSamples: Int32
    /// The model also returns a per-frame speech probability (`speechProbs`).
    public let hasSpeech: Bool
    /// The model's default cut-off: `ended` is `p >= threshold`.
    public let threshold: Double
}

/// The result of one `TurnSession` run, feed or flush.
public struct TurnResult: Sendable, Equatable {
    /// After `run`: the probability that the turn has ended at the end of the
    /// clip. 0 after a feed or flush.
    public let p: Float
    /// After `run`: `p >= threshold`. False after a feed or flush.
    public let ended: Bool
    /// The call's per-frame turn probabilities: `probs[i]` says whether the
    /// turn has ended at the end of frame `firstFrame + i`. Empty for models
    /// that cannot stream.
    public let probs: [Float]
    /// Stream position of `probs[0]`, in frames: 0 after `run`; for a feed or
    /// flush, the frames the stream had processed before the call.
    public let firstFrame: Int64
    /// The call's per-frame speech probabilities (models with `hasSpeech`).
    /// During a stream these may trail `probs`.
    public let speechProbs: [Float]
    /// Stream position of `speechProbs[0]`, in frames.
    public let speechFirstFrame: Int64
}

extension Model {
    /// Throws `.unsupportedRole` when `roles` lacks `.turn`.
    public var turnInfo: TurnInfo {
        get throws {
            var info = transcribe_turn_info()
            transcribe_turn_info_init(&info)
            try TranscribeError.check(transcribe_turn_get_info(ptr, &info), context: "turn_get_info")
            return TurnInfo(
                sampleRate: info.sample_rate, maxAudioMs: info.max_audio_ms,
                frameSamples: info.frame_samples, hasSpeech: info.has_speech,
                threshold: info.threshold)
        }
    }

    /// Create a turn session (`threads` 0 = library default). Throws
    /// `.unsupportedRole` when `roles` lacks `.turn`.
    public func turnSession(threads: Int32 = 0) throws -> TurnSession {
        var params = transcribe_turn_session_params()
        transcribe_turn_session_params_init(&params)
        params.n_threads = threads
        var out: OpaquePointer?
        try TranscribeError.check(
            transcribe_turn_session_init(ptr, &params, &out), context: "creating turn session")
        guard let out else {
            throw TranscribeError.other(status: 0, message: "null turn session handle")
        }
        return TurnSession(model: self, ptr: out)
    }
}

/// A TURN-role session: has the speaker finished talking? Same threading and
/// lifetime contract as `Session` (single-threaded; holds its `Model` alive).
/// Calls take the model's compute lock; `run`/`streamFeed`/`streamFlush` and
/// `streamFeedBatch` throw `.busy` while an ASR stream holds the lease. A turn
/// stream is session state and does not take the lease.
public final class TurnSession {
    let model: Model
    let ptr: OpaquePointer
    /// Strong ref to the installed cancellation token (see `Session`).
    var cancelToken: CancellationToken?

    init(model: Model, ptr: OpaquePointer) {
        self.model = model
        self.ptr = ptr
    }

    deinit { transcribe_turn_session_free(ptr) }

    /// Score one clip (16 kHz mono float32): has the turn ended at its end?
    /// Pass the audio of the current turn. `threshold` nil (or negative) uses
    /// the model's `TurnInfo.threshold`; otherwise [0, 1]. Resets any stream
    /// in progress. Streaming models also return every frame's probabilities.
    public func run(_ pcm: [Float], threshold: Double? = nil) throws -> TurnResult {
        try model.withCompute(
            busyIfStreaming: "a stream is active on this model; finish or drop it before turn run()"
        ) {
            var params = transcribe_turn_params()
            transcribe_turn_params_init(&params)
            if let threshold { params.threshold = threshold }
            let status = pcm.withUnsafeBufferPointer {
                transcribe_turn_run(ptr, $0.baseAddress, Int32($0.count), &params)
            }
            try TranscribeError.check(status, context: "turn_run")
            return try copyResult()
        }
    }

    /// `run` hopped off the caller's thread/actor onto a background queue, with
    /// Swift task cancellation bridged to the native abort. Same contract as
    /// `Session.run(_:options:) async`.
    public func run(_ pcm: [Float], threshold: Double? = nil) async throws -> TurnResult {
        nonisolated(unsafe) let this = self
        let bridged = (cancelToken == nil) ? CancellationToken() : nil
        if let bridged { setCancellationToken(bridged) }
        defer { if bridged != nil { clearCancellationToken() } }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (cont: CheckedContinuation<TurnResult, Error>) in
                DispatchQueue.global().async {
                    cont.resume(with: Result { try this.run(pcm, threshold: threshold) })
                }
            }
        } onCancel: {
            bridged?.cancel()
        }
    }

    /// Append audio to the session's stream and score every frame it
    /// completes; a partial frame waits for the next call. Throws
    /// `.notImplemented` when the model cannot stream (`frameSamples == 0`).
    /// Invalid input leaves the stream and last result intact; an abort or
    /// processing error resets the stream, so the next feed starts at frame 0.
    public func streamFeed(_ pcm: [Float]) throws -> TurnResult {
        try model.withCompute(
            busyIfStreaming: "a stream is active on this model; finish or drop it before turn streamFeed()"
        ) {
            let status = pcm.withUnsafeBufferPointer {
                transcribe_turn_stream_feed(ptr, $0.baseAddress, Int32($0.count))
            }
            try TranscribeError.check(status, context: "turn_stream_feed")
            return try copyResult()
        }
    }

    /// Feed several streams of one model in a single compute: `inputs[i].1`
    /// goes to `inputs[i].0`, and each session ends exactly as after its own
    /// `streamFeed`. Returns one result per input, in order. Every session must
    /// belong to the same model and appear once; an empty list throws
    /// `.invalidArgument`. Uses the first session's thread count and checks
    /// every session's cancellation token. An abort or processing error resets
    /// every listed stream.
    public static func streamFeedBatch(_ inputs: [(TurnSession, [Float])]) throws -> [TurnResult] {
        guard let model = inputs.first?.0.model else {
            throw TranscribeError.invalidArgument("turn streamFeedBatch: no sessions")
        }
        var seen = Set<ObjectIdentifier>()
        for (session, _) in inputs {
            guard session.model === model else {
                throw TranscribeError.invalidArgument(
                    "turn streamFeedBatch: every session must belong to the same model")
            }
            guard seen.insert(ObjectIdentifier(session)).inserted else {
                throw TranscribeError.invalidArgument(
                    "turn streamFeedBatch: a session appears more than once")
            }
        }
        // One flat copy of the audio keeps every buffer alive and in place for
        // the whole native call.
        var flat: [Float] = []
        flat.reserveCapacity(inputs.reduce(0) { $0 + $1.1.count })
        var offsets: [Int] = []
        var counts: [Int32] = []
        for (_, pcm) in inputs {
            offsets.append(flat.count)
            counts.append(Int32(pcm.count))
            flat.append(contentsOf: pcm)
        }
        let sessions: [OpaquePointer?] = inputs.map { $0.0.ptr }
        return try model.withCompute(
            busyIfStreaming:
                "a stream is active on this model; finish or drop it before turn streamFeedBatch()"
        ) {
            let status = flat.withUnsafeBufferPointer { buf -> transcribe_status in
                let pcmPtrs: [UnsafePointer<Float>?] = zip(offsets, counts).map { off, n in
                    n > 0 ? buf.baseAddress.map { $0 + off } : nil
                }
                return sessions.withUnsafeBufferPointer { s in
                    pcmPtrs.withUnsafeBufferPointer { p in
                        counts.withUnsafeBufferPointer { c in
                            transcribe_turn_stream_feed_batch(
                                s.baseAddress, p.baseAddress, c.baseAddress, Int32(inputs.count))
                        }
                    }
                }
            }
            try TranscribeError.check(status, context: "turn_stream_feed_batch")
            return try inputs.map { try $0.0.copyResult() }
        }
    }

    /// End the stream: score the buffered partial frame, zero padded (one turn
    /// probability, or none) plus every frame's speech probability not yet
    /// returned, and reset. The next feed starts at frame 0.
    public func streamFlush() throws -> TurnResult {
        try model.withCompute(
            busyIfStreaming: "a stream is active on this model; finish or drop it before turn streamFlush()"
        ) {
            try TranscribeError.check(transcribe_turn_stream_flush(ptr), context: "turn_stream_flush")
            return try copyResult()
        }
    }

    /// Drop the stream without scoring; the next feed starts at frame 0. The
    /// last result is kept.
    public func streamReset() {
        model.withCompute { transcribe_turn_stream_reset(ptr) }
    }

    /// Timings from the most recent call.
    public var timings: Timings {
        var t = transcribe_timings(); transcribe_timings_init(&t)
        _ = transcribe_turn_get_timings(ptr, &t)
        return Timings(t)
    }

    /// Copy the last call's result out; call under the compute lock.
    private func copyResult() throws -> TurnResult {
        var res = transcribe_turn_result(); transcribe_turn_result_init(&res)
        try TranscribeError.check(transcribe_turn_get_result(ptr, &res), context: "turn_get_result")
        var probs: [Float] = []
        if res.n_probs > 0, let p = transcribe_turn_probs(ptr) {
            probs = Array(UnsafeBufferPointer(start: p, count: Int(res.n_probs)))
        }
        var speech: [Float] = []
        if res.n_speech_probs > 0, let p = transcribe_turn_speech_probs(ptr) {
            speech = Array(UnsafeBufferPointer(start: p, count: Int(res.n_speech_probs)))
        }
        return TurnResult(
            p: res.p, ended: res.ended, probs: probs, firstFrame: res.first_frame,
            speechProbs: speech, speechFirstFrame: res.speech_first_frame)
    }
}
