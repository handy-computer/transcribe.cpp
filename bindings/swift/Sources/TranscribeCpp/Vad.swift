import CTranscribe
import Foundation

/// Static facts about a VAD model.
public struct VadInfo: Sendable, Equatable {
    /// Input PCM rate (16000).
    public let sampleRate: Int32
    /// Samples per probability (512 = 32 ms for Silero).
    public let frameSamples: Int32
}

/// Probabilities -> speech segments (`VadSession.run` only). The defaults are
/// those of `transcribe_vad_params_init` (Silero's `get_speech_timestamps`).
public struct VadOptions: Sendable {
    /// A frame with `p >= threshold` is speech. [0, 1].
    public var threshold: Double
    /// Inside speech, `p < negThreshold` is silence. Negative means
    /// `max(threshold - 0.15, 0.01)`; otherwise [0, threshold].
    public var negThreshold: Double
    /// Shorter segments are dropped.
    public var minSpeechMs: Int32
    /// Silence this long ends a segment.
    public var minSilenceMs: Int32
    /// Padding added to both ends of a segment.
    public var speechPadMs: Int32
    /// Longer segments are split at a pause; 0 = no limit.
    public var maxSpeechMs: Int32

    public init(
        threshold: Double = 0.5, negThreshold: Double = -1, minSpeechMs: Int32 = 250,
        minSilenceMs: Int32 = 100, speechPadMs: Int32 = 30, maxSpeechMs: Int32 = 0
    ) {
        self.threshold = threshold
        self.negThreshold = negThreshold
        self.minSpeechMs = minSpeechMs
        self.minSilenceMs = minSilenceMs
        self.speechPadMs = speechPadMs
        self.maxSpeechMs = maxSpeechMs
    }
}

/// One speech segment, in samples of the run's input: `[startSample, endSample)`.
public struct VadSegment: Sendable, Equatable {
    public let startSample: Int64
    public let endSample: Int64
}

/// The result of one `VadSession` run, feed or flush.
public struct VadResult: Sendable, Equatable {
    /// The call's per-frame speech probabilities.
    public let probs: [Float]
    /// Stream position of `probs[0]`, in frames: 0 after `run`; for a feed or
    /// flush, the frames the stream had scored before the call.
    public let firstFrame: Int64
    /// Speech segments, time-ordered; `run` only (empty after a feed or flush).
    public let segments: [VadSegment]
}

extension Model {
    /// Throws `.unsupportedRole` when `roles` lacks `.vad`.
    public var vadInfo: VadInfo {
        get throws {
            var info = transcribe_vad_info()
            transcribe_vad_info_init(&info)
            try TranscribeError.check(transcribe_vad_get_info(ptr, &info), context: "vad_get_info")
            return VadInfo(sampleRate: info.sample_rate, frameSamples: info.frame_samples)
        }
    }

    /// Create a VAD session (`threads` 0 = library default). Throws
    /// `.unsupportedRole` when `roles` lacks `.vad`.
    public func vadSession(threads: Int32 = 0) throws -> VadSession {
        var params = transcribe_vad_session_params()
        transcribe_vad_session_params_init(&params)
        params.n_threads = threads
        var out: OpaquePointer?
        try TranscribeError.check(
            transcribe_vad_session_init(ptr, &params, &out), context: "creating vad session")
        guard let out else {
            throw TranscribeError.other(status: 0, message: "null vad session handle")
        }
        return VadSession(model: self, ptr: out)
    }
}

/// A VAD-role session: where there is speech. Same threading and lifetime
/// contract as `Session` (single-threaded; holds its `Model` alive), and its
/// calls share the model's compute lock and stream lease with every `Session`.
public final class VadSession {
    let model: Model
    let ptr: OpaquePointer
    /// Strong ref to the installed cancellation token (see `Session`).
    var cancelToken: CancellationToken?

    init(model: Model, ptr: OpaquePointer) {
        self.model = model
        self.ptr = ptr
    }

    deinit { transcribe_vad_session_free(ptr) }

    /// Score and segment one clip (16 kHz mono float32) from a fresh model
    /// state. Resets any stream in progress.
    public func run(_ pcm: [Float], options: VadOptions = .init()) throws -> VadResult {
        try model.withCompute(
            busyIfStreaming: "a stream is active on this model; finish or drop it before vad run()"
        ) {
            var params = transcribe_vad_params()
            transcribe_vad_params_init(&params)
            params.threshold = options.threshold
            params.neg_threshold = options.negThreshold
            params.min_speech_ms = options.minSpeechMs
            params.min_silence_ms = options.minSilenceMs
            params.speech_pad_ms = options.speechPadMs
            params.max_speech_ms = options.maxSpeechMs
            let status = pcm.withUnsafeBufferPointer {
                transcribe_vad_run(ptr, $0.baseAddress, Int32($0.count), &params)
            }
            try TranscribeError.check(status, context: "vad_run")
            return try copyResult()
        }
    }

    /// `run` hopped off the caller's thread/actor onto a background queue, with
    /// Swift task cancellation bridged to the native abort. Same contract as
    /// `Session.run(_:options:) async`.
    public func run(_ pcm: [Float], options: VadOptions = .init()) async throws -> VadResult {
        nonisolated(unsafe) let this = self
        let bridged = (cancelToken == nil) ? CancellationToken() : nil
        if let bridged { setCancellationToken(bridged) }
        defer { if bridged != nil { clearCancellationToken() } }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (cont: CheckedContinuation<VadResult, Error>) in
                DispatchQueue.global().async {
                    cont.resume(with: Result { try this.run(pcm, options: options) })
                }
            }
        } onCancel: {
            bridged?.cancel()
        }
    }

    /// Append audio to the session's stream and score every frame it
    /// completes; a partial frame waits for the next call. A failure other
    /// than invalid input resets the stream.
    public func streamFeed(_ pcm: [Float]) throws -> VadResult {
        try model.withCompute(
            busyIfStreaming: "a stream is active on this model; finish or drop it before vad streamFeed()"
        ) {
            let status = pcm.withUnsafeBufferPointer {
                transcribe_vad_stream_feed(ptr, $0.baseAddress, Int32($0.count))
            }
            try TranscribeError.check(status, context: "vad_stream_feed")
            return try copyResult()
        }
    }

    /// End the stream: score the buffered partial frame, zero padded (one
    /// probability, or none), and reset. The next feed starts at frame 0.
    public func streamFlush() throws -> VadResult {
        try model.withCompute(
            busyIfStreaming: "a stream is active on this model; finish or drop it before vad streamFlush()"
        ) {
            try TranscribeError.check(transcribe_vad_stream_flush(ptr), context: "vad_stream_flush")
            return try copyResult()
        }
    }

    /// Drop the stream without scoring.
    public func streamReset() {
        model.withCompute { transcribe_vad_stream_reset(ptr) }
    }

    /// Timings from the most recent call (`encodeMs`: front end and encoder;
    /// `decodeMs`: the recurrent decoder).
    public var timings: Timings {
        var t = transcribe_timings(); transcribe_timings_init(&t)
        _ = transcribe_vad_get_timings(ptr, &t)
        return Timings(t)
    }

    /// Copy the last call's result out; call under the compute lock.
    private func copyResult() throws -> VadResult {
        var res = transcribe_vad_result(); transcribe_vad_result_init(&res)
        try TranscribeError.check(transcribe_vad_get_result(ptr, &res), context: "vad_get_result")
        var probs: [Float] = []
        if res.n_probs > 0, let p = transcribe_vad_probs(ptr) {
            probs = Array(UnsafeBufferPointer(start: p, count: Int(res.n_probs)))
        }
        let segments = (0..<res.n_segments).map { i -> VadSegment in
            var s = transcribe_vad_segment(); transcribe_vad_segment_init(&s)
            _ = transcribe_vad_get_segment(ptr, i, &s)
            return VadSegment(startSample: s.start_sample, endSample: s.end_sample)
        }
        return VadResult(probs: probs, firstFrame: res.first_frame, segments: segments)
    }
}

public struct VadIteratorOptions: Sendable {
    /// Speech starts at `p >= threshold`. [0, 1].
    public var threshold: Double
    /// Inside speech, `p < negThreshold` is silence. Negative means `threshold - 0.15`.
    public var negThreshold: Double
    /// Silence this long ends speech.
    public var minSilenceMs: Int32
    /// Padding added to event boundaries.
    public var speechPadMs: Int32
    public init(
        threshold: Double = 0.5, negThreshold: Double = -1, minSilenceMs: Int32 = 100, speechPadMs: Int32 = 30
    ) {
        self.threshold = threshold
        self.negThreshold = negThreshold
        self.minSilenceMs = minSilenceMs
        self.speechPadMs = speechPadMs
    }
}

/// A live START/END boundary, in 16 kHz samples (not the detection time).
public struct VadEvent: Sendable, Equatable {
    public enum Kind: Sendable { case start, end }
    public let kind: Kind
    public let sample: Int64
}

/// Silero's `VADIterator`: turns sequential per-frame probabilities (from
/// `VadSession.streamFeed`) into START/END events. Owns no model; use one
/// iterator from one thread at a time. Like upstream, end of input does not
/// emit END: close open speech yourself at the real audio length.
public final class VadIterator {
    let ptr: OpaquePointer

    /// `frameSamples` is normally `Model.vadInfo.frameSamples`.
    public init(frameSamples: Int32, options: VadIteratorOptions = .init()) throws {
        var params = transcribe_vad_iterator_params()
        transcribe_vad_iterator_params_init(&params)
        params.threshold = options.threshold
        params.neg_threshold = options.negThreshold
        params.min_silence_ms = options.minSilenceMs
        params.speech_pad_ms = options.speechPadMs
        var out: OpaquePointer?
        try TranscribeError.check(
            transcribe_vad_iterator_init(frameSamples, &params, &out), context: "creating vad iterator")
        guard let out else {
            throw TranscribeError.other(status: 0, message: "null vad iterator handle")
        }
        ptr = out
    }

    deinit { transcribe_vad_iterator_free(ptr) }

    /// Consume probabilities (each in [0, 1]) and return the events they
    /// produced, in detection order. A failure leaves the iterator unchanged.
    public func feed(_ probs: [Float]) throws -> [VadEvent] {
        let status = probs.withUnsafeBufferPointer {
            transcribe_vad_iterator_feed(ptr, $0.baseAddress, Int32($0.count))
        }
        try TranscribeError.check(status, context: "vad_iterator_feed")
        return (0..<state.n_events).map { i -> VadEvent in
            var e = transcribe_vad_event(); transcribe_vad_event_init(&e)
            _ = transcribe_vad_iterator_get_event(ptr, i, &e)
            return VadEvent(kind: e.type == TRANSCRIBE_VAD_EVENT_START ? .start : .end, sample: e.sample)
        }
    }

    /// Reset time and hysteresis; active speech is dropped without END.
    public func reset() { transcribe_vad_iterator_reset(ptr) }

    /// Probabilities consumed so far times `frameSamples`.
    public var currentSample: Int64 { state.current_sample }
    /// Speech is active (possibly awaiting silence).
    public var triggered: Bool { state.triggered }

    private var state: transcribe_vad_iterator_result {
        var r = transcribe_vad_iterator_result(); transcribe_vad_iterator_result_init(&r)
        _ = transcribe_vad_iterator_get_result(ptr, &r)
        return r
    }
}
