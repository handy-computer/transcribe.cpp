import CTranscribe
import Foundation

/// Static facts about a diarization model.
public struct DiarizeInfo: Sendable, Equatable {
    /// Input PCM rate (16000).
    public let sampleRate: Int32
    /// `SpeakerSegment.speakerId` is in `1...maxSpeakers`.
    public let maxSpeakers: Int32
}

public struct DiarizeOptions: Sendable {
    public var family: DiarizeExtension?
    public init(family: DiarizeExtension? = nil) { self.family = family }
}

extension Model {
    /// Throws `.unsupportedRole` when `roles` lacks `.diarize`.
    public var diarizeInfo: DiarizeInfo {
        get throws {
            var info = transcribe_diarize_info()
            transcribe_diarize_info_init(&info)
            try TranscribeError.check(transcribe_diarize_get_info(ptr, &info), context: "diarize_get_info")
            return DiarizeInfo(sampleRate: info.sample_rate, maxSpeakers: info.max_speakers)
        }
    }

    /// Create a DIARIZE session (`threads` 0 = library default). Throws
    /// `.unsupportedRole` when `roles` lacks `.diarize`.
    public func diarizeSession(threads: Int32 = 0) throws -> DiarizeSession {
        var params = transcribe_diarize_session_params()
        transcribe_diarize_session_params_init(&params)
        params.n_threads = threads
        var out: OpaquePointer?
        try TranscribeError.check(
            transcribe_diarize_session_init(ptr, &params, &out), context: "creating diarize session")
        guard let out else {
            throw TranscribeError.other(status: 0, message: "null diarize session handle")
        }
        return DiarizeSession(model: self, ptr: out)
    }
}

/// A DIARIZE-role session: who spoke when. Same threading and lifetime
/// contract as `Session` (single-threaded; holds its `Model` alive), and its
/// runs share the model's compute lock and stream lease with every `Session`.
public final class DiarizeSession {
    let model: Model
    let ptr: OpaquePointer
    /// Strong ref to the installed cancellation token (see `Session`).
    var cancelToken: CancellationToken?

    init(model: Model, ptr: OpaquePointer) {
        self.model = model
        self.ptr = ptr
    }

    deinit { transcribe_diarize_session_free(ptr) }

    /// Diarize one recording (16 kHz mono float32). Rows are grouped by
    /// speaker, time-ordered within a speaker; speakers may overlap.
    public func run(_ pcm: [Float], options: DiarizeOptions = .init()) throws -> [SpeakerSegment] {
        try model.withCompute(
            busyIfStreaming: "a stream is active on this model; finish or drop it before diarize run()"
        ) {
            let status = withDiarizeExtension(options.family) { ext in
                var params = transcribe_diarize_params()
                transcribe_diarize_params_init(&params)
                params.family = ext
                return pcm.withUnsafeBufferPointer {
                    transcribe_diarize_run(ptr, $0.baseAddress, Int32($0.count), &params)
                }
            }
            try TranscribeError.check(status, context: "diarize_run")
            return (0..<transcribe_diarize_n_segments(ptr)).map { i in
                var s = transcribe_speaker_segment(); transcribe_speaker_segment_init(&s)
                _ = transcribe_diarize_get_segment(ptr, i, &s)
                return SpeakerSegment(s)
            }
        }
    }

    /// `run` hopped off the caller's thread/actor onto a background queue, with
    /// Swift task cancellation bridged to the native abort. Same contract as
    /// `Session.run(_:options:) async`: a caller-installed token takes
    /// precedence, and the bridged token is removed when the call returns.
    public func run(_ pcm: [Float], options: DiarizeOptions = .init()) async throws -> [SpeakerSegment] {
        nonisolated(unsafe) let this = self
        let bridged = (cancelToken == nil) ? CancellationToken() : nil
        if let bridged { setCancellationToken(bridged) }
        defer { if bridged != nil { clearCancellationToken() } }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (cont: CheckedContinuation<[SpeakerSegment], Error>) in
                DispatchQueue.global().async {
                    cont.resume(with: Result { try this.run(pcm, options: options) })
                }
            }
        } onCancel: {
            bridged?.cancel()
        }
    }

    /// Timings from the most recent run (`decodeMs` is 0).
    public var timings: Timings {
        var t = transcribe_timings(); transcribe_timings_init(&t)
        _ = transcribe_diarize_get_timings(ptr, &t)
        return Timings(t)
    }
}
