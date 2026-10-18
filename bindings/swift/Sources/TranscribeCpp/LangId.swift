import CTranscribe
import Foundation

/// Static facts about a language ID model.
public struct LangIdInfo: Sendable, Equatable {
    /// Input PCM rate (16000).
    public let sampleRate: Int32
    /// Label indices are `0..<nLabels`.
    public let nLabels: Int32
    /// Shorter scored audio throws `.inputTooShort`.
    public let minAudioMs: Int32
    /// Longer input is scored on its first `maxAudioMs` (30000).
    public let maxAudioMs: Int32
}

public struct LangIdOptions: Sendable {
    /// Restrict the decision to these codes or aliases. `nil` means every
    /// label; an empty array throws `.invalidArgument`, and an unknown code
    /// throws `.unsupported`.
    public var allowed: [String]?
    public init(allowed: [String]? = nil) {
        self.allowed = allowed
    }
}

/// One ranked label. `code` is the model's own label ("en", "iw").
public struct LangIdCandidate: Sendable, Equatable {
    public let index: Int32
    public let code: String
    public let name: String
    /// Softmax renormalized over the allowed set.
    public let p: Float
    public let logit: Float
}

/// The result of one `LangIdSession.run`.
public struct LangIdResult: Sendable, Equatable {
    /// One per allowed label (every label when unrestricted), ranked by `p`,
    /// descending; ties keep label order.
    public let candidates: [LangIdCandidate]
    /// Unrestricted probability inside the allowed set (1.0 when
    /// unrestricted); a low value means the speech is probably outside it.
    public let allowedMass: Float
    /// The top candidate's code, if any.
    public var code: String? { candidates.first?.code }
}

extension Model {
    /// Throws `.unsupportedRole` when `roles` lacks `.langId`.
    public var langIdInfo: LangIdInfo {
        get throws {
            var info = transcribe_langid_info()
            transcribe_langid_info_init(&info)
            try TranscribeError.check(transcribe_langid_get_info(ptr, &info), context: "langid_get_info")
            return LangIdInfo(sampleRate: info.sample_rate, nLabels: info.n_labels,
                              minAudioMs: info.min_audio_ms, maxAudioMs: info.max_audio_ms)
        }
    }

    /// `(code, name)` per label index. Throws `.unsupportedRole` when `roles`
    /// lacks `.langId`.
    public var langIdLabels: [(code: String, name: String)] {
        get throws {
            let n = try langIdInfo.nLabels
            return (0..<n).map { i in
                (String(cString: transcribe_langid_label_code(ptr, i)),
                 String(cString: transcribe_langid_label_name(ptr, i)))
            }
        }
    }

    /// Label index of a code or alias ("he" and "iw" name the same label), or
    /// `nil` (also on a model without the LANGID role, or for a code with a
    /// NUL character, which C would silently cut).
    public func langIdLabelIndex(_ code: String) -> Int32? {
        if code.contains("\0") { return nil }
        let i = transcribe_langid_label_index(ptr, code)
        return i >= 0 ? i : nil
    }

    /// Create a LANGID session (`threads` 0 = library default). Throws
    /// `.unsupportedRole` when `roles` lacks `.langId`.
    public func langIdSession(threads: Int32 = 0) throws -> LangIdSession {
        var params = transcribe_langid_session_params()
        transcribe_langid_session_params_init(&params)
        params.n_threads = threads
        var out: OpaquePointer?
        try TranscribeError.check(
            transcribe_langid_session_init(ptr, &params, &out), context: "creating langid session")
        guard let out else {
            throw TranscribeError.other(status: 0, message: "null langid session handle")
        }
        return LangIdSession(model: self, ptr: out)
    }
}

/// A LANGID-role session: which language is spoken. Same threading and
/// lifetime contract as `Session` (single-threaded; holds its `Model` alive),
/// and its runs share the model's compute lock and stream lease with every
/// `Session`.
public final class LangIdSession {
    let model: Model
    let ptr: OpaquePointer
    /// Strong ref to the installed cancellation token (see `Session`).
    var cancelToken: CancellationToken?

    init(model: Model, ptr: OpaquePointer) {
        self.model = model
        self.ptr = ptr
    }

    deinit { transcribe_langid_session_free(ptr) }

    /// Identify the language of one clip (16 kHz mono float32). Input longer
    /// than `LangIdInfo.maxAudioMs` is scored on its first `maxAudioMs`.
    public func run(_ pcm: [Float], options: LangIdOptions = .init()) throws -> LangIdResult {
        if let allowed = options.allowed, allowed.isEmpty {
            // NULL would mean "every label", the opposite of an empty list.
            throw TranscribeError.invalidArgument("allowed is empty; pass nil for every label")
        }
        if let allowed = options.allowed, allowed.contains(where: { $0.contains("\0") }) {
            throw TranscribeError.invalidArgument("allowed contains a NUL character")
        }
        // strdup'd copies stay alive until the native call returns.
        let owned: [UnsafeMutablePointer<CChar>?] = options.allowed?.map { strdup($0) } ?? []
        defer { owned.forEach { free($0) } }
        let codes: [UnsafePointer<CChar>?] = owned.map { UnsafePointer($0) }

        return try model.withCompute(
            busyIfStreaming: "a stream is active on this model; finish or drop it before langid run()"
        ) {
            let status = codes.withUnsafeBufferPointer { codeBuf -> transcribe_status in
                var params = transcribe_langid_params()
                transcribe_langid_params_init(&params)
                if options.allowed != nil {
                    params.allowed = codeBuf.baseAddress
                    params.n_allowed = Int32(codeBuf.count)
                }
                return pcm.withUnsafeBufferPointer {
                    transcribe_langid_run(ptr, $0.baseAddress, Int32($0.count), &params)
                }
            }
            try TranscribeError.check(status, context: "langid_run")
            var res = transcribe_langid_result(); transcribe_langid_result_init(&res)
            try TranscribeError.check(transcribe_langid_get_result(ptr, &res), context: "langid_get_result")
            let candidates = (0..<res.n_candidates).map { i -> LangIdCandidate in
                var c = transcribe_langid_candidate(); transcribe_langid_candidate_init(&c)
                _ = transcribe_langid_get_candidate(ptr, i, &c)
                return LangIdCandidate(
                    index: c.index, code: c.code.map { String(cString: $0) } ?? "",
                    name: c.name.map { String(cString: $0) } ?? "",
                    p: c.p, logit: c.logit)
            }
            return LangIdResult(candidates: candidates, allowedMass: res.allowed_mass)
        }
    }

    /// `run` hopped off the caller's thread/actor onto a background queue, with
    /// Swift task cancellation bridged to the native abort. Same contract as
    /// `Session.run(_:options:) async`.
    public func run(_ pcm: [Float], options: LangIdOptions = .init()) async throws -> LangIdResult {
        nonisolated(unsafe) let this = self
        let bridged = (cancelToken == nil) ? CancellationToken() : nil
        if let bridged { setCancellationToken(bridged) }
        defer { if bridged != nil { clearCancellationToken() } }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (cont: CheckedContinuation<LangIdResult, Error>) in
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
        _ = transcribe_langid_get_timings(ptr, &t)
        return Timings(t)
    }
}
