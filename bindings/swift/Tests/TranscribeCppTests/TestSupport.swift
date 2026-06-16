import Foundation
import XCTest

@testable import TranscribeCpp

/// Shared model-gated test fixtures. Resolves the canary model + audio from the
/// `TRANSCRIBE_SMOKE_*` env vars (the same names Rust uses) or the in-repo
/// defaults, and skips cleanly (XCTSkip) when neither is present — the model
/// tier of the two-tier scheme (requirements §4).
enum Fixtures {
    static func repoRoot() -> URL {
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<10 {
            if FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("include/transcribe.h").path) {
                return dir
            }
            dir = dir.deletingLastPathComponent()
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    private static func env(_ key: String) -> String? {
        let value = ProcessInfo.processInfo.environment[key]
        return (value?.isEmpty == false) ? value : nil
    }

    static func modelPath() -> String? {
        if let override = env("TRANSCRIBE_SMOKE_MODEL") { return override }
        let path = repoRoot()
            .appendingPathComponent("models/whisper-tiny.en/whisper-tiny.en-Q5_K_M.gguf").path
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    static func audioPath() -> String? {
        if let override = env("TRANSCRIBE_SMOKE_AUDIO") { return override }
        let path = repoRoot().appendingPathComponent("samples/jfk.wav").path
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    static func streamingModelPath() -> String? {
        if let override = env("TRANSCRIBE_SMOKE_STREAMING_MODEL") { return override }
        let path = repoRoot()
            .appendingPathComponent(
                "models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf").path
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    // Extra family-extension canaries — not in the CI fetch-canary set, so these
    // gate on their own env var / in-repo GGUF and XCTSkip when absent (they run
    // locally where the GGUFs exist; CI skips them).
    private static func familyModel(_ envKey: String, _ relativePath: String) -> String? {
        if let override = env(envKey) { return override }
        let path = repoRoot().appendingPathComponent(relativePath).path
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    /// Parakeet cache-aware streaming (accepts PARAKEET_STREAM).
    static func parakeetStreamModelPath() -> String? {
        familyModel(
            "TRANSCRIBE_SMOKE_PARAKEET_STREAM_MODEL",
            "models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf")
    }
    /// Parakeet chunked/buffered streaming (accepts PARAKEET_BUFFERED_STREAM).
    static func parakeetBufferedModelPath() -> String? {
        familyModel(
            "TRANSCRIBE_SMOKE_PARAKEET_BUFFERED_MODEL",
            "models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-Q8_0.gguf")
    }
    /// Voxtral realtime streaming (accepts VOXTRAL_REALTIME_STREAM).
    static func voxtralRealtimeModelPath() -> String? {
        familyModel(
            "TRANSCRIBE_SMOKE_VOXTRAL_MODEL",
            "models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf")
    }

    /// The model path + decoded PCM, or `XCTSkip` when either is absent.
    static func modelAndAudio() throws -> (model: String, pcm: [Float]) {
        guard let model = modelPath() else {
            throw XCTSkip("no canary model (set TRANSCRIBE_SMOKE_MODEL)")
        }
        guard let audio = audioPath() else {
            throw XCTSkip("no canary audio (set TRANSCRIBE_SMOKE_AUDIO)")
        }
        return (model, try loadWav(audio))
    }

    static func streamingModelAndAudio() throws -> (model: String, pcm: [Float]) {
        guard let model = streamingModelPath() else {
            throw XCTSkip("no streaming canary (set TRANSCRIBE_SMOKE_STREAMING_MODEL)")
        }
        guard let audio = audioPath() else {
            throw XCTSkip("no canary audio (set TRANSCRIBE_SMOKE_AUDIO)")
        }
        return (model, try loadWav(audio))
    }

    /// Feed `pcm` to an active stream in fixed chunks, then finalize.
    @discardableResult
    static func drive(
        _ stream: TranscribeCpp.Stream, pcm: [Float], chunk: Int = 1600
    ) throws -> StreamUpdate {
        var i = 0
        while i < pcm.count {
            let end = min(i + chunk, pcm.count)
            _ = try stream.feed(Array(pcm[i..<end]))
            i = end
        }
        return try stream.finalize()
    }

    /// Minimal 16-bit PCM WAV decoder → mono float32 in [-1, 1]. Assumes the
    /// canary is already 16 kHz mono (the library's only accepted format).
    static func loadWav(_ path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        func u32(_ offset: Int) -> Int {
            Int(data[offset]) | Int(data[offset + 1]) << 8
                | Int(data[offset + 2]) << 16 | Int(data[offset + 3]) << 24
        }
        // Walk chunks after the 12-byte "RIFF....WAVE" header to find "data".
        var offset = 12
        var dataStart = -1
        var dataLength = 0
        while offset + 8 <= data.count {
            let id = String(bytes: data[offset..<offset + 4], encoding: .ascii) ?? ""
            let size = u32(offset + 4)
            if id == "data" {
                dataStart = offset + 8
                dataLength = size
                break
            }
            offset += 8 + size + (size & 1)  // chunks are word-aligned
        }
        guard dataStart >= 0 else {
            throw NSError(domain: "TestSupport", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "no data chunk in \(path)"])
        }
        let end = min(dataStart + dataLength, data.count)
        var samples: [Float] = []
        samples.reserveCapacity((end - dataStart) / 2)
        var i = dataStart
        while i + 1 < end {
            let raw = Int16(bitPattern: UInt16(data[i]) | (UInt16(data[i + 1]) << 8))
            samples.append(Float(raw) / 32768.0)
            i += 2
        }
        return samples
    }
}
