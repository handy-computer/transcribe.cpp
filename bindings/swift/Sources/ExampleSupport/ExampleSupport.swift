import Foundation

/// Shared plumbing for the canonical examples (the analog of Rust's
/// `examples/common/mod.rs`). Resolves the canary model/audio from the
/// `TRANSCRIBE_SMOKE_*` env vars or the in-repo defaults, decodes WAV, and
/// provides a clean headless skip so every example runs in CI under the same
/// rules as the model-gated tests (requirements §6).
public enum ExampleSupport {
    public static func env(_ key: String) -> String? {
        let value = ProcessInfo.processInfo.environment[key]
        return (value?.isEmpty == false) ? value : nil
    }

    public static func repoRoot() -> URL {
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

    private static func resolve(_ envKey: String, default relativePath: String) -> String? {
        if let override = env(envKey) { return override }
        let path = repoRoot().appendingPathComponent(relativePath).path
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    public static func modelPath() -> String? {
        resolve("TRANSCRIBE_SMOKE_MODEL", default: "models/whisper-tiny.en/whisper-tiny.en-Q5_K_M.gguf")
    }

    public static func streamingModelPath() -> String? {
        resolve(
            "TRANSCRIBE_SMOKE_STREAMING_MODEL",
            default: "models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf")
    }

    public static func audioPath() -> String? {
        resolve("TRANSCRIBE_SMOKE_AUDIO", default: "samples/jfk.wav")
    }

    /// Print a skip note and exit 0 — an example that can't find its fixtures is
    /// a clean no-op in CI, never a failure.
    public static func skip(_ reason: String) -> Never {
        print("skip: \(reason)")
        exit(0)
    }

    /// Minimal 16-bit PCM WAV decoder → mono float32 in [-1, 1].
    public static func loadWav(_ path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        func u32(_ offset: Int) -> Int {
            Int(data[offset]) | Int(data[offset + 1]) << 8
                | Int(data[offset + 2]) << 16 | Int(data[offset + 3]) << 24
        }
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
            offset += 8 + size + (size & 1)
        }
        guard dataStart >= 0 else {
            throw NSError(domain: "ExampleSupport", code: 1,
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
