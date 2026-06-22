import Foundation

/// Minimal 16-bit PCM WAV decoder → mono float32 in [-1, 1]. Self-contained so
/// the example needs no helper package (mirrors the canonical examples'
/// `ExampleSupport.loadWav`).
enum WavDecoder {
    static func load(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        func u32(_ offset: Int) -> Int {
            Int(data[offset]) | Int(data[offset + 1]) << 8
                | Int(data[offset + 2]) << 16 | Int(data[offset + 3]) << 24
        }
        var offset = 12  // skip RIFF/WAVE header
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
            throw NSError(domain: "WavDecoder", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "no data chunk in \(url.lastPathComponent)"])
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
