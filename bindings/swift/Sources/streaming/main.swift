// streaming — feed chunks, watch committed (stable) vs tentative (volatile) text.
import ExampleSupport
import TranscribeCpp

guard let modelPath = ExampleSupport.streamingModelPath(), let audioPath = ExampleSupport.audioPath()
else {
    ExampleSupport.skip("set TRANSCRIBE_SMOKE_STREAMING_MODEL + TRANSCRIBE_SMOKE_AUDIO")
}

let pcm = try ExampleSupport.loadWav(audioPath)

// Scope the model/session/stream so ARC frees the native Metal resources before
// the process exits — ggml-metal (macOS 15+) asserts every GPU resource is
// released before teardown, so a top-level `let` that outlives `main` aborts.
do {
    let session = try Model(path: modelPath).session()
    let stream = try session.stream()

    let chunk = 1600  // 100 ms at 16 kHz
    var offset = 0
    while offset < pcm.count {
        let end = min(offset + chunk, pcm.count)
        let update = try stream.feed(Array(pcm[offset..<end]))
        if update.committedChanged || update.tentativeChanged {
            let text = stream.text
            print("committed: \(text.committed)")
            print("  tentative: \(text.tentative)")
        }
        offset = end
    }

    try stream.finalize()
    print("\nfinal: \(stream.text.full)")
}
