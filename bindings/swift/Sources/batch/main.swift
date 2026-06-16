// batch — run several inputs in one dispatch, with per-result handling.
import ExampleSupport
import TranscribeCpp

guard let modelPath = ExampleSupport.modelPath(), let audioPath = ExampleSupport.audioPath() else {
    ExampleSupport.skip("set TRANSCRIBE_SMOKE_MODEL + TRANSCRIBE_SMOKE_AUDIO")
}

let pcm = try ExampleSupport.loadWav(audioPath)

// Scope the model/session so ARC frees the native Metal resources before the
// process exits — ggml-metal (macOS 15+) asserts every GPU resource is released
// before teardown, so a top-level `let` that outlives `main` aborts.
do {
    let session = try Model(path: modelPath).session()

    // Two utterances here; one bad input would surface as a `.failure` in its
    // slot without sinking the others.
    let results = try session.runBatch([pcm, pcm])
    for (i, result) in results.enumerated() {
        switch result {
        case .success(let transcript):
            print("[\(i)] \(transcript.text)")
        case .failure(let error):
            print("[\(i)] error: \(error)")
        }
    }
}
