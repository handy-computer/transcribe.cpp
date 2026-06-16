// transcribe-file — load a model, transcribe a WAV, print text + segments.
import ExampleSupport
import TranscribeCpp

guard let modelPath = ExampleSupport.modelPath(), let audioPath = ExampleSupport.audioPath() else {
    ExampleSupport.skip("set TRANSCRIBE_SMOKE_MODEL + TRANSCRIBE_SMOKE_AUDIO (or provide the in-repo canary)")
}

let pcm = try ExampleSupport.loadWav(audioPath)

// Scope the model/session so ARC frees the native Metal resources before the
// process exits. ggml-metal (macOS 15+ residency sets) asserts every GPU
// resource is released before its teardown, so a model held in a top-level
// `let` — which outlives `main` — aborts at exit.
do {
    let model = try Model(path: modelPath)
    print("loaded \(model.arch)/\(model.variant) on \(model.backend)")

    let session = try model.session()
    let transcript = try session.run(pcm, options: RunOptions(timestamps: .segment))

    print("\ntext: \(transcript.text)")
    if let language = transcript.language { print("language: \(language)") }
    print("segments (\(transcript.segments.count)):")
    for segment in transcript.segments {
        let t0 = Double(segment.t0Ms) / 1000, t1 = Double(segment.t1Ms) / 1000
        print(String(format: "  [%6.2f – %6.2f]  %@", t0, t1, segment.text))
    }
}
