// error-handling — idiomatic error mapping, cancellation, cleanup.
import ExampleSupport
import TranscribeCpp

// 1. A missing file maps to a distinct, matchable case.
do {
    _ = try Model(path: "/no/such/model.gguf")
} catch let error as TranscribeError {
    if case .modelFileNotFound(let message) = error {
        print("missing file -> .modelFileNotFound: \(message)")
    }
}

guard let modelPath = ExampleSupport.modelPath() else {
    ExampleSupport.skip("set TRANSCRIBE_SMOKE_MODEL to demo the run-time error paths")
}

// Scope the model/session so ARC frees the native Metal resources before the
// process exits — ggml-metal (macOS 15+) asserts every GPU resource is released
// before teardown, so a top-level `let` that outlives `main` aborts.
do {
    let session = try Model(path: modelPath).session()

    // 2. Empty PCM is rejected before any compute.
    do {
        _ = try session.run([])
    } catch let error as TranscribeError {
        if case .invalidArgument(let message) = error {
            print("empty pcm -> .invalidArgument: \(message)")
        }
    }

    // 3. Cancellation: a pre-cancelled token aborts the run, preserving any partial.
    if let audioPath = ExampleSupport.audioPath() {
        let pcm = try ExampleSupport.loadWav(audioPath)
        let long = Array(repeating: pcm, count: 4).flatMap { $0 }
        let token = CancellationToken()
        token.cancel()
        session.setCancellationToken(token)
        do {
            _ = try session.run(long)
            print("run completed before the abort was polled")
        } catch let error as TranscribeError {
            if case .aborted(let message, let partial) = error {
                print("cancelled -> .aborted: \(message) (partial: \(partial?.text.isEmpty == false ? "present" : "empty"))")
            }
        }
        session.clearCancellationToken()
    }
}

print("done")
