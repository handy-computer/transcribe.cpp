# TranscribeCpp

Swift bindings for [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp),
a C/C++ speech-to-text library built on ggml. Native code ships as a prebuilt
`.xcframework` SwiftPM `binaryTarget`, with Metal embedded on supported Apple
slices.

> Status: in development (0.0.1). Core model, session, run, stream,
> cancellation, backend, and family-extension APIs are implemented and tested.

## Install

Apple platforms only: **macOS 13+** and **iOS 16+**.

For development from this repository, use the package in `bindings/swift`. It
expects `bindings/swift/build-apple/TranscribeCpp.xcframework` by default, or a
custom artifact path through `TRANSCRIBE_XCFRAMEWORK_PATH`.

The standalone SwiftPM mirror is planned but not published yet:

```swift
.package(url: "https://github.com/handy-computer/transcribe-cpp-swift.git", from: "0.0.1")
```

Until that mirror repo and tag exist, use the release xcframework directly when
you only need the raw C module:

```swift
.binaryTarget(
    name: "CTranscribe",
    url: "https://github.com/handy-computer/transcribe.cpp/releases/download/v0.0.1/TranscribeCpp.xcframework.zip",
    checksum: "<published with the release>"
)
```

The direct binary target exposes `import CTranscribe`; the local and planned
Swift packages expose the wrapper product, `import TranscribeCpp`.

## Quickstart

```swift
import TranscribeCpp

let model = try Model(path: "/path/to/model.gguf")
let session = try model.session()

// pcm: mono float32 at 16 kHz, in [-1, 1]
let transcript = try session.run(pcm, options: RunOptions(timestamps: .segment))
print(transcript.text)

for segment in transcript.segments {
    print("[\(segment.t0Ms)-\(segment.t1Ms)ms] \(segment.text)")
}
```

`run` is blocking; `try await session.run(pcm)` uses the async convenience
overload and hops the work off the caller's thread.

Streaming models expose committed/tentative text for UI display:

```swift
let stream = try session.stream()
for chunk in chunks {                 // 16 kHz mono float32 frames
    let update = try stream.feed(chunk)
    if update.committedChanged { print(stream.text.committed) }
}
try stream.finalize()
```

Runnable examples live in
`Sources/{transcribe-file,streaming,batch,backend-select,error-handling}`.

## Backends

Backends are compiled into the xcframework per Apple slice:

| Slice                | Backend     |
| -------------------- | ----------- |
| macOS arm64          | Metal + CPU |
| macOS x86_64         | CPU only    |
| iOS device arm64     | Metal + CPU |
| iOS simulator        | CPU only    |

Request a backend with `ModelOptions(backend:)`; probe availability with
`Transcribe.backendAvailable(_:)` or inspect `Transcribe.devices()`.

## Concurrency and lifetime

- `Model` is shareable. In 0.x, compute is serialized per model, so concurrent
  runs queue; load one `Model` per worker for true parallelism.
- `Session` is single-threaded. Use one session from one thread at a time.
- An active `Stream` holds the model's compute lease until `finalize`, `reset`,
  or drop. Other runs/streams on that model fail with `TranscribeError.busy`.
- `Transcribe.setLogHandler` is best installed at startup. Repeated calls are
  safe; they swap the Swift handler behind one native trampoline.
- On Metal, do not keep models in globals in short-lived programs. Scope models
  so ARC releases GPU resources before process exit; the examples use `do { }`
  blocks for this reason.

## Cancellation

Install a token on a session and cancel it from any thread:

```swift
let token = CancellationToken()
session.setCancellationToken(token)
token.cancel()
```

The active `run`, `runBatch`, or stream feed throws `TranscribeError.aborted`
with any partial transcript preserved. Async `run`/`runBatch` also bridge Swift
task cancellation when no custom token is installed.

## C, Objective-C, and C++

The xcframework also exposes the raw C module as `CTranscribe`. Objective-C and
C++ callers use the bundled C headers directly, for example
`#import <CTranscribe/transcribe/extensions.h>`.
