# TranscribeCpp — Swift bindings for transcribe.cpp

Idiomatic Swift bindings for [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp),
a C/C++ speech-to-text library. Native code ships as a prebuilt, Metal-embedded
`.xcframework` consumed as a SwiftPM `binaryTarget` — no C toolchain, no build
step on the consumer side.

> Status: in development (0.0.1). Every **core** capability is implemented and
> tested; see `notes/bindings-requirements.md` for the matrix.

## Install

Swift Package Manager — add the dependency in Xcode (File → Add Package
Dependencies) or in your `Package.swift`. Apple platforms only: **macOS 13+,
iOS 16+** (device + simulator).

**Option A — the package (recommended).** Add the dependency and the
`"TranscribeCpp"` product:

```swift
.package(url: "https://github.com/handy-computer/transcribe-cpp-swift.git", from: "0.0.1")
```

**Option B — point directly at the prebuilt xcframework.** No extra package
needed: drop a `binaryTarget(url:checksum:)` into your own `Package.swift`,
pointing at the release asset (the checksum is published with each release).
This is the same pattern whisper.cpp documents.

```swift
let package = Package(
    name: "MyApp",
    targets: [
        .executableTarget(name: "MyApp", dependencies: ["CTranscribe"]),
        .binaryTarget(
            name: "CTranscribe",
            url: "https://github.com/handy-computer/transcribe.cpp/releases/download/v0.0.1/TranscribeCpp.xcframework.zip",
            checksum: "<published with the release>"
        ),
    ]
)
```

Option B gives you the raw C module (`import CTranscribe`); Option A adds the
idiomatic Swift wrapper (`import TranscribeCpp`) on top of it.

## Quickstart

```swift
import TranscribeCpp

let model = try Model(path: "/path/to/model.gguf")
let session = try model.session()

// pcm: mono float32 at 16 kHz, in [-1, 1]
let transcript = try session.run(pcm, options: RunOptions(timestamps: .segment))
print(transcript.text)
for segment in transcript.segments {
    print("[\(segment.t0Ms)–\(segment.t1Ms)ms] \(segment.text)")
}
```

Async (hops the blocking call off the calling thread/actor):

```swift
let transcript = try await session.run(pcm)
```

Streaming:

```swift
let stream = try session.stream()
for chunk in chunks {                 // 16 kHz mono float32 frames
    let update = try stream.feed(chunk)
    if update.committedChanged { print(stream.text.committed) }   // flicker-free prefix
}
try stream.finalize()
```

See `Sources/{transcribe-file,streaming,batch,backend-select,error-handling}`
for the five runnable examples — the same set, with the same names, every
first-class binding ships.

## Backend selection

Backends are compiled into the xcframework per Apple slice:

| Slice                    | Backend       |
| ------------------------ | ------------- |
| macOS arm64              | Metal + CPU   |
| macOS x86_64 (Intel)     | CPU only      |
| iOS device (arm64)       | Metal + CPU   |
| iOS simulator            | CPU only      |

`Metal` on Intel Macs and the iOS simulator is intentionally omitted —
ggml's Metal backend is tuned for Apple-GPU families; those targets run on CPU.
Request a backend explicitly with `ModelOptions(backend:)`; an unsatisfiable
request fails cleanly from the load path. Probe first with
`Transcribe.backendAvailable(_:)` / `Transcribe.devices()`.

## Thread-safety contract

- **`Model`** is `Sendable` (`@unchecked`): share it across threads freely.
  Queries and session creation are concurrent; the compute path
  (`run`/`runBatch`/stream feed) is serialized by an internal per-model lock,
  matching the C library's "one in-flight run per model" rule. Concurrent runs
  from many threads queue and execute one at a time.
- **An active stream holds the model's compute lease for its whole lifetime**
  (begin → `finalize`/`reset`/drop). While a stream is live, a `run`,
  `runBatch`, or second `stream` on *any* session of the same model is refused
  with `TranscribeError.busy` rather than racing into undefined behavior — the
  C "one in-flight run per model" rule covers streams, not just offline runs.
  Finish (or drop) the stream before issuing other compute on that model.
- **`Session`** is single-threaded: use one from one thread at a time (it may
  move between threads if never used concurrently). For parallel transcription,
  use one `Session` per thread off a shared `Model` (serialized, one in flight
  at a time); for true parallelism, load one `Model` per worker.
- **Callbacks** (cancellation, logging) may fire on native worker threads and
  are thread-safe. `Transcribe.setLogHandler` should be called once at startup.
- Resource cleanup is automatic under ARC (`deinit` frees the handles); a
  `Session`'s strong reference to its `Model` keeps close ordering safe.
- **Release `Model`s before the process exits on Metal.** On macOS 15+ / iOS 18+,
  ggml's Metal backend keeps model GPU memory resident and asserts every GPU
  resource is released before its own teardown at process exit. A `Model` parked
  in a top-level/global `let` that outlives `main` is therefore still alive when
  that teardown runs and aborts the process *after* your work completes. Normal
  object lifetimes are fine — this only bites the "global model, then exit"
  pattern. In a short-lived program (CLI, script), scope the model so ARC frees
  it before exit; the bundled examples wrap their work in a `do { … }` block for
  exactly this reason.

## Cancellation

Install a `CancellationToken` on a session and call `cancel()` from any thread
to abort an in-flight `run`/`runBatch`/stream `feed`. The aborted call throws
`TranscribeError.aborted` with the partial transcript preserved. This is the
primary, thread-agnostic mechanism, and the same model the Rust and Python
bindings use.

```swift
let token = CancellationToken()
session.setCancellationToken(token)
// later, from anywhere — a Stop button, a timeout:
token.cancel()
```

The `async` `run`/`runBatch` additionally bridge Swift structured concurrency:
cancelling the surrounding `Task` aborts the run (it throws `.aborted`, again
with the partial). This auto-bridge is active **only when you have not installed
your own token** — the native abort has a single slot, so a token you install
takes precedence and `Task.cancel()` is then not observed. Pick one mechanism
per session.

```swift
let task = Task { try await session.run(pcm) }   // no token installed
// …user navigates away / times out:
task.cancel()                                     // run aborts, throws .aborted
```

(The run-mode enum is named `TranscriptionTask`, not `Task`, so it never
shadows Swift's `Task` in files that `import TranscribeCpp`.)

## Objective-C and C++ consumers

The Swift wrapper is Swift-native (value-type results, typed `throws`,
associated-value errors) and does not bridge to Objective-C. ObjC and C++
callers use the **C API directly** — the xcframework bundles the public headers,
so `#import <CTranscribe/transcribe/extensions.h>` exposes the full C surface.
One artifact serves both audiences.

## ABI verification

The per-field struct-layout check that the ctypes (Python) binding performs at
load time is **waived** here: Swift's Clang importer reads the C headers with a
real compiler, so struct layout is correct by construction. What remains:

- A **load-time base-version gate** (`Transcribe.ensureCompatible`, run on every
  `Model.load`): the linked library and this package must agree on the base
  `MAJOR.MINOR.PATCH`.
- A **public-ABI drift gate** in CI: a pinned hash
  (`Transcribe.pinnedHeaderHash`) is checked against `include/transcribe.abihash`
  so a header ABI change turns the build red for conscious review.

## Native artifact notes

The macOS slices contain the same native bytes as the project's desktop
distribution. The **iOS slices are a separate cross-compile** with no desktop
twin — they share the source and the version, but not the byte-for-byte
artifact.
