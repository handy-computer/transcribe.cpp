# TranscribeDemo — iOS on-device example

A minimal SwiftUI app that runs the `TranscribeCpp` binding on a real iPhone (or
the simulator): pick a compute backend (Auto / CPU / Metal), transcribe the
bundled `jfk.wav`, and see the resolved device and the real-time factor (e.g.
`55.0× realtime`) alongside the transcript.

It consumes the binding the normal way a real app would — the local Swift
package and `import TranscribeCpp`. Xcode embeds the dynamic framework into the
app bundle automatically; there are no special linker flags or build settings.

## One-time setup

1. **Build the xcframework** from the repo root:
   ```sh
   scripts/ci/build_xcframework.sh
   # device-only (faster) while iterating on a phone:
   #   TRANSCRIBE_XCFRAMEWORK_SLICES=ios-device scripts/ci/build_xcframework.sh
   ```
   This writes `bindings/swift/build-apple/TranscribeCpp.xcframework`, which the
   local Swift package resolves.

2. **Drop in a model.** A small GGUF is expected at
   `TranscribeDemo/Resources/whisper-tiny.en-Q5_K_M.gguf` (gitignored — models
   are large/licensed). Any small whisper-family `.gguf` works; if you rename it,
   update the `forResource:` string in `ContentView.swift`.

3. **Generate the project and open it:**
   ```sh
   brew install xcodegen          # one-time
   cd bindings/swift/Examples/TranscribeDemo
   xcodegen generate
   open TranscribeDemo.xcodeproj
   ```

## Run

- **On your iPhone:** select the `TranscribeDemo` scheme + your device, set your
  Team under *Signing & Capabilities* (automatic signing is fine), and Run.
- **On the simulator:** pick any iOS Simulator destination and Run. The simulator
  slice is CPU-only (no Metal) — the **Metal** button will report an error there;
  use **Auto** or **CPU**. (Requires the `ios-sim` slice — run the full
  `build_xcframework.sh`, not the device-only one.)

> **Toolchain note:** to *debug-launch* on a device running an iOS that's newer
> than your Xcode's SDK (e.g. an iOS 27 beta with a release Xcode), use the
> matching Xcode (the beta). A toolchain/OS mismatch crashes the app at launch
> before any of its code runs — it is **not** a problem with the binding.
