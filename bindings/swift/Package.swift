// swift-tools-version: 5.9
import PackageDescription
import Foundation

// transcribe.cpp Swift bindings (`TranscribeCpp`).
//
// Native code is consumed as a prebuilt `.xcframework` binaryTarget (the
// project's distribution posture for Swift; see notes/swift-bindings-plan.md
// and notes/bindings-requirements.md §5). The xcframework bundles a merged
// static `libtranscribe` (Metal embedded on the slices that support it) plus
// the public C headers + module map, so `import CTranscribe` exposes the full
// C surface to the Swift wrapper.
//
// Resolution of the binaryTarget:
//   - DEV/CI: set TRANSCRIBE_XCFRAMEWORK_PATH to a locally built xcframework,
//     or rely on the default `build-apple/TranscribeCpp.xcframework`
//     (produced by scripts/ci/build_xcframework.sh).
//   - RELEASE (mirror repo): a remote `binaryTarget(url:checksum:)` pointing at
//     the GitHub release asset is substituted by the publish job.
let xcframeworkPath = Context.environment["TRANSCRIBE_XCFRAMEWORK_PATH"]
    ?? "build-apple/TranscribeCpp.xcframework"

let package = Package(
    name: "transcribe-cpp",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "TranscribeCpp", targets: ["TranscribeCpp"]),
        // The five canonical, CI-executed examples (requirements §6), identical
        // names across every first-class binding.
        .executable(name: "transcribe-file", targets: ["transcribe-file"]),
        .executable(name: "streaming", targets: ["streaming"]),
        .executable(name: "batch", targets: ["batch"]),
        .executable(name: "backend-select", targets: ["backend-select"]),
        .executable(name: "error-handling", targets: ["error-handling"]),
    ],
    targets: [
        // The prebuilt native artifact. Module `CTranscribe` (per the bundled
        // module.modulemap) is the raw C surface.
        .binaryTarget(name: "CTranscribe", path: xcframeworkPath),

        // The idiomatic Swift wrapper. System libs/frameworks that the merged
        // static archive does NOT carry are linked here (the canonical set
        // comes from lib/transcribe-link.json). Linux Swift is out of scope for
        // v1 — these settings are Apple-only by construction.
        .target(
            name: "TranscribeCpp",
            dependencies: ["CTranscribe"],
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedLibrary("z"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
            ]
        ),

        .testTarget(
            name: "TranscribeCppTests",
            dependencies: ["TranscribeCpp"]
        ),

        // Examples: shared fixture plumbing + one executable per canonical name.
        .target(name: "ExampleSupport", dependencies: ["TranscribeCpp"]),
        .executableTarget(name: "transcribe-file", dependencies: ["TranscribeCpp", "ExampleSupport"]),
        .executableTarget(name: "streaming", dependencies: ["TranscribeCpp", "ExampleSupport"]),
        .executableTarget(name: "batch", dependencies: ["TranscribeCpp", "ExampleSupport"]),
        .executableTarget(name: "backend-select", dependencies: ["TranscribeCpp", "ExampleSupport"]),
        .executableTarget(name: "error-handling", dependencies: ["TranscribeCpp", "ExampleSupport"]),
    ]
)
