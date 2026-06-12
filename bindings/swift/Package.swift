// swift-tools-version: 5.9
import PackageDescription

// Placeholder Swift package for transcribe.cpp bindings.
//
// SwiftPM has no central name registry: packages are identified by their Git
// URL, so the "name" you reserve is really the repository URL plus the product
// name below. To be consumable as a remote dependency, a Package.swift must
// live at the ROOT of a git repository (a subdirectory package can only be used
// as a local `path:` dependency), so this skeleton is expected to move to the
// root of a dedicated bindings repo before publishing.
let package = Package(
    name: "transcribe-cpp",
    products: [
        .library(name: "TranscribeCpp", targets: ["TranscribeCpp"]),
    ],
    targets: [
        .target(name: "TranscribeCpp"),
    ]
)
