import Foundation
import XCTest

@testable import TranscribeCpp

/// Model-gated device-selection tier. Exercises `Model.device` correlation and
/// `ModelOptions(gpuDevice:)` validation against a real load. Skips cleanly
/// (XCTSkip) when no canary model is present (see `Fixtures.modelPath()`).
///
/// Each loaded model is scoped in a `do { }` block so ARC frees the native
/// Metal resources before the test process exits — ggml-metal (macOS 15+
/// residency sets) asserts every GPU resource is released before teardown (see
/// the comment in Sources/transcribe-file/main.swift).
final class DeviceSelectTests: XCTestCase {
    /// `Model.device` reports the compute device the loaded model runs on. It
    /// has no registry `index` (nil), and correlates back to one of the
    /// enumerated `devices()` by `name` (and `deviceId` when reported).
    func testModelDeviceCorrelatesToEnumeration() throws {
        guard let modelPath = Fixtures.modelPath() else {
            throw XCTSkip("no canary model (set TRANSCRIBE_SMOKE_MODEL)")
        }
        do {
            let model = try Model(path: modelPath)
            let device = try model.device
            // `Model.device` does not expose a registry index.
            XCTAssertNil(device.index, "Model.device should not carry a registry index")

            let enumerated = Transcribe.devices()
            let match = enumerated.first { candidate in
                guard candidate.name == device.name else { return false }
                if let id = device.deviceId { return candidate.deviceId == id }
                return true
            }
            XCTAssertNotNil(
                match,
                "model device \(device.name)/\(device.deviceId ?? "nil") not found in devices()")
        }
    }

    /// A negative gpuDevice index is invalid.
    func testNegativeGpuDeviceIsRejected() throws {
        guard let modelPath = Fixtures.modelPath() else {
            throw XCTSkip("no canary model (set TRANSCRIBE_SMOKE_MODEL)")
        }
        XCTAssertThrowsError(
            try Model(path: modelPath, options: ModelOptions(gpuDevice: -1))
        ) { error in
            guard case TranscribeError.invalidArgument = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
        }
    }

    /// A gpuDevice index past the end of the registry is invalid.
    func testOutOfRangeGpuDeviceIsRejected() throws {
        guard let modelPath = Fixtures.modelPath() else {
            throw XCTSkip("no canary model (set TRANSCRIBE_SMOKE_MODEL)")
        }
        let outOfRange = Int32(Transcribe.devices().count) + 1000
        XCTAssertThrowsError(
            try Model(path: modelPath, options: ModelOptions(gpuDevice: outOfRange))
        ) { error in
            guard case TranscribeError.invalidArgument = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
        }
    }

    /// Selecting a GPU device on the CPU backend is invalid (no GPU to select).
    /// Hardware-independent: the CPU backend exposes a single device at index 0.
    func testGpuDeviceOnCpuBackendIsRejected() throws {
        guard let modelPath = Fixtures.modelPath() else {
            throw XCTSkip("no canary model (set TRANSCRIBE_SMOKE_MODEL)")
        }
        XCTAssertThrowsError(
            try Model(path: modelPath, options: ModelOptions(backend: .cpu, gpuDevice: 1))
        ) { error in
            guard case TranscribeError.invalidArgument = error else {
                return XCTFail("expected .invalidArgument, got \(error)")
            }
        }
    }
}
