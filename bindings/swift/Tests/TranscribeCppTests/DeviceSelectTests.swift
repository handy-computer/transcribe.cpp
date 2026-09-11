import XCTest
@testable import TranscribeCpp

final class DeviceSelectTests: XCTestCase {
    func testEnumeratedDeviceCanBeSelectedExactly() throws {
        guard let modelPath = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        guard let device = Transcribe.devices().first(where: { $0.deviceType != .accel }) else {
            throw XCTSkip("no selectable device")
        }
        do {
            let model = try Model(path: modelPath, options: ModelOptions(device: device))
            XCTAssertEqual(try model.device, device)
        } catch TranscribeError.backend {
            // A registered device may still fail driver initialization; exact
            // selection reports that failure instead of moving elsewhere.
        }
    }

    func testExplicitBackendMustMatchDevice() throws {
        guard let modelPath = Fixtures.modelPath() else { throw XCTSkip("no canary model") }
        guard let gpu = Transcribe.devices().first(where: {
            $0.deviceType == .gpu || $0.deviceType == .igpu
        }) else { throw XCTSkip("no GPU device") }

        XCTAssertThrowsError(
            try Model(path: modelPath, options: ModelOptions(backend: .cpu, device: gpu))
        ) { error in
            guard case TranscribeError.invalidArgument = error else {
                return XCTFail("expected invalidArgument, got \(error)")
            }
        }
    }
}
