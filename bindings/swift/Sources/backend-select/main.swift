// backend-select — device discovery, exact selection, backend-policy failure.
import ExampleSupport
import TranscribeCpp

let devices = Transcribe.devices()
print("registered devices:")
for device in devices {
    print("  [\(device.kind)] \(device.name) — \(device.description)")
}

print("\nbackend availability:")
for backend in [Backend.cpu, .metal, .vulkan, .cuda, .rocm] {
    print("  \(backend): \(Transcribe.backendAvailable(backend))")
}

guard let modelPath = ExampleSupport.modelPath() else {
    ExampleSupport.skip("set TRANSCRIBE_SMOKE_MODEL to demo exact device selection")
}
guard let cpu = devices.first(where: { $0.deviceType == .cpu }) else {
    ExampleSupport.skip("no selectable CPU device")
}

// Scope the model so ARC frees it before exit, consistent with the other
// examples (the Metal residency teardown — see transcribe-file).
do {
    // Exact selection passes an enumerated process-local Device, not its
    // display index. CPU provides a deterministic example on every slice.
    let exact = try Model(path: modelPath, options: ModelOptions(device: cpu))
    let selected = try exact.device
    print("\nselected exact device \(selected.name), bound to: \(exact.backend)")
    precondition(selected == cpu)

    // A request that can't be satisfied fails cleanly from the load path.
    if !Transcribe.backendAvailable(.cuda) {
        do {
            _ = try Model(path: modelPath, options: ModelOptions(backend: .cuda))
            print("unexpected: cuda load succeeded")
        } catch {
            print("backend=cuda not available, failed cleanly: \(error)")
        }
    }
}
