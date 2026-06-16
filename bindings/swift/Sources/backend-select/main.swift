// backend-select — device discovery, explicit backend request, graceful failure.
import ExampleSupport
import TranscribeCpp

print("registered devices:")
for device in Transcribe.devices() {
    print("  [\(device.kind)] \(device.name) — \(device.description)")
}

print("\nbackend availability:")
for backend in [Backend.cpu, .metal, .vulkan, .cuda] {
    print("  \(backend): \(Transcribe.backendAvailable(backend))")
}

guard let modelPath = ExampleSupport.modelPath() else {
    ExampleSupport.skip("set TRANSCRIBE_SMOKE_MODEL to demo explicit backend selection")
}

// Scope the model so ARC frees it before exit, consistent with the other
// examples (the Metal residency teardown — see transcribe-file). This load is
// CPU-only so the assert wouldn't fire, but keeping the shape uniform avoids
// "why doesn't this one scope?" confusion.
do {
    // CPU always succeeds.
    let onCpu = try Model(path: modelPath, options: ModelOptions(backend: .cpu))
    print("\nloaded with backend=cpu, bound to: \(onCpu.backend)")

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
