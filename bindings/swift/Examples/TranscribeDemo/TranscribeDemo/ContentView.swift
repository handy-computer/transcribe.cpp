import SwiftUI
import UIKit
import TranscribeCpp

private enum Compute: String, CaseIterable, Identifiable {
    case auto = "Auto", cpu = "CPU", metal = "Metal"
    var id: String { rawValue }
    var backend: Backend {
        switch self {
        case .auto: return .auto
        case .cpu: return .cpu
        case .metal: return .metal
        }
    }
}

private struct Run {
    let backend: String, device: String
    let audioSec: Double, wallSec: Double, transcript: String
    var realtime: Double { wallSec > 0 ? audioSec / wallSec : 0 }
}

struct ContentView: View {
    @State private var compute: Compute = .auto
    @State private var run: Run?
    @State private var error: String?
    @State private var busy = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Picker("Compute", selection: $compute) {
                        ForEach(Compute.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)

                    Button(action: transcribe) {
                        Label(busy ? "Transcribing…" : "Transcribe jfk.wav",
                              systemImage: busy ? "waveform" : "play.fill")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(busy)

                    if let run {
                        card("Result") {
                            metric("Backend", run.backend)
                            metric("Device", run.device)
                            metric("Audio", String(format: "%.1f s", run.audioSec))
                            metric("Compute", String(format: "%.1f s", run.wallSec))
                            metric("Speed", String(format: "%.1f× realtime", run.realtime))
                        }
                        card("Transcript") {
                            Text(run.transcript)
                                .font(.callout)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }

                    if let error {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .font(.callout)
                            .foregroundStyle(.red)
                    }
                }
                .padding()
            }
            .navigationTitle("transcribe.cpp")
        }
    }

    @ViewBuilder
    private func card<Content: View>(_ title: String,
                                     @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            content()
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground),
                    in: RoundedRectangle(cornerRadius: 12))
    }

    private func metric(_ name: String, _ value: String) -> some View {
        HStack {
            Text(name).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.system(.body, design: .monospaced))
        }
    }

    private func transcribe() {
        busy = true
        error = nil
        let choice = compute
        Task.detached(priority: .userInitiated) {
            do {
                let result = try Self.work(choice)
                await MainActor.run { run = result; busy = false }
            } catch {
                await MainActor.run { self.error = error.localizedDescription; busy = false }
            }
        }
    }

    // 16 kHz mono is the model's expected input; jfk.wav is already that.
    private static let sampleRate = 16_000.0

    private static func work(_ choice: Compute) throws -> Run {
        guard let modelURL = Bundle.main.url(forResource: "whisper-tiny.en-Q5_K_M", withExtension: "gguf"),
              let audioURL = Bundle.main.url(forResource: "jfk", withExtension: "wav") else {
            throw DemoError("bundle is missing the model or jfk.wav")
        }
        let pcm = try WavDecoder.load(audioURL)
        let audioSec = Double(pcm.count) / sampleRate

        // Scope model/session so ARC frees native Metal resources at the brace
        // (ggml-metal asserts every GPU resource is released before teardown).
        var out = ""
        var backendName = ""
        var deviceName = ""
        var wall = 0.0
        do {
            let model = try Model(path: modelURL.path, options: ModelOptions(backend: choice.backend))
            let session = try model.session()
            let start = Date()
            let transcript = try session.run(pcm, options: RunOptions(timestamps: .segment))
            wall = Date().timeIntervalSince(start)
            out = transcript.text.trimmingCharacters(in: .whitespacesAndNewlines)
            backendName = model.backend
            deviceName = model.device.name
        }
        return Run(backend: backendName, device: deviceName,
                   audioSec: audioSec, wallSec: wall, transcript: out)
    }
}

private struct DemoError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

#Preview { ContentView() }
