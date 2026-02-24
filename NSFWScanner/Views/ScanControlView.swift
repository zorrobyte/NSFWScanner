import SwiftUI

struct ScanControlView: View {
    @Environment(ScanOrchestrator.self) private var orchestrator

    var body: some View {
        @Bindable var orchestrator = orchestrator

        List {
            if orchestrator.state == .scanning {
                Section("Progress") {
                    if orchestrator.totalImages > 0 {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Image(systemName: "photo")
                                    .foregroundStyle(.blue)
                                Text("\(orchestrator.processedImages) / \(orchestrator.totalImages)")
                                    .monospacedDigit()
                                Spacer()
                                if orchestrator.flaggedImageCount > 0 {
                                    Text("\(orchestrator.flaggedImageCount) flagged")
                                        .font(.caption2)
                                        .foregroundStyle(.red)
                                }
                            }
                            .font(.caption)
                            ProgressView(value: orchestrator.imageProgress)
                        }
                    }
                    if orchestrator.totalVideos > 0 {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Image(systemName: "film")
                                    .foregroundStyle(.purple)
                                Text("\(orchestrator.processedVideos) / \(orchestrator.totalVideos)")
                                    .monospacedDigit()
                                Spacer()
                                if orchestrator.flaggedVideoCount > 0 {
                                    Text("\(orchestrator.flaggedVideoCount) flagged")
                                        .font(.caption2)
                                        .foregroundStyle(.red)
                                }
                            }
                            .font(.caption)
                            ProgressView(value: orchestrator.videoProgress)
                                .tint(.purple)
                        }
                    }
                }
            }

            if orchestrator.state == .scanning || orchestrator.state == .reviewing {
                Section("Results") {
                    HStack {
                        Text("Images flagged")
                        Spacer()
                        Text("\(orchestrator.flaggedImageCount)")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Videos flagged")
                        Spacer()
                        Text("\(orchestrator.flaggedVideoCount)")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Total scanned")
                        Spacer()
                        Text("\(orchestrator.processedCount)")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section("Scan Types") {
                Toggle("Photos", isOn: $orchestrator.scanImages)
                Toggle("Videos", isOn: $orchestrator.scanVideos)
            }
            .disabled(orchestrator.state == .scanning)

            Section("Model") {
                Picker("Model", selection: $orchestrator.selectedModel) {
                    ForEach(NSFWModel.allCases) { model in
                        VStack(alignment: .leading) {
                            Text(model.displayName)
                            Text(model.subtitle)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        .tag(model)
                    }
                }
                .pickerStyle(.menu)

            }
            .disabled(orchestrator.state == .scanning)

            Section("Settings") {
                TextField("Album name", text: $orchestrator.albumName)
                    .textFieldStyle(.roundedBorder)

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Confidence")
                        Spacer()
                        Text("\(Int(orchestrator.confidenceThreshold * 100))%")
                            .foregroundStyle(.secondary)
                    }
                    Slider(
                        value: $orchestrator.confidenceThreshold,
                        in: 0.50...0.99,
                        step: 0.01
                    )
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Frame interval")
                        Spacer()
                        Text("\(Int(orchestrator.frameInterval))s")
                            .foregroundStyle(.secondary)
                    }
                    Slider(
                        value: $orchestrator.frameInterval,
                        in: 1...30,
                        step: 1
                    )
                }
            }
            .disabled(orchestrator.state == .scanning)

            Section("Performance") {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Photo tasks")
                        Spacer()
                        Text("\(Int(orchestrator.imageConcurrency))")
                            .foregroundStyle(.secondary)
                    }
                    Slider(
                        value: $orchestrator.imageConcurrency,
                        in: 1...128,
                        step: 1
                    )
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Video tasks")
                        Spacer()
                        Text("\(Int(orchestrator.videoConcurrency))")
                            .foregroundStyle(.secondary)
                    }
                    Slider(
                        value: $orchestrator.videoConcurrency,
                        in: 1...128,
                        step: 1
                    )
                }
            }
            .disabled(orchestrator.state == .scanning)
        }
        .navigationTitle("NSFWScanner")
    }
}
