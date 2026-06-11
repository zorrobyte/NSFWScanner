import SwiftUI

struct SortControlView: View {
    @Environment(SortOrchestrator.self) private var orchestrator

    var body: some View {
        @Bindable var orchestrator = orchestrator

        List {
            if orchestrator.state == .scanning {
                Section("Progress") {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Image(systemName: "photo")
                                .foregroundStyle(.green)
                            Text("\(orchestrator.processedPhotos) / \(orchestrator.totalPhotos)")
                                .monospacedDigit()
                            Spacer()
                        }
                        .font(.caption)
                        ProgressView(value: orchestrator.photoProgress)
                            .tint(.green)
                    }
                }
            }

            if orchestrator.state == .scanning || orchestrator.state == .reviewing {
                Section("Results") {
                    HStack {
                        Text("Classified")
                        Spacer()
                        Text("\(orchestrator.sortedResults.count)")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Moved to albums")
                        Spacer()
                        Text("\(orchestrator.committedPhotos)")
                            .monospacedDigit()
                            .foregroundStyle(.green)
                    }
                    ForEach(orchestrator.categoryCounts, id: \.category) { item in
                        HStack {
                            Text(item.category)
                                .font(.caption)
                            Spacer()
                            Text("\(item.count)")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                                .font(.caption)
                        }
                    }
                }
            }

            Section("Categories") {
                ForEach(orchestrator.categories) { category in
                    Text(category.name)
                        .font(.callout)
                }
            }

            Section("Endpoint") {
                TextField("Vision API URL", text: $orchestrator.endpointURL)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
            }
            .disabled(orchestrator.state == .scanning)

            Section("Settings") {
                TextField("Album prefix", text: $orchestrator.albumPrefix)
                    .textFieldStyle(.roundedBorder)

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Concurrency")
                        Spacer()
                        Text("\(Int(orchestrator.concurrency))")
                            .foregroundStyle(.secondary)
                    }
                    Slider(
                        value: $orchestrator.concurrency,
                        in: 1...32,
                        step: 1
                    )
                }
            }
            .disabled(orchestrator.state == .scanning)
        }
        .navigationTitle("Sort Library")
    }
}
