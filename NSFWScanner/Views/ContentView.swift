import SwiftUI
import AppKit

struct ContentView: View {
    @Environment(ScanOrchestrator.self) private var orchestrator

    var body: some View {
        NavigationSplitView {
            ScanControlView()
                .navigationSplitViewColumnWidth(min: 240, ideal: 260, max: 320)
        } detail: {
            switch orchestrator.state {
            case .idle:
                VStack(spacing: 32) {
                    Spacer()

                    ContentUnavailableView(
                        "Ready to Scan",
                        systemImage: "eye.trianglebadge.exclamationmark",
                        description: Text("Press Start Scan to analyze your Photos library for NSFW content.")
                    )

                    VStack(alignment: .leading, spacing: 16) {
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Confidence")
                                    .font(.subheadline.bold())
                                Text("How sure the AI needs to be before flagging a photo or video. Raise this if you're seeing too many false positives, lower it to catch more.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        } icon: {
                            Image(systemName: "gauge.with.needle")
                                .foregroundStyle(.blue)
                        }

                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Frame interval")
                                    .font(.subheadline.bold())
                                Text("How many seconds between each video frame check. Lower values are more thorough but take longer to scan.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        } icon: {
                            Image(systemName: "film")
                                .foregroundStyle(.purple)
                        }

                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Photo / Video tasks")
                                    .font(.subheadline.bold())
                                Text("How many items to process at the same time. Higher values use more CPU and GPU but finish the scan faster.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        } icon: {
                            Image(systemName: "bolt.fill")
                                .foregroundStyle(.orange)
                        }
                    }
                    .padding(.horizontal, 60)
                    .frame(maxWidth: 500)

                HStack(spacing: 12) {
                    Image(systemName: "cpu")
                        .foregroundStyle(.teal)
                        .font(.title3)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Current Model: \(orchestrator.selectedModel.displayName)")
                            .font(.subheadline.bold())
                        Text(orchestrator.selectedModel.subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
                .frame(maxWidth: 500)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))

                    Spacer()
                }
            case .requestingPermission:
                ProgressView("Requesting Photos access...")
            case .scanning:
                scanProgressView
            case .reviewing, .committingToAlbum:
                if orchestrator.flaggedResults.isEmpty {
                    ContentUnavailableView(
                        "All Clear",
                        systemImage: "checkmark.shield.fill",
                        description: Text("Scanned \(orchestrator.processedCount) items. No NSFW content detected.")
                    )
                } else {
                    ResultsGridView()
                }
            case .error(let message):
                let isPermissionError = message.contains("denied") || message.contains("access")
                ContentUnavailableView {
                    Label("Error", systemImage: "exclamationmark.triangle.fill")
                } description: {
                    Text(message)
                } actions: {
                    if isPermissionError {
                        Button("Open System Settings") {
                            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Photos") {
                                NSWorkspace.shared.open(url)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    Button("Try Again") {
                        orchestrator.resetToIdle()
                    }
                }
            }
        }
        .animation(.easeInOut(duration: 0.3), value: orchestrator.state)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                switch orchestrator.state {
                case .idle, .error:
                    Button("Start Scan", systemImage: "play.fill") {
                        orchestrator.startScan()
                    }
                    .disabled(!orchestrator.scanImages && !orchestrator.scanVideos)
                case .scanning:
                    Button("Cancel", systemImage: "stop.fill", role: .destructive) {
                        orchestrator.cancelScan()
                    }
                case .reviewing, .committingToAlbum:
                    Button("New Scan", systemImage: "arrow.counterclockwise") {
                        orchestrator.resetToIdle()
                    }
                case .requestingPermission:
                    ProgressView()
                        .controlSize(.small)
                }
            }
        }
    }

    @ViewBuilder
    private var scanProgressView: some View {
        VStack(spacing: 24) {
            Spacer()

            // Image progress
            if orchestrator.totalImages > 0 {
                progressRow(
                    icon: "photo.fill",
                    label: "Images",
                    processed: orchestrator.processedImages,
                    total: orchestrator.totalImages,
                    flagged: orchestrator.flaggedImageCount,
                    progress: orchestrator.imageProgress,
                    tint: .blue
                )
            }

            // Video progress
            if orchestrator.totalVideos > 0 {
                progressRow(
                    icon: "film.fill",
                    label: "Videos",
                    processed: orchestrator.processedVideos,
                    total: orchestrator.totalVideos,
                    flagged: orchestrator.flaggedVideoCount,
                    progress: orchestrator.videoProgress,
                    tint: .purple
                )
            }

            // Overall
            Text("\(orchestrator.processedCount) of \(orchestrator.totalAssets) total")
                .font(.subheadline)
                .foregroundStyle(.tertiary)

            Spacer()
        }
        .padding(.horizontal, 60)
    }

    private func progressRow(
        icon: String,
        label: String,
        processed: Int,
        total: Int,
        flagged: Int,
        progress: Double,
        tint: Color
    ) -> some View {
        VStack(spacing: 8) {
            HStack {
                Label(label, systemImage: icon)
                    .font(.headline)
                    .foregroundStyle(tint)
                Spacer()
                Text("\(processed) / \(total)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: progress)
                .tint(tint)
            if flagged > 0 {
                HStack {
                    Spacer()
                    Text("\(flagged) flagged")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
        }
    }
}
