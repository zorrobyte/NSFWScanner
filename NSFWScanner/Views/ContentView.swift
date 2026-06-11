import SwiftUI
import AppKit

enum AppMode: String, CaseIterable {
    case nsfwScan = "NSFW Scan"
    case sortLibrary = "Sort Library"
}

struct ContentView: View {
    @Environment(ScanOrchestrator.self) private var orchestrator
    @Environment(SortOrchestrator.self) private var sortOrchestrator
    @State private var mode: AppMode = .nsfwScan

    private var isBusy: Bool {
        orchestrator.state == .scanning || sortOrchestrator.state == .scanning
    }

    var body: some View {
        NavigationSplitView {
            VStack(spacing: 0) {
                Picker("Mode", selection: $mode) {
                    ForEach(AppMode.allCases, id: \.self) { m in
                        Text(m.rawValue).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .padding(.bottom, 4)
                .disabled(isBusy)

                switch mode {
                case .nsfwScan:
                    ScanControlView()
                case .sortLibrary:
                    SortControlView()
                }
            }
            .navigationSplitViewColumnWidth(min: 240, ideal: 280, max: 340)
        } detail: {
            switch mode {
            case .nsfwScan:
                nsfwDetailView
            case .sortLibrary:
                sortDetailView
            }
        }
        .animation(.easeInOut(duration: 0.3), value: orchestrator.state)
        .animation(.easeInOut(duration: 0.3), value: sortOrchestrator.state)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                switch mode {
                case .nsfwScan:
                    nsfwToolbarButtons
                case .sortLibrary:
                    sortToolbarButtons
                }
            }
        }
    }

    // MARK: - NSFW Scan Detail

    @ViewBuilder
    private var nsfwDetailView: some View {
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
            nsfwScanProgressView
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
            errorView(message: message, onRetry: { orchestrator.resetToIdle() })
        }
    }

    // MARK: - Sort Library Detail

    @ViewBuilder
    private var sortDetailView: some View {
        switch sortOrchestrator.state {
        case .idle:
            VStack(spacing: 32) {
                Spacer()

                ContentUnavailableView(
                    "Ready to Sort",
                    systemImage: "rectangle.3.group",
                    description: Text("Press Start Sort to classify your photos into albums using the vision model on your 5090.")
                )

                VStack(alignment: .leading, spacing: 16) {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Vision Model")
                                .font(.subheadline.bold())
                            Text("Photos are sent to Qwen3-VL-8B on your 5090 desktop for classification into categories.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: "eye.circle")
                            .foregroundStyle(.green)
                    }

                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Album Prefix")
                                .font(.subheadline.bold())
                            Text("Albums will be named with a prefix like \"Sorted - Landscapes\" to keep them distinct.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: "folder.badge.gearshape")
                            .foregroundStyle(.orange)
                    }

                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Categories")
                                .font(.subheadline.bold())
                            Text("Each photo is classified into exactly one category. Configure the list in the sidebar.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: "tag")
                            .foregroundStyle(.purple)
                    }
                }
                .padding(.horizontal, 60)
                .frame(maxWidth: 500)

                HStack(spacing: 12) {
                    Image(systemName: "server.rack")
                        .foregroundStyle(.green)
                        .font(.title3)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Endpoint: \(sortOrchestrator.endpointURL)")
                            .font(.subheadline.bold())
                            .lineLimit(1)
                        Text("\(sortOrchestrator.categories.count) categories configured")
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
            sortProgressView
        case .reviewing, .committingToAlbum:
            if sortOrchestrator.sortedResults.isEmpty {
                ContentUnavailableView(
                    "All Done",
                    systemImage: "checkmark.circle.fill",
                    description: Text("Processed \(sortOrchestrator.processedPhotos) photos. All albums updated.")
                )
            } else {
                SortResultsGridView()
            }
        case .error(let message):
            errorView(message: message, onRetry: { sortOrchestrator.resetToIdle() })
        }
    }

    // MARK: - Progress Views

    @ViewBuilder
    private var nsfwScanProgressView: some View {
        VStack(spacing: 24) {
            Spacer()

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

            Text("\(orchestrator.processedCount) of \(orchestrator.totalAssets) total")
                .font(.subheadline)
                .foregroundStyle(.tertiary)

            Spacer()
        }
        .padding(.horizontal, 60)
    }

    @ViewBuilder
    private var sortProgressView: some View {
        VStack(spacing: 16) {
            progressRow(
                icon: "photo.fill",
                label: "Classifying",
                processed: sortOrchestrator.processedPhotos,
                total: sortOrchestrator.totalPhotos,
                flagged: sortOrchestrator.sortedResults.count,
                progress: sortOrchestrator.photoProgress,
                tint: .green,
                flaggedLabel: "classified"
            )
            .padding(.horizontal, 60)

            if sortOrchestrator.committedPhotos > 0 {
                HStack(spacing: 6) {
                    Image(systemName: "folder.badge.plus")
                        .foregroundStyle(.green)
                    Text("\(sortOrchestrator.committedPhotos) moved to albums")
                        .font(.subheadline.bold())
                        .foregroundStyle(.green)
                }
            }

            // Live category counts
            if !sortOrchestrator.categoryCounts.isEmpty {
                HStack(spacing: 16) {
                    ForEach(sortOrchestrator.categoryCounts.prefix(6), id: \.category) { item in
                        VStack(spacing: 2) {
                            Text("\(item.count)")
                                .monospacedDigit()
                                .font(.title3.bold())
                                .foregroundStyle(.green)
                            Text(item.category)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                }
                .padding(.horizontal)
            }

            // Live log view
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(sortOrchestrator.logEntries.enumerated()), id: \.offset) { index, entry in
                            Text(entry)
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(entry.contains("ERROR") || entry.contains("FAIL") ? .red : .secondary)
                                .id(index)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(.black.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.quaternary, lineWidth: 1)
                )
                .padding(.horizontal, 20)
                .onChange(of: sortOrchestrator.logEntries.count) { _, newCount in
                    if newCount > 0 {
                        proxy.scrollTo(newCount - 1, anchor: .bottom)
                    }
                }
            }
        }
        .padding(.top, 16)
    }

    // MARK: - Toolbar Buttons

    @ViewBuilder
    private var nsfwToolbarButtons: some View {
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

    @ViewBuilder
    private var sortToolbarButtons: some View {
        switch sortOrchestrator.state {
        case .idle, .error:
            Button("Start Sort", systemImage: "play.fill") {
                sortOrchestrator.startSort()
            }
            .tint(.green)
        case .scanning:
            Button("Cancel", systemImage: "stop.fill", role: .destructive) {
                sortOrchestrator.cancelSort()
            }
        case .reviewing, .committingToAlbum:
            Button("New Sort", systemImage: "arrow.counterclockwise") {
                sortOrchestrator.resetToIdle()
            }
        case .requestingPermission:
            ProgressView()
                .controlSize(.small)
        }
    }

    // MARK: - Shared Helpers

    private func progressRow(
        icon: String,
        label: String,
        processed: Int,
        total: Int,
        flagged: Int,
        progress: Double,
        tint: Color,
        flaggedLabel: String = "flagged"
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
                    Text("\(flagged) \(flaggedLabel)")
                        .font(.caption)
                        .foregroundStyle(tint)
                }
            }
        }
    }

    private func errorView(message: String, onRetry: @escaping () -> Void) -> some View {
        let isPermissionError = message.contains("denied") || message.contains("access")
        return ContentUnavailableView {
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
                onRetry()
            }
        }
    }
}
