import SwiftUI
import Photos

struct ResultsGridView: View {
    @Environment(ScanOrchestrator.self) private var orchestrator
    @State private var selectedIDs: Set<String> = []
    @State private var inspectedResult: ScanResult?
    @State private var commitInProgress = false

    private let columns = [
        GridItem(.adaptive(minimum: 150, maximum: 200), spacing: 8)
    ]

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVGrid(columns: columns, spacing: 8) {
                    ForEach(orchestrator.flaggedResults) { result in
                        ThumbnailCell(result: result, isSelected: selectedIDs.contains(result.id))
                            .contentShape(Rectangle())
                            .onTapGesture(count: 2) {
                                inspectedResult = result
                            }
                            .onTapGesture(count: 1) {
                                toggleSelection(result.id)
                            }
                    }
                }
                .padding()
            }

            Divider()

            HStack(spacing: 12) {
                Text("\(orchestrator.flaggedResults.count) flagged, \(selectedIDs.count) selected")
                    .foregroundStyle(.secondary)
                    .font(.callout)

                Spacer()

                Button("Select All") {
                    selectedIDs = Set(orchestrator.flaggedResults.map(\.id))
                }

                Button("Deselect") {
                    selectedIDs.removeAll()
                }
                .disabled(selectedIDs.isEmpty)

                Button("Dismiss Selected", systemImage: "xmark.circle") {
                    orchestrator.dismissResults(ids: selectedIDs)
                    selectedIDs.removeAll()
                }
                .disabled(selectedIDs.isEmpty)

                if commitInProgress {
                    ProgressView()
                        .controlSize(.small)
                        .padding(.horizontal, 8)
                } else {
                    Button("Add to NSFW Album", systemImage: "folder.badge.plus") {
                        let idsToCommit = selectedIDs.isEmpty
                            ? Set(orchestrator.flaggedResults.map(\.id))
                            : selectedIDs
                        commitInProgress = true
                        Task {
                            await orchestrator.commitFlaggedToAlbum(selectedIDs: idsToCommit)
                            commitInProgress = false
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding()
        }
        .sheet(item: $inspectedResult) { result in
            AssetDetailView(result: result)
        }
    }

    private func toggleSelection(_ id: String) {
        if selectedIDs.contains(id) {
            selectedIDs.remove(id)
        } else {
            selectedIDs.insert(id)
        }
    }
}

struct ThumbnailCell: View {
    let result: ScanResult
    let isSelected: Bool

    @State private var thumbnail: NSImage?
    private static let photoLibrary = PhotoLibraryService()

    var body: some View {
        ZStack(alignment: .topTrailing) {
            ZStack {
                if let thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .aspectRatio(1, contentMode: .fill)
                        .clipped()
                } else {
                    Rectangle()
                        .fill(.quaternary)
                        .aspectRatio(1, contentMode: .fill)
                        .overlay {
                            ProgressView()
                        }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isSelected ? Color.accentColor : .clear, lineWidth: 3)
            )

            // Selection checkmark
            if isSelected {
                Image(systemName: "checkmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.white, Color.accentColor)
                    .padding(6)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }

            VStack(alignment: .trailing, spacing: 4) {
                Text(confidenceText)
                    .font(.caption2.bold())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.red, in: Capsule())
                    .foregroundStyle(.white)

                Image(systemName: result.mediaType == .video ? "film" : "camera")
                    .font(.caption2)
                    .padding(4)
                    .background(.black.opacity(0.6), in: Circle())
                    .foregroundStyle(.white)
            }
            .padding(6)
        }
        .task {
            do {
                thumbnail = try await Self.photoLibrary.thumbnail(for: result.asset, size: CGSize(width: 300, height: 300))
            } catch {
                // Thumbnail load failed, show placeholder
            }
        }
    }

    private var confidenceText: String {
        let pct = min(max(Int(result.confidence * 100), 0), 100)
        return "\(pct)%"
    }
}
