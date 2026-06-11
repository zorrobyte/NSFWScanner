import SwiftUI
import Photos

struct SortResultsGridView: View {
    @Environment(SortOrchestrator.self) private var orchestrator
    @State private var commitInProgress = false

    private let columns = [
        GridItem(.adaptive(minimum: 150, maximum: 200), spacing: 8)
    ]

    /// Results grouped by category, sorted by count descending.
    private var groupedResults: [(category: String, results: [SortResult])] {
        var groups: [String: [SortResult]] = [:]
        for result in orchestrator.sortedResults {
            groups[result.category, default: []].append(result)
        }
        return groups.sorted { $0.value.count > $1.value.count }
            .map { (category: $0.key, results: $0.value) }
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16, pinnedViews: .sectionHeaders) {
                    ForEach(groupedResults, id: \.category) { group in
                        Section {
                            LazyVGrid(columns: columns, spacing: 8) {
                                ForEach(group.results) { result in
                                    SortThumbnailCell(result: result)
                                }
                            }
                        } header: {
                            HStack {
                                Text(group.category)
                                    .font(.headline)
                                Text("\(group.results.count)")
                                    .font(.caption)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 2)
                                    .background(.green.opacity(0.2), in: Capsule())
                                    .foregroundStyle(.green)
                                Spacer()
                            }
                            .padding(.horizontal)
                            .padding(.vertical, 8)
                            .background(.bar)
                        }
                    }
                }
                .padding()
            }

            Divider()

            HStack(spacing: 12) {
                Text("\(orchestrator.sortedResults.count) photos categorized")
                    .foregroundStyle(.secondary)
                    .font(.callout)

                Spacer()

                if commitInProgress {
                    ProgressView()
                        .controlSize(.small)
                    Text("Creating albums...")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    Button("Move All to Albums", systemImage: "folder.badge.plus") {
                        commitInProgress = true
                        Task {
                            await orchestrator.commitToAlbums()
                            commitInProgress = false
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                }
            }
            .padding()
        }
    }
}

struct SortThumbnailCell: View {
    let result: SortResult

    @State private var thumbnail: NSImage?
    @State private var isHovered = false
    private static let photoLibrary = PhotoLibraryService()

    var body: some View {
        ZStack(alignment: .bottomLeading) {
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

            Text(result.category)
                .font(.caption2.bold())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.green.opacity(0.85), in: Capsule())
                .foregroundStyle(.white)
                .padding(6)
        }
        .scaleEffect(isHovered ? 1.02 : 1.0)
        .shadow(color: .black.opacity(isHovered ? 0.2 : 0), radius: 4, y: 2)
        .animation(.easeInOut(duration: 0.15), value: isHovered)
        .onHover { hovering in isHovered = hovering }
        .task {
            do {
                thumbnail = try await Self.photoLibrary.thumbnail(for: result.asset, size: CGSize(width: 300, height: 300))
            } catch {
                // Thumbnail load failed
            }
        }
    }
}
