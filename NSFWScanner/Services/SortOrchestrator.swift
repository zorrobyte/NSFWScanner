import Photos
import Observation
import AppKit
import os

private let logger = Logger(subsystem: "com.zorrobyte.nsfwscanner", category: "SortOrchestrator")

@Observable
final class SortOrchestrator {
    // State
    var state: ScanState = .idle

    // Progress
    var totalPhotos: Int = 0
    var processedPhotos: Int = 0
    var committedPhotos: Int = 0
    var photoProgress: Double = 0

    // Results grouped by category
    var sortedResults: [SortResult] = []

    // UI log buffer (most recent entries shown in the UI)
    var logEntries: [String] = []
    private let maxLogEntries = 200

    // Settings
    var categories: [SortCategory] = SortCategory.defaults
    var albumPrefix: String = ""
    var endpointURL: String = "http://100.83.43.98:1235/v1/chat/completions"
    var concurrency: Double = 8

    private let visionClassifier = VisionClassifierService()
    private let photoLibrary = PhotoLibraryService()
    private var sortTask: Task<Void, Never>?

    // Auto-commit buffer: category name → pending assets
    private var pendingCommit: [String: [PHAsset]] = [:]
    private let commitBatchSize = 25

    // Pre-created album cache: album name → PHAssetCollection
    private var albumCache: [String: PHAssetCollection] = [:]

    /// Category counts for display.
    var categoryCounts: [(category: String, count: Int)] {
        var counts: [String: Int] = [:]
        for result in sortedResults {
            counts[result.category, default: 0] += 1
        }
        return categories.map { (category: $0.name, count: counts[$0.name, default: 0]) }
            .filter { $0.count > 0 }
            .sorted { $0.count > $1.count }
    }

    init() {
        let defaults = UserDefaults.standard
        if let saved = defaults.string(forKey: "sort_endpointURL"), !saved.isEmpty {
            endpointURL = saved
        }
        if let saved = defaults.string(forKey: "sort_albumPrefix") {
            albumPrefix = saved
        }
        if let saved = defaults.object(forKey: "sort_concurrency") as? Double, saved > 0 {
            concurrency = saved
        }
        if let data = defaults.data(forKey: "sort_categories"),
           let saved = try? JSONDecoder().decode([SortCategory].self, from: data), !saved.isEmpty {
            categories = saved
        }
    }

    func saveSettings() {
        let defaults = UserDefaults.standard
        defaults.set(endpointURL, forKey: "sort_endpointURL")
        defaults.set(albumPrefix, forKey: "sort_albumPrefix")
        defaults.set(concurrency, forKey: "sort_concurrency")
        if let data = try? JSONEncoder().encode(categories) {
            defaults.set(data, forKey: "sort_categories")
        }
    }

    func startSort() {
        guard state == .idle || state == .reviewing else { return }

        saveSettings()

        sortedResults = []
        logEntries = []
        pendingCommit = [:]
        albumCache = [:]
        committedPhotos = 0
        totalPhotos = 0
        processedPhotos = 0
        photoProgress = 0
        state = .requestingPermission
        appendLog("Requesting Photos authorization...")

        sortTask = Task {
            do {
                let status = await photoLibrary.requestAuthorization()
                guard status == .authorized || status == .limited else {
                    appendLog("ERROR: Photos access denied")
                    state = .error("Photos access denied. Please grant access in System Settings.")
                    return
                }
                appendLog("Photos access granted")

                state = .scanning

                guard let url = URL(string: endpointURL) else {
                    appendLog("ERROR: Invalid endpoint URL: \(endpointURL)")
                    state = .error("Invalid endpoint URL: \(endpointURL)")
                    return
                }

                appendLog("Fetching photo library...")
                logger.info("Fetching all photo assets...")
                let fetchResult = await photoLibrary.fetchAllAssets()

                appendLog("Checking existing sorted albums...")
                let excludedIDs = await allSortedAlbumAssetIDs()

                var photoAssets: [PHAsset] = []
                fetchResult.enumerateObjects { asset, _, _ in
                    guard !excludedIDs.contains(asset.localIdentifier) else { return }
                    if asset.mediaType == .image {
                        photoAssets.append(asset)
                    }
                }

                totalPhotos = photoAssets.count
                appendLog("Found \(totalPhotos) photos to sort (skipped \(excludedIDs.count) already sorted)")
                appendLog("Endpoint: \(endpointURL)")
                appendLog("Concurrency: \(Int(concurrency)) parallel requests")
                appendLog("Auto-commit: every \(commitBatchSize) photos per category")
                logger.info("Found \(self.totalPhotos) photos to sort (skipped \(excludedIDs.count) already sorted)")

                guard totalPhotos > 0 else {
                    state = .reviewing
                    return
                }

                let categoryNames = categories.map(\.name)
                let maxConcurrent = Int(concurrency)
                let classificationSize = CGSize(width: 384, height: 384)

                await withTaskGroup(of: SortResult?.self) { group in
                    var inFlight = 0
                    var index = 0

                    while index < photoAssets.count || inFlight > 0 {
                        while inFlight < maxConcurrent, index < photoAssets.count {
                            let asset = photoAssets[index]
                            index += 1
                            inFlight += 1

                            group.addTask { [visionClassifier = self.visionClassifier, photoLibrary = self.photoLibrary] in
                                do {
                                    let nsImage = try await photoLibrary.thumbnail(for: asset, size: classificationSize)

                                    guard let jpegData = visionClassifier.encodeJPEG(image: nsImage, quality: 0.7) else {
                                        logger.warning("Failed to encode \(asset.localIdentifier.prefix(8)) as JPEG")
                                        return nil
                                    }

                                    let matched = try await visionClassifier.classify(
                                        imageData: jpegData,
                                        categories: categoryNames,
                                        endpointURL: url
                                    )

                                    return SortResult(
                                        id: asset.localIdentifier,
                                        asset: asset,
                                        category: matched,
                                        mediaType: .image
                                    )
                                } catch {
                                    logger.warning("Sort failed for \(asset.localIdentifier.prefix(8)): \(error.localizedDescription)")
                                    await MainActor.run {
                                        self.appendLog("FAIL: \(error.localizedDescription)")
                                    }
                                    return nil
                                }
                            }
                        }

                        if let result = await group.next() {
                            inFlight -= 1
                            await MainActor.run {
                                self.processedPhotos += 1
                                self.photoProgress = self.totalPhotos > 0
                                    ? Double(self.processedPhotos) / Double(self.totalPhotos)
                                    : 1

                                if let sortResult = result {
                                    self.sortedResults.append(sortResult)
                                    self.pendingCommit[sortResult.category, default: []].append(sortResult.asset)
                                    self.appendLog("\(sortResult.category) — photo \(self.processedPhotos)/\(self.totalPhotos)")
                                }

                                if self.processedPhotos % 50 == 0 || self.processedPhotos == self.totalPhotos {
                                    let pct = Int(self.photoProgress * 100)
                                    logger.info("Sorted: \(self.processedPhotos)/\(self.totalPhotos) (\(pct)%)")
                                    self.appendLog("Progress: \(self.processedPhotos)/\(self.totalPhotos) (\(pct)%)")
                                }
                            }

                            // Auto-commit any categories that hit the batch size
                            await flushReadyBatches()
                        }
                    }
                }

                // Flush any remaining photos
                await flushAllPending()

                appendLog("Sort complete! \(self.processedPhotos) processed, \(self.committedPhotos) moved to albums.")
                logger.info("Sort complete. \(self.processedPhotos) processed, \(self.committedPhotos) committed.")
                if !Task.isCancelled {
                    state = .reviewing
                }
            } catch {
                appendLog("ERROR: \(error.localizedDescription)")
                logger.error("Sort failed: \(error.localizedDescription)")
                if !Task.isCancelled {
                    state = .error(error.localizedDescription)
                }
            }
        }
    }

    func cancelSort() {
        sortTask?.cancel()
        sortTask = nil
        // Flush whatever we have so far before stopping
        Task {
            await flushAllPending()
            state = .idle
        }
    }

    func commitToAlbums() async {
        state = .committingToAlbum
        await flushAllPending()
        sortedResults.removeAll()
        state = .reviewing
    }

    func dismissResults(ids: Set<String>) {
        sortedResults.removeAll { ids.contains($0.id) }
    }

    func resetToIdle() {
        state = .idle
        logEntries.removeAll()
    }

    func appendLog(_ message: String) {
        let timestamp = Self.timeFormatter.string(from: Date())
        let entry = "[\(timestamp)] \(message)"
        logEntries.append(entry)
        if logEntries.count > maxLogEntries {
            logEntries.removeFirst(logEntries.count - maxLogEntries)
        }
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()

    // MARK: - Auto-commit

    /// Flush any category buffers that have reached the batch size.
    private func flushReadyBatches() async {
        for (categoryName, assets) in pendingCommit {
            if assets.count >= commitBatchSize {
                await commitBatch(categoryName: categoryName, assets: assets)
                pendingCommit[categoryName] = []
            }
        }
    }

    /// Flush all remaining pending photos to albums.
    private func flushAllPending() async {
        for (categoryName, assets) in pendingCommit {
            guard !assets.isEmpty else { continue }
            await commitBatch(categoryName: categoryName, assets: assets)
        }
        pendingCommit = [:]
    }

    /// Commit a batch of photos to a specific category album.
    private func commitBatch(categoryName: String, assets: [PHAsset]) async {
        let category = categories.first(where: { $0.name == categoryName }) ?? SortCategory(name: categoryName)
        let albumName = category.albumName(prefix: albumPrefix)

        do {
            let album: PHAssetCollection
            if let cached = albumCache[albumName] {
                album = cached
            } else {
                album = try await photoLibrary.createAlbumIfNeeded(named: albumName)
                albumCache[albumName] = album
            }

            try await photoLibrary.addAssets(assets, toAlbum: album)
            committedPhotos += assets.count
            appendLog("Moved \(assets.count) → \(albumName) (\(committedPhotos) total)")
            logger.info("Committed \(assets.count) to '\(albumName)' (\(self.committedPhotos) total)")
        } catch {
            appendLog("ERROR committing to \(albumName): \(error.localizedDescription)")
            logger.error("Failed to commit to '\(albumName)': \(error.localizedDescription)")
        }
    }

    // MARK: - Private

    /// Collect asset IDs from all albums matching the prefix pattern.
    private func allSortedAlbumAssetIDs() async -> Set<String> {
        var ids = Set<String>()
        for category in categories {
            let albumName = category.albumName(prefix: albumPrefix)
            let albumIDs = await photoLibrary.albumAssetIDs(albumName: albumName)
            ids.formUnion(albumIDs)
        }
        return ids
    }
}
