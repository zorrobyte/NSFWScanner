import Photos
import Observation
import os

private let logger = Logger(subsystem: "com.zorrobyte.nsfwscanner", category: "ScanOrchestrator")

@Observable
final class ScanOrchestrator {
    var state: ScanState = .idle

    // Image queue progress
    var totalImages: Int = 0
    var processedImages: Int = 0
    var imageProgress: Double = 0
    var flaggedImageCount: Int = 0

    // Video queue progress
    var totalVideos: Int = 0
    var processedVideos: Int = 0
    var videoProgress: Double = 0
    var flaggedVideoCount: Int = 0

    // Scan options
    var scanImages: Bool = true
    var scanVideos: Bool = true

    // Combined
    var flaggedResults: [ScanResult] = []
    var confidenceThreshold: Float = 0.85
    var frameInterval: Double = 5.0
    var imageConcurrency: Double = 20
    var videoConcurrency: Double = 16

    var totalAssets: Int { totalImages + totalVideos }
    var processedCount: Int { processedImages + processedVideos }
    var progress: Double {
        guard totalAssets > 0 else { return 0 }
        return Double(processedCount) / Double(totalAssets)
    }

    private let classifier = ClassifierService()
    private let photoLibrary = PhotoLibraryService()
    private let frameExtractor = VideoFrameExtractor()
    private var scanTask: Task<Void, Never>?

    func startScan() {
        guard state == .idle || state == .reviewing else { return }

        flaggedResults = []
        flaggedImageCount = 0
        flaggedVideoCount = 0
        totalImages = 0
        totalVideos = 0
        processedImages = 0
        processedVideos = 0
        imageProgress = 0
        videoProgress = 0
        state = .requestingPermission

        scanTask = Task {
            do {
                logger.info("Requesting Photos authorization...")
                let status = await photoLibrary.requestAuthorization()
                logger.info("Authorization status: \(String(describing: status))")
                guard status == .authorized || status == .limited else {
                    state = .error("Photos access denied. Please grant access in System Settings.")
                    return
                }

                state = .scanning

                logger.info("Loading ML model...")
                try await classifier.loadModel()
                logger.info("Model loaded successfully")

                logger.info("Fetching all assets...")
                let fetchResult = await photoLibrary.fetchAllAssets()
                let excludedIDs = await photoLibrary.nsfwAlbumAssetIDs()

                let wantImages = scanImages
                let wantVideos = scanVideos

                var imageAssets: [PHAsset] = []
                var videoAssets: [PHAsset] = []
                fetchResult.enumerateObjects { asset, _, _ in
                    guard !excludedIDs.contains(asset.localIdentifier) else { return }
                    if wantImages && asset.mediaType == .image {
                        imageAssets.append(asset)
                    } else if wantVideos && asset.mediaType == .video {
                        videoAssets.append(asset)
                    }
                }

                totalImages = imageAssets.count
                totalVideos = videoAssets.count
                logger.info("Found \(self.totalImages) images and \(self.totalVideos) videos to scan (skipped \(excludedIDs.count) in NSFW album)")

                guard totalAssets > 0 else {
                    state = .reviewing
                    return
                }

                let threshold = confidenceThreshold
                let interval = frameInterval
                let maxImages = Int(imageConcurrency)
                let maxVideos = Int(videoConcurrency)

                // Run image and video queues in parallel
                await withTaskGroup(of: Void.self) { outerGroup in
                    // Image queue — high concurrency (images are fast)
                    if !imageAssets.isEmpty { outerGroup.addTask {
                        await withTaskGroup(of: ScanResult?.self) { group in
                            var inFlight = 0
                            var index = 0

                            while index < imageAssets.count || inFlight > 0 {
                                while inFlight < maxImages, index < imageAssets.count {
                                    let asset = imageAssets[index]
                                    index += 1
                                    inFlight += 1

                                    group.addTask { [classifier = self.classifier, photoLibrary = self.photoLibrary] in
                                        do {
                                            return try await Self.classifyImage(
                                                asset: asset,
                                                classifier: classifier,
                                                photoLibrary: photoLibrary,
                                                threshold: threshold
                                            )
                                        } catch {
                                            logger.warning("Image \(asset.localIdentifier) failed: \(error.localizedDescription)")
                                            return nil
                                        }
                                    }
                                }

                                if let result = await group.next() {
                                    inFlight -= 1
                                    await MainActor.run {
                                        self.processedImages += 1
                                        self.imageProgress = self.totalImages > 0 ? Double(self.processedImages) / Double(self.totalImages) : 1

                                        if self.processedImages % 100 == 0 || self.processedImages == self.totalImages {
                                            logger.info("Images: \(self.processedImages)/\(self.totalImages) (\(Int(self.imageProgress * 100))%) — \(self.flaggedImageCount) flagged")
                                        }

                                        if let scanResult = result {
                                            logger.notice("FLAGGED image: \(scanResult.id) — \(scanResult.label) \(Int(scanResult.confidence * 100))%")
                                            self.flaggedResults.append(scanResult)
                                            self.flaggedImageCount += 1
                                        }
                                    }
                                }
                            }
                        }
                        logger.info("Image queue complete: \(self.processedImages) processed, \(self.flaggedImageCount) flagged")
                    }}

                    // Video queue — high concurrency for M4 Max
                    if !videoAssets.isEmpty { outerGroup.addTask {
                        await withTaskGroup(of: ScanResult?.self) { group in
                            var inFlight = 0
                            var index = 0

                            while index < videoAssets.count || inFlight > 0 {
                                while inFlight < maxVideos, index < videoAssets.count {
                                    let asset = videoAssets[index]
                                    index += 1
                                    inFlight += 1

                                    group.addTask { [classifier = self.classifier, photoLibrary = self.photoLibrary, frameExtractor = self.frameExtractor] in
                                        do {
                                            return try await Self.classifyVideo(
                                                asset: asset,
                                                classifier: classifier,
                                                photoLibrary: photoLibrary,
                                                frameExtractor: frameExtractor,
                                                threshold: threshold,
                                                frameInterval: interval
                                            )
                                        } catch {
                                            logger.warning("Video \(asset.localIdentifier) failed: \(error.localizedDescription)")
                                            return nil
                                        }
                                    }
                                }

                                if let result = await group.next() {
                                    inFlight -= 1
                                    await MainActor.run {
                                        self.processedVideos += 1
                                        self.videoProgress = self.totalVideos > 0 ? Double(self.processedVideos) / Double(self.totalVideos) : 1

                                        if self.processedVideos % 20 == 0 || self.processedVideos == self.totalVideos {
                                            logger.info("Videos: \(self.processedVideos)/\(self.totalVideos) (\(Int(self.videoProgress * 100))%) — \(self.flaggedVideoCount) flagged")
                                        }

                                        if let scanResult = result {
                                            logger.notice("FLAGGED video: \(scanResult.id) — \(scanResult.label) \(Int(scanResult.confidence * 100))%")
                                            self.flaggedResults.append(scanResult)
                                            self.flaggedVideoCount += 1
                                        }
                                    }
                                }
                            }
                        }
                        logger.info("Video queue complete: \(self.processedVideos) processed, \(self.flaggedVideoCount) flagged")
                    }}
                }

                logger.info("Scan complete. \(self.processedCount) processed, \(self.flaggedResults.count) flagged.")
                if !Task.isCancelled {
                    state = .reviewing
                }
            } catch {
                logger.error("Scan failed: \(error.localizedDescription)")
                if !Task.isCancelled {
                    state = .error(error.localizedDescription)
                }
            }
        }
    }

    func cancelScan() {
        scanTask?.cancel()
        scanTask = nil
        state = .idle
    }

    func commitFlaggedToAlbum(selectedIDs: Set<String>) async {
        state = .committingToAlbum
        do {
            let assetsToAdd = flaggedResults.filter { selectedIDs.contains($0.id) }.map(\.asset)
            logger.info("Committing \(assetsToAdd.count) assets to NSFW album...")
            guard !assetsToAdd.isEmpty else {
                logger.warning("No assets to commit")
                state = .reviewing
                return
            }

            let album = try await photoLibrary.createNSFWAlbumIfNeeded()
            logger.info("Album ready: \(album.localizedTitle ?? "NSFW")")
            try await photoLibrary.addAssets(assetsToAdd, toAlbum: album)
            logger.info("Successfully added \(assetsToAdd.count) assets to album")

            // Remove committed items from the results
            flaggedResults.removeAll { selectedIDs.contains($0.id) }
            flaggedImageCount = flaggedResults.filter { $0.mediaType == .image }.count
            flaggedVideoCount = flaggedResults.filter { $0.mediaType == .video }.count
            state = .reviewing
        } catch {
            logger.error("Album commit failed: \(error.localizedDescription)")
            state = .error(error.localizedDescription)
        }
    }

    func dismissResults(ids: Set<String>) {
        flaggedResults.removeAll { ids.contains($0.id) }
        let removedImages = ids.count // approximate — recalculate from remaining
        flaggedImageCount = flaggedResults.filter { $0.mediaType == .image }.count
        flaggedVideoCount = flaggedResults.filter { $0.mediaType == .video }.count
        logger.info("Dismissed \(ids.count) results, \(self.flaggedResults.count) remaining")
    }

    func resetToIdle() {
        state = .idle
    }

    // MARK: - Private classification helpers

    private static let classificationSize = CGSize(width: 384, height: 384)

    nonisolated private static func classifyImage(
        asset: PHAsset,
        classifier: ClassifierService,
        photoLibrary: PhotoLibraryService,
        threshold: Float
    ) async throws -> ScanResult? {
        let fetchStart = CFAbsoluteTimeGetCurrent()
        let cgImage = try await photoLibrary.classificationImage(for: asset, targetSize: classificationSize)
        let fetchTime = CFAbsoluteTimeGetCurrent() - fetchStart

        let classifyStart = CFAbsoluteTimeGetCurrent()
        let result = try await classifier.classify(cgImage: cgImage)
        let classifyTime = CFAbsoluteTimeGetCurrent() - classifyStart

        logger.debug("Image \(asset.localIdentifier.prefix(8)): fetch=\(String(format: "%.0f", fetchTime * 1000))ms classify=\(String(format: "%.0f", classifyTime * 1000))ms → \(result.label) \(Int(result.confidence * 100))%")

        if result.label == "NSFW" && result.confidence >= threshold {
            return ScanResult(
                id: asset.localIdentifier,
                asset: asset,
                label: result.label,
                confidence: result.confidence,
                mediaType: .image,
                flaggedFrameTime: nil
            )
        }
        return nil
    }

    nonisolated private static func classifyVideo(
        asset: PHAsset,
        classifier: ClassifierService,
        photoLibrary: PhotoLibraryService,
        frameExtractor: VideoFrameExtractor,
        threshold: Float,
        frameInterval: Double
    ) async throws -> ScanResult? {
        let videoStart = CFAbsoluteTimeGetCurrent()
        let avAsset = try await photoLibrary.requestAVAsset(for: asset)
        let fetchTime = CFAbsoluteTimeGetCurrent() - videoStart

        let extractStart = CFAbsoluteTimeGetCurrent()
        let frames = try await frameExtractor.extractFrames(from: avAsset, intervalSeconds: frameInterval)
        let extractTime = CFAbsoluteTimeGetCurrent() - extractStart

        logger.debug("Video \(asset.localIdentifier.prefix(8)): fetch=\(String(format: "%.0f", fetchTime * 1000))ms extract=\(String(format: "%.0f", extractTime * 1000))ms frames=\(frames.count)")

        var worstConfidence: Float = 0
        var worstFrameTime: Double = 0

        for (cgImage, time) in frames {
            let result = try await classifier.classify(cgImage: cgImage)
            if result.label == "NSFW" && result.confidence > worstConfidence {
                worstConfidence = result.confidence
                worstFrameTime = time.seconds
            }
        }

        if worstConfidence >= threshold {
            return ScanResult(
                id: asset.localIdentifier,
                asset: asset,
                label: "NSFW",
                confidence: worstConfidence,
                mediaType: .video,
                flaggedFrameTime: worstFrameTime
            )
        }
        return nil
    }
}
