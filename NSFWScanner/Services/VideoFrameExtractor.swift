import AVFoundation
import CoreMedia
import os

private let extractorLogger = Logger(subsystem: "com.zorrobyte.nsfwscanner", category: "VideoFrameExtractor")

actor VideoFrameExtractor {
    /// Max frames to extract per video to prevent long videos from stalling the pipeline.
    static let maxFramesPerVideo = 30

    func extractFrames(
        from avAsset: AVAsset,
        intervalSeconds: Double = 5.0
    ) async throws -> [(CGImage, CMTime)] {
        let duration = try await avAsset.load(.duration).seconds

        guard duration > 0 else { return [] }

        let generator = AVAssetImageGenerator(asset: avAsset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = CMTime.zero
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)

        // Calculate effective interval — widen it for long videos to stay within frame cap
        let naturalFrameCount = Int(duration / intervalSeconds) + 1
        let effectiveInterval: Double
        if naturalFrameCount > Self.maxFramesPerVideo {
            effectiveInterval = duration / Double(Self.maxFramesPerVideo)
            extractorLogger.info("Long video (\(String(format: "%.0f", duration))s, \(naturalFrameCount) frames at \(String(format: "%.1f", intervalSeconds))s) — widening interval to \(String(format: "%.1f", effectiveInterval))s (\(Self.maxFramesPerVideo) frames)")
        } else {
            effectiveInterval = intervalSeconds
        }

        var requestedTimes: [CMTime] = []
        var t = 0.0
        while t < duration {
            requestedTimes.append(CMTime(seconds: t, preferredTimescale: 600))
            t += effectiveInterval
        }

        var frames: [(CGImage, CMTime)] = []

        for time in requestedTimes {
            do {
                let (image, actualTime) = try await generator.image(at: time)
                frames.append((image, actualTime))
            } catch {
                continue
            }
        }

        return frames
    }
}
