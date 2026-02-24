import SwiftUI
import Photos
import AVKit

struct AssetDetailView: View {
    let result: ScanResult
    @Environment(\.dismiss) private var dismiss
    @State private var fullImage: NSImage?
    @State private var playerItem: AVPlayerItem?

    private static let photoLibrary = PhotoLibraryService()

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text(result.mediaType == .video ? "Video Detail" : "Image Detail")
                    .font(.headline)
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal)
            .padding(.top)

            if result.mediaType == .video, let playerItem {
                VideoPlayer(player: AVPlayer(playerItem: playerItem))
                    .frame(minHeight: 400)
            } else if let fullImage {
                Image(nsImage: fullImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(minHeight: 400)
            } else {
                ProgressView()
                    .frame(minHeight: 400)
            }

            HStack(spacing: 24) {
                Label(result.label.capitalized, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.title3.bold())

                Text("Confidence: \(Int(result.confidence * 100))%")
                    .font(.title3)

                if let frameTime = result.flaggedFrameTime {
                    Text("Flagged at \(formatTime(frameTime))")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }

                Text(result.mediaType == .video ? "Video" : "Image")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.quaternary, in: Capsule())
            }
            .padding(.bottom)
        }
        .frame(minWidth: 700, minHeight: 600)
        .task {
            await loadContent()
        }
    }

    private func loadContent() async {
        if result.mediaType == .video {
            do {
                let avAsset = try await Self.photoLibrary.requestAVAsset(for: result.asset)
                playerItem = AVPlayerItem(asset: avAsset)
            } catch {
                // Video load failed
            }
        } else {
            do {
                let (data, _) = try await Self.photoLibrary.imageData(for: result.asset)
                fullImage = NSImage(data: data)
            } catch {
                // Image load failed
            }
        }
    }

    private func formatTime(_ seconds: Double) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}
