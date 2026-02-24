import Photos
import AppKit
import os

private let photoLogger = Logger(subsystem: "com.zorrobyte.nsfwscanner", category: "PhotoLibraryService")

actor PhotoLibraryService {
    func requestAuthorization() async -> PHAuthorizationStatus {
        await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    }

    func fetchAllAssets() -> PHFetchResult<PHAsset> {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.predicate = NSPredicate(
            format: "mediaType == %d OR mediaType == %d",
            PHAssetMediaType.image.rawValue,
            PHAssetMediaType.video.rawValue
        )
        return PHAsset.fetchAssets(with: options)
    }

    /// Returns the set of asset local identifiers already in the named album.
    func albumAssetIDs(albumName: String) -> Set<String> {
        let fetchOptions = PHFetchOptions()
        fetchOptions.predicate = NSPredicate(format: "title == %@", albumName)
        let albums = PHAssetCollection.fetchAssetCollections(with: .album, subtype: .any, options: fetchOptions)

        guard let album = albums.firstObject else { return [] }

        let assetOptions = PHFetchOptions()
        let assets = PHAsset.fetchAssets(in: album, options: assetOptions)

        var ids = Set<String>()
        assets.enumerateObjects { asset, _, _ in
            ids.insert(asset.localIdentifier)
        }
        photoLogger.info("Found \(ids.count) assets already in \(albumName) album — will skip")
        return ids
    }

    func imageData(for asset: PHAsset) async throws -> (Data, CGImagePropertyOrientation) {
        try await withCheckedThrowingContinuation { continuation in
            let options = PHImageRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            options.isSynchronous = false

            PHImageManager.default().requestImageDataAndOrientation(
                for: asset,
                options: options
            ) { data, _, orientation, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                } else if let data {
                    continuation.resume(returning: (data, orientation))
                } else {
                    continuation.resume(throwing: PhotoLibraryError.noImageData)
                }
            }
        }
    }

    /// Fetch a downsized CGImage for classification with a timeout to skip stuck iCloud downloads.
    func classificationImage(for asset: PHAsset, targetSize: CGSize) async throws -> CGImage {
        try await withThrowingTaskGroup(of: CGImage.self) { group in
            group.addTask {
                try await withCheckedThrowingContinuation { continuation in
                    let options = PHImageRequestOptions()
                    options.isNetworkAccessAllowed = true
                    options.deliveryMode = .highQualityFormat
                    options.resizeMode = .exact
                    options.isSynchronous = false

                    PHImageManager.default().requestImage(
                        for: asset,
                        targetSize: targetSize,
                        contentMode: .aspectFill,
                        options: options
                    ) { image, info in
                        if let error = info?[PHImageErrorKey] as? Error {
                            continuation.resume(throwing: error)
                            return
                        }
                        guard let image,
                              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                            continuation.resume(throwing: PhotoLibraryError.noImageData)
                            return
                        }
                        continuation.resume(returning: cgImage)
                    }
                }
            }

            group.addTask {
                try await Task.sleep(for: .seconds(15))
                throw PhotoLibraryError.timeout
            }

            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }

    /// Request AVAsset with timeout.
    func requestAVAsset(for asset: PHAsset) async throws -> AVAsset {
        try await withThrowingTaskGroup(of: AVAsset.self) { group in
            group.addTask {
                try await withCheckedThrowingContinuation { continuation in
                    let options = PHVideoRequestOptions()
                    options.isNetworkAccessAllowed = true
                    options.deliveryMode = .highQualityFormat

                    PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, info in
                        if let error = info?[PHImageErrorKey] as? Error {
                            continuation.resume(throwing: error)
                        } else if let avAsset {
                            continuation.resume(returning: avAsset)
                        } else {
                            continuation.resume(throwing: PhotoLibraryError.noVideoAsset)
                        }
                    }
                }
            }

            group.addTask {
                try await Task.sleep(for: .seconds(30))
                throw PhotoLibraryError.timeout
            }

            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }

    func thumbnail(for asset: PHAsset, size: CGSize) async throws -> NSImage {
        try await withCheckedThrowingContinuation { continuation in
            let options = PHImageRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            options.resizeMode = .fast
            options.isSynchronous = false

            PHImageManager.default().requestImage(
                for: asset,
                targetSize: size,
                contentMode: .aspectFill,
                options: options
            ) { image, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                } else if let image {
                    continuation.resume(returning: image)
                } else {
                    continuation.resume(throwing: PhotoLibraryError.noImageData)
                }
            }
        }
    }

    func createAlbumIfNeeded(named albumName: String) async throws -> PHAssetCollection {
        let fetchOptions = PHFetchOptions()
        fetchOptions.predicate = NSPredicate(format: "title == %@", albumName)
        let existing = PHAssetCollection.fetchAssetCollections(with: .album, subtype: .any, options: fetchOptions)

        if let album = existing.firstObject {
            return album
        }

        var placeholder: PHObjectPlaceholder?
        try await PHPhotoLibrary.shared().performChanges {
            let request = PHAssetCollectionChangeRequest.creationRequestForAssetCollection(withTitle: albumName)
            placeholder = request.placeholderForCreatedAssetCollection
        }

        guard let localID = placeholder?.localIdentifier,
              let album = PHAssetCollection.fetchAssetCollections(
                withLocalIdentifiers: [localID], options: nil
              ).firstObject else {
            throw PhotoLibraryError.albumCreationFailed
        }

        return album
    }

    func hideAssets(_ assets: [PHAsset]) async throws {
        try await PHPhotoLibrary.shared().performChanges {
            for asset in assets {
                let request = PHAssetChangeRequest(for: asset)
                request.isHidden = true
            }
        }
        photoLogger.info("Hidden \(assets.count) assets")
    }

    func addAssets(_ assets: [PHAsset], toAlbum album: PHAssetCollection) async throws {
        try await PHPhotoLibrary.shared().performChanges {
            guard let request = PHAssetCollectionChangeRequest(for: album) else { return }
            request.addAssets(assets as NSFastEnumeration)
        }
    }
}

enum PhotoLibraryError: LocalizedError {
    case noImageData
    case noVideoAsset
    case albumCreationFailed
    case timeout

    var errorDescription: String? {
        switch self {
        case .noImageData: "Failed to retrieve image data from Photos library."
        case .noVideoAsset: "Failed to retrieve video asset from Photos library."
        case .albumCreationFailed: "Failed to create NSFW album."
        case .timeout: "Request timed out (asset may be downloading from iCloud)."
        }
    }
}
