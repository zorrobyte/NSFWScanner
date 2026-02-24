import Photos

struct ScanResult: Identifiable, Hashable {
    let id: String
    let asset: PHAsset
    let label: String
    let confidence: Float
    let mediaType: PHAssetMediaType
    let flaggedFrameTime: Double?

    static func == (lhs: ScanResult, rhs: ScanResult) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
