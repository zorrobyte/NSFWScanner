import Photos

struct SortResult: Identifiable, Hashable {
    let id: String
    let asset: PHAsset
    let category: String
    let mediaType: PHAssetMediaType

    static func == (lhs: SortResult, rhs: SortResult) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
