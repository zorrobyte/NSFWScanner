import Foundation

struct SortCategory: Identifiable, Hashable, Codable {
    let name: String

    var id: String { name }

    /// Album name with optional prefix, e.g. "Sorted - Landscapes"
    func albumName(prefix: String) -> String {
        prefix.isEmpty ? name : "\(prefix)\(name)"
    }

    static let defaults: [SortCategory] = [
        SortCategory(name: "People & Portraits"),
        SortCategory(name: "Family"),
        SortCategory(name: "Pets & Animals"),
        SortCategory(name: "Food & Drinks"),
        SortCategory(name: "Nature & Landscapes"),
        SortCategory(name: "Architecture & Urban"),
        SortCategory(name: "Travel"),
        SortCategory(name: "Tech & Setups"),
        SortCategory(name: "Gaming"),
        SortCategory(name: "Sports & Action"),
        SortCategory(name: "Screenshots"),
        SortCategory(name: "Documents & Receipts"),
        SortCategory(name: "Night & Lights"),
        SortCategory(name: "Memes"),
        SortCategory(name: "Other"),
    ]
}
