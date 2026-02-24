import Foundation

enum NSFWModel: String, CaseIterable, Identifiable, Codable {
    case marqo = "marqo-vit-tiny"
    case falconsai = "falconsai-vit-base"
    case viddexa = "viddexa-efficientnet"
    case adamcodd = "adamcodd-vit-base"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .marqo: "Marqo ViT-Tiny"
        case .falconsai: "Falconsai ViT-Base"
        case .viddexa: "Viddexa EfficientNet (5-class)"
        case .adamcodd: "AdamCodd ViT-Base"
        }
    }

    var subtitle: String {
        switch self {
        case .marqo: "Fast & accurate (384x384, ~11 MB)"
        case .falconsai: "Most popular (224x224, ~164 MB)"
        case .viddexa: "Granular categories (380x380, ~34 MB)"
        case .adamcodd: "Strong on drawings (384x384, ~164 MB)"
        }
    }

    /// The filename of the .mlpackage in the app bundle (without extension).
    var resourceName: String {
        switch self {
        case .marqo: "NSFWClassifier"
        case .falconsai: "FalconsaiNSFW"
        case .viddexa: "ViddexaNSFW"
        case .adamcodd: "AdamCoddNSFW"
        }
    }

    /// Labels that indicate NSFW content for this model (lowercased).
    /// For binary models this is the single NSFW label.
    /// For viddexa 5-class, these are the default flagged categories.
    var defaultNSFWLabels: Set<String> {
        switch self {
        case .marqo: ["nsfw"]
        case .falconsai: ["nsfw"]
        case .viddexa: ["hentai", "porn", "sexy"]
        case .adamcodd: ["nsfw"]
        }
    }

    /// Whether this model supports granular category selection.
    var hasCategories: Bool { self == .viddexa }

    /// All selectable categories for the 5-class model.
    static let viddexaCategories = ["hentai", "porn", "sexy", "drawing"]
}
