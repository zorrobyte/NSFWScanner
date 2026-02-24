import Foundation

enum NSFWModel: String, CaseIterable, Identifiable, Codable {
    case marqo = "marqo-vit-tiny"
    case falconsai = "falconsai-vit-base"
    case adamcodd = "adamcodd-vit-base"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .marqo: "Marqo ViT-Tiny"
        case .falconsai: "Falconsai ViT-Base"
        case .adamcodd: "AdamCodd ViT-Base"
        }
    }

    var subtitle: String {
        switch self {
        case .marqo: "Fast & accurate (384x384, ~11 MB)"
        case .falconsai: "Most popular (224x224, ~164 MB)"
        case .adamcodd: "Strong on drawings (384x384, ~164 MB)"
        }
    }

    /// The filename of the .mlpackage in the app bundle (without extension).
    var resourceName: String {
        switch self {
        case .marqo: "NSFWClassifier"
        case .falconsai: "FalconsaiNSFW"
        case .adamcodd: "AdamCoddNSFW"
        }
    }

    /// Labels that indicate NSFW content for this model (lowercased).
    var defaultNSFWLabels: Set<String> {
        switch self {
        case .marqo: ["nsfw"]
        case .falconsai: ["nsfw"]
        case .adamcodd: ["nsfw"]
        }
    }
}
