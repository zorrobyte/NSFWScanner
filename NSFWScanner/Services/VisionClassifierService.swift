import Foundation
import AppKit
import os

private let logger = Logger(subsystem: "com.zorrobyte.nsfwscanner", category: "VisionClassifier")

actor VisionClassifierService {
    private let session: URLSession

    init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 120
        self.session = URLSession(configuration: config)
    }

    /// Classify an image into exactly one of the given categories using the remote vision model.
    func classify(
        imageData: Data,
        categories: [String],
        endpointURL: URL
    ) async throws -> String {
        let base64 = imageData.base64EncodedString()
        let categoryList = categories.joined(separator: ", ")

        let prompt = "Classify this image into exactly ONE category from this list: \(categoryList). Reply with ONLY the category name, nothing else."

        let requestBody: [String: Any] = [
            "model": "qwen3-vl-8b",
            "messages": [
                ["role": "system", "content": "You are a photo classifier. Respond with only one category name. /no_think"],
                ["role": "user", "content": [
                    ["type": "text", "text": prompt],
                    ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(base64)"]]
                ] as [[String: Any]]]
            ] as [[String: Any]],
            "max_tokens": 20,
            "temperature": 0
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: requestBody)

        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw VisionClassifierError.invalidResponse
        }

        guard httpResponse.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            logger.error("Vision API returned \(httpResponse.statusCode): \(body)")
            throw VisionClassifierError.serverError(httpResponse.statusCode, body)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw VisionClassifierError.parseError
        }

        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)

        // Match to a known category
        let matched = matchOne(response: trimmed, categories: categories)
        logger.debug("Vision classified as: '\(trimmed)' → matched: \(matched)")
        return matched
    }

    /// Encode an NSImage as JPEG data at the given quality.
    nonisolated func encodeJPEG(image: NSImage, quality: CGFloat = 0.7) -> Data? {
        guard let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff) else { return nil }
        return rep.representation(using: .jpeg, properties: [.compressionFactor: quality])
    }

    /// Match a response string to the best known category.
    private func matchOne(response: String, categories: [String]) -> String {
        let lower = response.lowercased()
        let categoryLower = categories.map { ($0, $0.lowercased()) }

        // Exact match
        if let found = categoryLower.first(where: { $0.1 == lower }) {
            return found.0
        }

        // Contains match (either direction)
        if let found = categoryLower.first(where: { lower.contains($0.1) || $0.1.contains(lower) }) {
            return found.0
        }

        // Fall back to "Other"
        return categories.first(where: { $0 == "Other" }) ?? categories.last ?? "Other"
    }
}

enum VisionClassifierError: LocalizedError {
    case invalidResponse
    case serverError(Int, String)
    case parseError
    case encodingFailed

    var errorDescription: String? {
        switch self {
        case .invalidResponse: "Invalid response from vision model."
        case .serverError(let code, let body): "Vision model returned HTTP \(code): \(body)"
        case .parseError: "Failed to parse vision model response."
        case .encodingFailed: "Failed to encode image as JPEG."
        }
    }
}
