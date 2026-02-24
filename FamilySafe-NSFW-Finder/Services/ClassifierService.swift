import CoreML
import Vision
import CoreGraphics
import CoreVideo
import os

private let classifierLogger = Logger(subsystem: "com.familysafe.nsfwfinder", category: "ClassifierService")

actor ClassifierService {
    private var mlModel: MLModel?
    private var vnModel: VNCoreMLModel?
    private let classLabels = ["NSFW", "SFW"]

    func loadModel() throws {
        guard mlModel == nil else { return }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        let compiledURL: URL
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let cachedModelURL = cacheDir.appendingPathComponent("NSFWClassifier.mlmodelc")

        if FileManager.default.fileExists(atPath: cachedModelURL.path) {
            compiledURL = cachedModelURL
        } else {
            guard let bundleURL = Bundle.main.url(forResource: "NSFWClassifier", withExtension: "mlmodelc")
                    ?? Bundle.main.url(forResource: "NSFWClassifier", withExtension: "mlpackage") else {
                throw ClassifierError.modelNotFound
            }

            if bundleURL.pathExtension == "mlpackage" {
                compiledURL = try MLModel.compileModel(at: bundleURL)
                try? FileManager.default.copyItem(at: compiledURL, to: cachedModelURL)
            } else {
                compiledURL = bundleURL
            }
        }

        classifierLogger.info("Loading model from \(compiledURL.lastPathComponent), computeUnits=.all")
        mlModel = try MLModel(contentsOf: compiledURL, configuration: config)
        vnModel = try VNCoreMLModel(for: mlModel!)
        classifierLogger.info("Model loaded and VNCoreMLModel created")
    }

    func classify(cgImage: CGImage, orientation: CGImagePropertyOrientation = .up) throws -> (label: String, confidence: Float) {
        guard let model = vnModel else {
            throw ClassifierError.modelNotLoaded
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation)
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill

        try handler.perform([request])

        guard let results = request.results as? [VNClassificationObservation],
              let top = results.first else {
            throw ClassifierError.noResults
        }

        return (label: top.identifier, confidence: top.confidence)
    }

    func classify(imageData: Data, orientation: CGImagePropertyOrientation) throws -> (label: String, confidence: Float) {
        guard let model = vnModel else {
            throw ClassifierError.modelNotLoaded
        }

        let handler = VNImageRequestHandler(data: imageData, orientation: orientation)
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill

        try handler.perform([request])

        guard let results = request.results as? [VNClassificationObservation],
              let top = results.first else {
            throw ClassifierError.noResults
        }

        return (label: top.identifier, confidence: top.confidence)
    }
}

enum ClassifierError: LocalizedError {
    case modelNotFound
    case modelNotLoaded
    case noResults

    var errorDescription: String? {
        switch self {
        case .modelNotFound: "NSFWClassifier model not found in app bundle."
        case .modelNotLoaded: "Model has not been loaded. Call loadModel() first."
        case .noResults: "Classification produced no results."
        }
    }
}
