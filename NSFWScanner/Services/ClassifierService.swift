import CoreML
import Vision
import CoreGraphics
import os

private let classifierLogger = Logger(subsystem: "com.zorrobyte.nsfwscanner", category: "ClassifierService")

actor ClassifierService {
    private var vnModel: VNCoreMLModel?
    private var currentModel: NSFWModel?

    func loadModel(_ model: NSFWModel) throws {
        guard model != currentModel else { return }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let cachedModelURL = cacheDir.appendingPathComponent("\(model.resourceName).mlmodelc")

        let compiledURL: URL
        if FileManager.default.fileExists(atPath: cachedModelURL.path) {
            compiledURL = cachedModelURL
            classifierLogger.info("Using cached model: \(model.resourceName)")
        } else {
            guard let bundleURL = Bundle.main.url(forResource: model.resourceName, withExtension: "mlmodelc")
                    ?? Bundle.main.url(forResource: model.resourceName, withExtension: "mlpackage") else {
                throw ClassifierError.modelNotFound
            }

            if bundleURL.pathExtension == "mlpackage" {
                classifierLogger.info("Compiling model: \(model.resourceName)")
                compiledURL = try MLModel.compileModel(at: bundleURL)
                try? FileManager.default.copyItem(at: compiledURL, to: cachedModelURL)
            } else {
                compiledURL = bundleURL
            }
        }

        classifierLogger.info("Loading \(model.displayName) from \(compiledURL.lastPathComponent)")
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: config)
        vnModel = try VNCoreMLModel(for: mlModel)
        currentModel = model
        classifierLogger.info("Model \(model.displayName) loaded successfully")
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
        case .modelNotFound: "ML model not found in app bundle."
        case .modelNotLoaded: "Model has not been loaded. Call loadModel() first."
        case .noResults: "Classification produced no results."
        }
    }
}
