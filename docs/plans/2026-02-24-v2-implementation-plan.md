# NSFWScanner v2.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 3 new bundled ML models with a model picker, category toggles for the 5-class model, Hidden album support, configurable album name, persisted settings, and polished native macOS UI.

**Architecture:** Multi-model ClassifierService with an NSFWModel enum, parameterized PhotoLibraryService for album name + hidden album, @AppStorage for persistence, and toolbar-driven macOS native UI. Model conversion happens in Python with verification scripts.

**Tech Stack:** Swift/SwiftUI, CoreML, Vision, Photos framework, Python (coremltools, transformers, timm, torch)

---

## Phase 1: Model Conversion

### Task 1: Create conversion script for Falconsai/nsfw_image_detection

**Files:**
- Create: `model_conversion/convert_falconsai.py`

**Step 1: Write the conversion script**

```python
#!/usr/bin/env python3
"""Convert Falconsai/nsfw_image_detection (ViT-Base-224) to CoreML .mlpackage."""

import sys
from pathlib import Path

import coremltools as ct
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np


def main():
    output_path = Path(__file__).parent.parent / "NSFWScanner" / "Resources" / "FalconsaiNSFW.mlpackage"

    # Step 1: Load model from HuggingFace
    print("1/5  Loading Falconsai/nsfw_image_detection...")
    model = ViTForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
    model.eval()
    print(f"     Labels: {model.config.id2label}")

    # Step 2: Trace with fixed-shape input
    print("2/5  Tracing model...")
    # Falconsai uses standard ViT preprocessing: ImageNet normalization
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    example_input = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    print("     Trace successful")

    # Step 3: Define CoreML input with baked-in normalization
    # ImageNet: pixel/255 normalized with mean/std
    # CoreML: pixel * scale + bias per channel
    # scale = 1/(255*std), bias = -mean/std
    print("3/5  Defining CoreML input with baked-in normalization...")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    scale = 1.0 / 255.0  # We'll handle per-channel in bias
    # Actually for per-channel normalization with CoreML ImageType:
    # CoreML applies: (pixel * scale) + bias uniformly, then the model sees it
    # But ImageType only supports a single scale. For proper ImageNet normalization
    # we need to use a different approach — use preprocessing_args or
    # just use scale=1/255 and let the model handle normalization.
    # Simpler: bake normalization into a wrapper.

    # Wrapper model that does ImageNet normalization
    class NormalizedModel(torch.nn.Module):
        def __init__(self, base_model, mean, std):
            super().__init__()
            self.base_model = base_model
            self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            return self.base_model(x).logits

    wrapped = NormalizedModel(model, mean, std)
    wrapped.eval()

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_input)

    # CoreML input: scale pixel [0,255] to [0,1]
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, 224, 224),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB,
    )

    # Step 4: Convert
    print("4/5  Converting to CoreML (float16)...")
    class_labels = ["normal", "nsfw"]  # Falconsai label order: 0=normal, 1=nsfw
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(class_labels),
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    # Step 5: Save
    print("5/5  Saving...")
    mlmodel.author = "NSFWScanner (model: Falconsai/nsfw_image_detection)"
    mlmodel.short_description = "NSFW classifier (ViT-Base-224). Classes: normal, nsfw."
    mlmodel.version = "1.0"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\nDone! {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
```

**Step 2: Update requirements.txt**

Add `transformers>=4.40` to `model_conversion/requirements.txt` (needed for Falconsai, viddexa, AdamCodd).

**Step 3: Run conversion**

```bash
cd model_conversion
source venv/bin/activate
pip install -r requirements.txt
python convert_falconsai.py
```

Expected: `FalconsaiNSFW.mlpackage` created in `NSFWScanner/Resources/` (~170 MB F16 for ViT-Base)

**Step 4: Verify the model loads**

Open the `.mlpackage` in Xcode or run a quick Python verification:
```python
import coremltools as ct
m = ct.models.MLModel("../NSFWScanner/Resources/FalconsaiNSFW.mlpackage")
print(m.get_spec().description)
```

**Step 5: Commit**

```bash
git add model_conversion/convert_falconsai.py model_conversion/requirements.txt
git commit -m "feat: add Falconsai model conversion script"
```

Note: The .mlpackage is large — commit it separately or with the other models in a single commit at the end of Phase 1.

---

### Task 2: Create conversion script for viddexa/nsfw-detection-2-mini

**Files:**
- Create: `model_conversion/convert_viddexa.py`

**Step 1: Write the conversion script**

```python
#!/usr/bin/env python3
"""Convert viddexa/nsfw-detection-2-mini (EfficientNet-B4, 5-class) to CoreML .mlpackage."""

import sys
from pathlib import Path

import coremltools as ct
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
import numpy as np


def main():
    output_path = Path(__file__).parent.parent / "NSFWScanner" / "Resources" / "ViddexaNSFW.mlpackage"

    print("1/5  Loading viddexa/nsfw-detection-2-mini...")
    model = AutoModelForImageClassification.from_pretrained("viddexa/nsfw-detection-2-mini")
    processor = AutoImageProcessor.from_pretrained("viddexa/nsfw-detection-2-mini")
    model.eval()
    print(f"     Labels: {model.config.id2label}")
    print(f"     Processor: size={processor.size}, normalize mean={processor.image_mean}, std={processor.image_std}")

    # Get normalization params from processor
    mean = processor.image_mean  # typically ImageNet
    std = processor.image_std
    size = processor.size.get("shortest_edge", processor.size.get("height", 224))

    print(f"2/5  Wrapping model with normalization (input {size}x{size})...")

    class NormalizedModel(torch.nn.Module):
        def __init__(self, base_model, mean, std):
            super().__init__()
            self.base_model = base_model
            self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            return self.base_model(x).logits

    wrapped = NormalizedModel(model, mean, std)
    wrapped.eval()

    example_input = torch.rand(1, 3, size, size)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_input)
    print("     Trace successful")

    print("3/5  Defining CoreML input...")
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, size, size),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB,
    )

    # 5 classes: Normal, Porn, Hentai, Drawing, Sexy
    # Get labels in index order from model config
    class_labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    print(f"     Class labels (by index): {class_labels}")

    print("4/5  Converting to CoreML (float16)...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(class_labels),
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    print("5/5  Saving...")
    mlmodel.author = "NSFWScanner (model: viddexa/nsfw-detection-2-mini)"
    mlmodel.short_description = f"5-class NSFW classifier (EfficientNet-B4, {size}x{size}). Classes: {', '.join(class_labels)}."
    mlmodel.version = "1.0"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\nDone! {output_path} ({size_mb:.1f} MB)")
    print(f"     Labels: {class_labels}")


if __name__ == "__main__":
    main()
```

**Step 2: Run conversion**

```bash
cd model_conversion && source venv/bin/activate && python convert_viddexa.py
```

Expected: `ViddexaNSFW.mlpackage` in Resources (~20-65 MB)

**Step 3: Verify class labels match expected order**

The 5-class output must be: check `model.config.id2label` and record the exact label order. This is critical for the Swift category toggles.

**Step 4: Commit**

```bash
git add model_conversion/convert_viddexa.py
git commit -m "feat: add viddexa 5-class model conversion script"
```

---

### Task 3: Create conversion script for AdamCodd/vit-base-nsfw-detector

**Files:**
- Create: `model_conversion/convert_adamcodd.py`

**Step 1: Write the conversion script**

Same pattern as Falconsai but for AdamCodd — ViT-Base-384, 2 classes (NSFW, SFW):

```python
#!/usr/bin/env python3
"""Convert AdamCodd/vit-base-nsfw-detector (ViT-Base-384) to CoreML .mlpackage."""

import sys
from pathlib import Path

import coremltools as ct
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor


def main():
    output_path = Path(__file__).parent.parent / "NSFWScanner" / "Resources" / "AdamCoddNSFW.mlpackage"

    print("1/5  Loading AdamCodd/vit-base-nsfw-detector...")
    model = AutoModelForImageClassification.from_pretrained("AdamCodd/vit-base-nsfw-detector")
    processor = AutoImageProcessor.from_pretrained("AdamCodd/vit-base-nsfw-detector")
    model.eval()
    print(f"     Labels: {model.config.id2label}")

    mean = processor.image_mean
    std = processor.image_std
    size = processor.size.get("shortest_edge", processor.size.get("height", 384))

    print(f"2/5  Wrapping model with normalization (input {size}x{size})...")

    class NormalizedModel(torch.nn.Module):
        def __init__(self, base_model, mean, std):
            super().__init__()
            self.base_model = base_model
            self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            return self.base_model(x).logits

    wrapped = NormalizedModel(model, mean, std)
    wrapped.eval()

    example_input = torch.rand(1, 3, size, size)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_input)
    print("     Trace successful")

    print("3/5  Defining CoreML input...")
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, size, size),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB,
    )

    class_labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    print(f"     Class labels: {class_labels}")

    print("4/5  Converting to CoreML (float16)...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(class_labels),
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    print("5/5  Saving...")
    mlmodel.author = "NSFWScanner (model: AdamCodd/vit-base-nsfw-detector)"
    mlmodel.short_description = f"NSFW classifier (ViT-Base-{size}). Classes: {', '.join(class_labels)}."
    mlmodel.version = "1.0"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\nDone! {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
```

**Step 2: Run conversion**

```bash
cd model_conversion && source venv/bin/activate && python convert_adamcodd.py
```

Expected: `AdamCoddNSFW.mlpackage` in Resources (~170 MB F16 for ViT-Base)

**Step 3: Commit**

```bash
git add model_conversion/convert_adamcodd.py
git commit -m "feat: add AdamCodd model conversion script"
```

---

### Task 4: Run all conversions, verify, and commit .mlpackage files

**Step 1: Run all 3 new conversion scripts**

```bash
cd model_conversion && source venv/bin/activate
python convert_falconsai.py
python convert_viddexa.py
python convert_adamcodd.py
```

**Step 2: Verify all 4 .mlpackage files exist in Resources**

```bash
ls -lh ../NSFWScanner/Resources/*.mlpackage
```

Expected 4 packages: `NSFWClassifier.mlpackage`, `FalconsaiNSFW.mlpackage`, `ViddexaNSFW.mlpackage`, `AdamCoddNSFW.mlpackage`

**Step 3: Quick verification — load each in Python**

```python
import coremltools as ct
for name in ["NSFWClassifier", "FalconsaiNSFW", "ViddexaNSFW", "AdamCoddNSFW"]:
    m = ct.models.MLModel(f"../NSFWScanner/Resources/{name}.mlpackage")
    spec = m.get_spec().description
    print(f"{name}: input={spec.input[0].name}, classes={[c for c in spec.predictedProbabilitiesName]}")
```

**Step 4: Commit all .mlpackage files**

```bash
git add NSFWScanner/Resources/*.mlpackage
git commit -m "feat: bundle all 4 ML models (Marqo, Falconsai, viddexa, AdamCodd)"
```

Note: This will be a large commit (~350+ MB). Consider using Git LFS if the repo size becomes problematic.

---

## Phase 2: Multi-Model Swift Architecture

### Task 5: Create NSFWModel enum

**Files:**
- Create: `NSFWScanner/Models/NSFWModel.swift`

**Step 1: Write NSFWModel enum**

```swift
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
        case .marqo: "Fast & accurate (384x384, 11 MB)"
        case .falconsai: "Most popular (224x224, ~170 MB)"
        case .viddexa: "Granular categories (224x224, ~20 MB)"
        case .adamcodd: "Strong on drawings (384x384, ~170 MB)"
        }
    }

    var resourceName: String {
        switch self {
        case .marqo: "NSFWClassifier"
        case .falconsai: "FalconsaiNSFW"
        case .viddexa: "ViddexaNSFW"
        case .adamcodd: "AdamCoddNSFW"
        }
    }

    /// The label(s) that indicate NSFW content for this model.
    /// For binary models, this is ["NSFW"] or ["nsfw"].
    /// For viddexa 5-class, this is the full set of NSFW categories.
    var nsfwLabels: Set<String> {
        switch self {
        case .marqo: ["NSFW"]
        case .falconsai: ["nsfw"]
        case .viddexa: ["porn", "hentai", "sexy"]  // default NSFW categories
        case .adamcodd: ["nsfw"]
        }
    }

    /// Whether this model supports granular category selection.
    var hasCategories: Bool { self == .viddexa }

    /// All categories for the 5-class model.
    static let viddexaCategories = ["porn", "hentai", "sexy", "drawing"]
}
```

**Step 2: Build to verify**

```bash
xcodebuild -project NSFWScanner.xcodeproj -scheme NSFWScanner -configuration Debug build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

**Step 3: Commit**

```bash
git add NSFWScanner/Models/NSFWModel.swift
git commit -m "feat: add NSFWModel enum with metadata for 4 models"
```

---

### Task 6: Update ClassifierService for multi-model support

**Files:**
- Modify: `NSFWScanner/Services/ClassifierService.swift`

**Step 1: Rewrite ClassifierService to support multiple models**

Replace the entire file. Key changes:
- `loadModel(_ model: NSFWModel)` instead of `loadModel()`
- Track `currentModel: NSFWModel?` to avoid redundant reloads
- Cache compiled models per-model in Caches directory
- `classify()` returns the raw `(label, confidence)` — the caller decides if it's NSFW based on model-specific labels

```swift
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
```

**Step 2: Build to verify**

```bash
xcodebuild -project NSFWScanner.xcodeproj -scheme NSFWScanner build 2>&1 | tail -5
```

Expected: Build will fail because ScanOrchestrator still calls `classifier.loadModel()` without a parameter. That's fixed in Task 7.

**Step 3: Commit**

```bash
git add NSFWScanner/Services/ClassifierService.swift
git commit -m "feat: ClassifierService supports multiple models"
```

---

### Task 7: Update ScanOrchestrator for model selection + categories + @AppStorage

**Files:**
- Modify: `NSFWScanner/Services/ScanOrchestrator.swift`

**Step 1: Add model selection, category toggles, album name, and @AppStorage persistence**

Key changes to ScanOrchestrator:
- Add `selectedModel: NSFWModel` property (default `.marqo`)
- Add `albumName: String` property (default `"NSFW"`)
- Add `viddexaCategories: Set<String>` for the 5-class toggle state
- Update `startScan()` to call `classifier.loadModel(selectedModel)`
- Update NSFW detection logic: for viddexa, check if the label is in `viddexaCategories`; for others, check against `model.nsfwLabels`
- Update `commitFlaggedToAlbum` to use `albumName`
- Add `hideAssets(ids:)` method for Hidden album support
- Persist all settings with `@AppStorage` wrapped in a helper (since `@Observable` can't use `@AppStorage` directly, use `UserDefaults` in a didSet or init)

The `classifyImage` and `classifyVideo` static methods need an additional parameter: the set of labels to treat as NSFW.

In `startScan()`, before the scan loop:
```swift
let nsfwLabels: Set<String>
if selectedModel == .viddexa {
    nsfwLabels = viddexaCategories
} else {
    nsfwLabels = selectedModel.nsfwLabels
}
```

Then pass `nsfwLabels` through to the classify helpers. Change the check from:
```swift
if result.label == "NSFW" && result.confidence >= threshold
```
to:
```swift
if nsfwLabels.contains(result.label.lowercased()) && result.confidence >= threshold
```

For persistence, in `init()`:
```swift
init() {
    let defaults = UserDefaults.standard
    if let raw = defaults.string(forKey: "selectedModel"),
       let model = NSFWModel(rawValue: raw) {
        selectedModel = model
    }
    confidenceThreshold = defaults.float(forKey: "confidenceThreshold").nonZero ?? 0.85
    frameInterval = defaults.double(forKey: "frameInterval").nonZero ?? 5.0
    imageConcurrency = defaults.double(forKey: "imageConcurrency").nonZero ?? 20
    videoConcurrency = defaults.double(forKey: "videoConcurrency").nonZero ?? 16
    albumName = defaults.string(forKey: "albumName") ?? "NSFW"
    // ... load viddexaCategories from defaults
}
```

And save on changes using a `saveSettings()` helper called from relevant property observers or a debounced save.

Add `hideAssets(ids:)`:
```swift
func hideAssets(ids: Set<String>) async {
    state = .committingToAlbum
    do {
        let assetsToHide = flaggedResults.filter { ids.contains($0.id) }.map(\.asset)
        try await photoLibrary.hideAssets(assetsToHide)
        flaggedResults.removeAll { ids.contains($0.id) }
        flaggedImageCount = flaggedResults.filter { $0.mediaType == .image }.count
        flaggedVideoCount = flaggedResults.filter { $0.mediaType == .video }.count
        state = .reviewing
    } catch {
        state = .error(error.localizedDescription)
    }
}
```

**Step 2: Build and fix any compilation errors**

**Step 3: Commit**

```bash
git add NSFWScanner/Services/ScanOrchestrator.swift
git commit -m "feat: model selection, category toggles, album name, settings persistence"
```

---

### Task 8: Add hideAssets to PhotoLibraryService

**Files:**
- Modify: `NSFWScanner/Services/PhotoLibraryService.swift`

**Step 1: Add hideAssets method and parameterize album name**

Add to PhotoLibraryService:

```swift
func hideAssets(_ assets: [PHAsset]) async throws {
    try await PHPhotoLibrary.shared().performChanges {
        for asset in assets {
            let request = PHAssetChangeRequest(for: asset)
            request.isHidden = true
        }
    }
    photoLogger.info("Hidden \(assets.count) assets")
}
```

Also update `createNSFWAlbumIfNeeded()` to accept a name parameter:
```swift
func createAlbumIfNeeded(named albumName: String) async throws -> PHAssetCollection {
    // ... same logic but use albumName instead of hardcoded "NSFW"
}
```

And update `nsfwAlbumAssetIDs()` to accept a name:
```swift
func albumAssetIDs(albumName: String) async -> Set<String> {
    // ... same logic but search for albumName
}
```

**Step 2: Update ScanOrchestrator calls to use the new parameterized methods**

**Step 3: Build and verify**

**Step 4: Commit**

```bash
git add NSFWScanner/Services/PhotoLibraryService.swift NSFWScanner/Services/ScanOrchestrator.swift
git commit -m "feat: hideAssets + parameterized album name in PhotoLibraryService"
```

---

## Phase 3: UI Polish

### Task 9: Toolbar and sidebar restructure

**Files:**
- Modify: `NSFWScanner/Views/ContentView.swift`
- Modify: `NSFWScanner/Views/ScanControlView.swift`

**Step 1: Add toolbar to ContentView**

Add `.toolbar { }` to the NavigationSplitView with:
- Start Scan / Cancel button (primary action)
- Model picker as a compact `Menu` or `Picker`

```swift
.toolbar {
    ToolbarItemGroup(placement: .primaryAction) {
        switch orchestrator.state {
        case .idle, .error:
            Button("Start Scan", systemImage: "play.fill") {
                orchestrator.startScan()
            }
            .disabled(!orchestrator.scanImages && !orchestrator.scanVideos)
        case .scanning:
            Button("Cancel", systemImage: "stop.fill", role: .destructive) {
                orchestrator.cancelScan()
            }
        case .reviewing, .committingToAlbum:
            Button("New Scan", systemImage: "arrow.counterclockwise") {
                orchestrator.resetToIdle()
            }
        case .requestingPermission:
            ProgressView()
                .controlSize(.small)
        }
    }
}
```

**Step 2: Clean up ScanControlView sidebar**

Remove the scan button section from the sidebar (it's now in the toolbar). The sidebar becomes:
1. Progress section (during scan)
2. Results section (during/after scan)
3. Scan Types — toggles
4. Model — picker + category toggles
5. Settings — confidence, frame interval, album name
6. Performance — concurrency sliders

Add model picker:
```swift
Section("Model") {
    Picker("Model", selection: $orchestrator.selectedModel) {
        ForEach(NSFWModel.allCases) { model in
            VStack(alignment: .leading) {
                Text(model.displayName)
                Text(model.subtitle)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .tag(model)
        }
    }
    .pickerStyle(.menu)

    if orchestrator.selectedModel.hasCategories {
        // Category toggles for viddexa
        ForEach(NSFWModel.viddexaCategories, id: \.self) { category in
            Toggle(category.capitalized, isOn: Binding(
                get: { orchestrator.viddexaCategories.contains(category) },
                set: { enabled in
                    if enabled {
                        orchestrator.viddexaCategories.insert(category)
                    } else {
                        orchestrator.viddexaCategories.remove(category)
                    }
                }
            ))
        }
    }
}
.disabled(orchestrator.state == .scanning)
```

Add album name field in Settings section:
```swift
TextField("Album name", text: $orchestrator.albumName)
    .textFieldStyle(.roundedBorder)
```

**Step 3: Build and verify**

**Step 4: Commit**

```bash
git add NSFWScanner/Views/ContentView.swift NSFWScanner/Views/ScanControlView.swift
git commit -m "feat: toolbar actions, model picker, category toggles, album name field"
```

---

### Task 10: Results grid polish + dual action buttons

**Files:**
- Modify: `NSFWScanner/Views/ResultsGridView.swift`

**Step 1: Update the bottom action bar**

Replace the single "Add to NSFW Album" button with two buttons:

```swift
HStack(spacing: 12) {
    Text("\(orchestrator.flaggedResults.count) flagged, \(selectedIDs.count) selected")
        .foregroundStyle(.secondary)
        .font(.callout)

    Spacer()

    Button("Select All") {
        selectedIDs = Set(orchestrator.flaggedResults.map(\.id))
    }
    .disabled(orchestrator.flaggedResults.isEmpty)

    Button("Deselect") {
        selectedIDs.removeAll()
    }
    .disabled(selectedIDs.isEmpty)

    Divider()
        .frame(height: 20)

    Button("Dismiss", systemImage: "xmark.circle") {
        orchestrator.dismissResults(ids: effectiveSelection)
        selectedIDs.removeAll()
    }
    .disabled(orchestrator.flaggedResults.isEmpty)

    if commitInProgress {
        ProgressView()
            .controlSize(.small)
    } else {
        Button("Hide", systemImage: "eye.slash") {
            commitInProgress = true
            Task {
                await orchestrator.hideAssets(ids: effectiveSelection)
                selectedIDs.removeAll()
                commitInProgress = false
            }
        }

        Button("Move to Album", systemImage: "folder.badge.plus") {
            commitInProgress = true
            Task {
                await orchestrator.commitFlaggedToAlbum(selectedIDs: effectiveSelection)
                selectedIDs.removeAll()
                commitInProgress = false
            }
        }
        .buttonStyle(.borderedProminent)
    }
}
```

Where `effectiveSelection` is:
```swift
private var effectiveSelection: Set<String> {
    selectedIDs.isEmpty ? Set(orchestrator.flaggedResults.map(\.id)) : selectedIDs
}
```

**Step 2: Add hover effect to ThumbnailCell**

```swift
@State private var isHovered = false

// On the ZStack:
.scaleEffect(isHovered ? 1.02 : 1.0)
.shadow(color: .black.opacity(isHovered ? 0.2 : 0), radius: 4, y: 2)
.animation(.easeInOut(duration: 0.15), value: isHovered)
.onHover { hovering in isHovered = hovering }
```

**Step 3: Add keyboard shortcuts**

On the VStack body:
```swift
.keyboardShortcut(.delete, modifiers: [])  // Delete to dismiss
// Use .onKeyPress or focusable + keyboard handling for Cmd+A
```

**Step 4: Build and verify**

**Step 5: Commit**

```bash
git add NSFWScanner/Views/ResultsGridView.swift
git commit -m "feat: dual action buttons (album + hide), hover effects, keyboard shortcuts"
```

---

### Task 11: Detail sheet and idle state polish

**Files:**
- Modify: `NSFWScanner/Views/AssetDetailView.swift`
- Modify: `NSFWScanner/Views/ContentView.swift`

**Step 1: Polish AssetDetailView**

- Larger default frame: `minWidth: 700, minHeight: 600`
- Better typography hierarchy
- Add model name that flagged the asset (if we store it on ScanResult)

**Step 2: Polish ContentView idle state**

- Add selected model info card
- Refine help text typography and spacing
- Add scan complete summary when transitioning from scanning to reviewing:

In the `.reviewing` case, if results are empty:
```swift
ContentUnavailableView(
    "All Clear",
    systemImage: "checkmark.shield.fill",
    description: Text("Scanned \(orchestrator.processedCount) items. No NSFW content detected.")
)
```

**Step 3: Add state transition animations**

```swift
.animation(.easeInOut, value: orchestrator.state)
```

**Step 4: Build and verify**

**Step 5: Commit**

```bash
git add NSFWScanner/Views/AssetDetailView.swift NSFWScanner/Views/ContentView.swift
git commit -m "feat: polished detail sheet, idle state, scan summary, animations"
```

---

## Phase 4: Final

### Task 12: Update README and clean up

**Files:**
- Modify: `README.md`
- Modify: `model_conversion/requirements.txt`

**Step 1: Update README**

- Add all 4 models to the ML Model section with a comparison table
- Update the Settings table with new settings (model picker, album name, category toggles)
- Update Project Structure to reflect new files
- Update Features list

**Step 2: Update .gitignore if needed**

Ensure model_conversion temp files are excluded.

**Step 3: Commit**

```bash
git add README.md .gitignore model_conversion/requirements.txt
git commit -m "docs: update README for v2.0 multi-model release"
```

---

### Task 13: Full build, test, and release

**Step 1: Clean build**

```bash
xcodebuild -project NSFWScanner.xcodeproj -scheme NSFWScanner -configuration Release clean build
```

Expected: BUILD SUCCEEDED

**Step 2: Manual testing checklist**

- [ ] App launches, shows idle state with model info
- [ ] Model picker shows all 4 models with descriptions
- [ ] Selecting viddexa shows category toggles
- [ ] Category toggles appear/disappear when switching models
- [ ] Album name field accepts custom text
- [ ] Start Scan works with each model
- [ ] Progress bars show correctly during scan
- [ ] Results grid displays flagged items
- [ ] "Move to Album" moves items to the named album
- [ ] "Hide" hides items in Photos
- [ ] "Dismiss" removes items from results
- [ ] Hover effects work on thumbnail cells
- [ ] Detail sheet opens on double-click
- [ ] Settings persist after app restart
- [ ] New Scan button works after reviewing

**Step 3: Tag and push**

```bash
git tag v2.0.0
git push origin main --tags
```

The GitHub Actions workflow will automatically build, notarize, and create the release.

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Model conversion fails for a model | Each script is independent — ship with whatever models convert successfully, add others later |
| .mlpackage files too large for git | Use Git LFS if total exceeds ~500 MB |
| Tracing fails due to attention fast paths | Apply same SDPA/MHA disabling trick from Marqo conversion |
| Falconsai/AdamCodd label order differs | Read `model.config.id2label` and normalize in Swift (case-insensitive label matching) |
| App bundle size too large | F16 quantization already applied; INT8 quantization available as fallback |
