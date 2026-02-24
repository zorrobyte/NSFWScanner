# NSFWScanner

A native macOS app that scans your Apple Photos library for NSFW content using on-device machine learning. Flagged photos and videos can be reviewed and moved to a dedicated "NSFW" album.

All classification runs locally on your Mac — no images are uploaded anywhere.

## Features

- Scans your entire Photos library (images and videos)
- On-device ML classification using Core ML and the Vision framework
- Video scanning with configurable frame extraction intervals
- Separate parallel queues for photos and videos with independent progress tracking
- Adjustable confidence threshold to tune sensitivity
- Configurable concurrency for both photo and video processing
- Review flagged results in a thumbnail grid
- Dismiss false positives before committing
- Move confirmed NSFW content to a dedicated "NSFW" album
- Skips assets already in the NSFW album on subsequent scans
- Optimized for Apple Silicon (GPU + ANE acceleration)

## Requirements

- macOS 15.0+
- Xcode 16+
- Apple Silicon or Intel Mac with Neural Engine

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/zorrobyte/NSFWScanner.git
   ```

2. Open `NSFWScanner.xcodeproj` in Xcode.

3. Build and run. The CoreML model is bundled in the repo — no additional setup needed.

4. Grant Photos library access when prompted.

5. Press **Start Scan** and review the results.

## How It Works

The app requests read/write access to your Photos library, then processes each asset through a Vision-based CoreML classifier. Images are downscaled to 384x384 for classification. Videos are sampled at a configurable frame interval (default: 5 seconds) and each frame is classified independently — if any frame exceeds the confidence threshold, the video is flagged.

Photos and videos are processed in separate concurrent queues so slow video processing doesn't block fast image scanning. Both queues have configurable concurrency levels.

## ML Model

This app uses the [**Marqo/nsfw-image-detection-384**](https://huggingface.co/Marqo/nsfw-image-detection-384) model by [Marqo](https://www.marqo.ai/), converted from PyTorch to CoreML format.

| Detail | Value |
|---|---|
| Architecture | ViT-Tiny (`vit_tiny_patch16_384`) |
| Input | 384x384 RGB |
| Classes | NSFW, SFW |
| Accuracy | 98.56% on the model author's benchmark |
| Size | ~11 MB (float16) |
| License | Apache 2.0 (model weights) |

The conversion scripts are in the `model_conversion/` directory. To re-convert the model:

```bash
cd model_conversion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python convert_model.py
python verify_model.py
```

Image normalization (mean=0.5, std=0.5) is baked into the CoreML model at conversion time, so the Swift app passes raw images with zero preprocessing.

### Model Attribution

The NSFW classification model was created by [Marqo](https://huggingface.co/Marqo) and is available on Hugging Face under the Apache 2.0 license:
- **Model**: [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384)
- **Paper/Blog**: [Marqo Blog](https://www.marqo.ai/blog)

## Project Structure

```
NSFWScanner/
├── NSFWScannerApp.swift               # App entry point
├── Models/
│   ├── ScanResult.swift               # Flagged asset data
│   └── ScanState.swift                # State machine enum
├── Services/
│   ├── ClassifierService.swift        # CoreML + Vision classification
│   ├── PhotoLibraryService.swift      # Photos framework integration
│   ├── ScanOrchestrator.swift         # Scan state machine & parallel queues
│   └── VideoFrameExtractor.swift      # AVAssetImageGenerator frame extraction
├── Views/
│   ├── ContentView.swift              # Main NavigationSplitView
│   ├── ScanControlView.swift          # Sidebar controls & settings
│   ├── ResultsGridView.swift          # Flagged results thumbnail grid
│   └── AssetDetailView.swift          # Full-size preview sheet
├── Resources/
│   └── NSFWClassifier.mlpackage       # Bundled CoreML model
└── model_conversion/
    ├── convert_model.py               # PyTorch → CoreML conversion
    ├── verify_model.py                # Conversion validation
    └── requirements.txt               # Python dependencies
```

## Settings

| Setting | Default | Range | Description |
|---|---|---|---|
| Confidence | 85% | 50–99% | How sure the AI must be before flagging. Higher = fewer false positives. |
| Frame interval | 5s | 1–30s | Seconds between video frame checks. Lower = more thorough. |
| Photo tasks | 20 | 1–128 | Concurrent photo classification tasks. |
| Video tasks | 16 | 1–128 | Concurrent video classification tasks. |

## Privacy

- All processing happens on-device using CoreML
- No network requests are made for classification
- No data leaves your Mac
- Photos library access is required for scanning and album creation

## License

MIT License — see [LICENSE](LICENSE) for details.

The bundled ML model weights are licensed under Apache 2.0 by [Marqo](https://huggingface.co/Marqo/nsfw-image-detection-384).
