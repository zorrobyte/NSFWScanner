# NSFWScanner

A native macOS app that scans your Apple Photos library for NSFW content using on-device machine learning. Flagged photos and videos can be reviewed, moved to a configurable album, or hidden in the Photos Hidden album.

All classification runs locally on your Mac — no images are uploaded anywhere.

## Features

- **4 bundled ML models** — choose the best model for your use case
- Scans your entire Photos library (images and videos)
- On-device ML classification using Core ML and the Vision framework
- Video scanning with configurable frame extraction intervals
- Separate parallel queues for photos and videos with independent progress tracking
- Adjustable confidence threshold to tune sensitivity
- Configurable concurrency for both photo and video processing
- Review flagged results in a thumbnail grid with hover previews
- Dismiss false positives before committing
- **Move to Album** — move flagged content to a configurable album (default "NSFW")
- **Hide in Photos** — hide flagged content in the system Hidden album
- Granular category toggles for the 5-class model (Porn, Hentai, Sexy, Drawing)
- Settings persist across app launches
- Skips assets already in the target album on subsequent scans
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

3. Build and run. All 4 CoreML models are bundled in the repo — no additional setup needed.

4. Grant Photos library access when prompted.

5. Select a model from the sidebar, adjust settings, and press **Start Scan**.

## How It Works

The app requests read/write access to your Photos library, then processes each asset through a Vision-based CoreML classifier. Images are resized to the model's input size for classification. Videos are sampled at a configurable frame interval (default: 5 seconds) and each frame is classified independently — if any frame exceeds the confidence threshold, the video is flagged.

Photos and videos are processed in separate concurrent queues so slow video processing doesn't block fast image scanning. Both queues have configurable concurrency levels.

## ML Models

NSFWScanner bundles 4 models. Choose the best fit from the sidebar:

| Model | Architecture | Input | Classes | Size (F16) | Best for |
|---|---|---|---|---|---|
| [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384) | ViT-Tiny | 384x384 | NSFW, SFW | ~11 MB | Speed + accuracy (default) |
| [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | ViT-Base | 224x224 | nsfw, normal | ~164 MB | Most popular, well-tested |
| [viddexa/nsfw-detection-2-mini](https://huggingface.co/viddexa/nsfw-detection-2-mini) | EfficientNet-B4 | 380x380 | safe, hentai, porn, sexy, drawing | ~34 MB | Granular category control |
| [AdamCodd/vit-base-nsfw-detector](https://huggingface.co/AdamCodd/vit-base-nsfw-detector) | ViT-Base | 384x384 | nsfw, sfw | ~164 MB | Strong on drawings/illustrations |

The conversion scripts are in the `model_conversion/` directory. To re-convert any model:

```bash
cd model_conversion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python convert_model.py           # Marqo (default)
python convert_falconsai.py       # Falconsai
python convert_viddexa.py         # viddexa
python convert_adamcodd.py        # AdamCodd
```

Image normalization is baked into each CoreML model at conversion time, so the Swift app passes raw images with zero preprocessing.

### Model Attribution

All models are available on Hugging Face:
- **Marqo/nsfw-image-detection-384** — Apache 2.0 by [Marqo](https://www.marqo.ai/)
- **Falconsai/nsfw_image_detection** — by [Falconsai](https://huggingface.co/Falconsai)
- **viddexa/nsfw-detection-2-mini** — by [viddexa](https://huggingface.co/viddexa)
- **AdamCodd/vit-base-nsfw-detector** — by [AdamCodd](https://huggingface.co/AdamCodd)

## Project Structure

```
NSFWScanner/
├── NSFWScannerApp.swift               # App entry point
├── Models/
│   ├── NSFWModel.swift                # Model enum with metadata for 4 models
│   ├── ScanResult.swift               # Flagged asset data
│   └── ScanState.swift                # State machine enum
├── Services/
│   ├── ClassifierService.swift        # Multi-model CoreML + Vision classification
│   ├── PhotoLibraryService.swift      # Photos framework integration + hide support
│   ├── ScanOrchestrator.swift         # Scan state machine & parallel queues
│   └── VideoFrameExtractor.swift      # AVAssetImageGenerator frame extraction
├── Views/
│   ├── ContentView.swift              # Main NavigationSplitView with toolbar
│   ├── ScanControlView.swift          # Sidebar: model picker, settings, progress
│   ├── ResultsGridView.swift          # Flagged results grid with dual actions
│   └── AssetDetailView.swift          # Full-size preview sheet
├── Resources/
│   ├── NSFWClassifier.mlpackage       # Marqo model (~11 MB)
│   ├── FalconsaiNSFW.mlpackage        # Falconsai model (~164 MB, Git LFS)
│   ├── ViddexaNSFW.mlpackage          # viddexa model (~34 MB)
│   └── AdamCoddNSFW.mlpackage         # AdamCodd model (~164 MB, Git LFS)
└── model_conversion/
    ├── convert_model.py               # Marqo PyTorch → CoreML
    ├── convert_falconsai.py           # Falconsai conversion
    ├── convert_viddexa.py             # viddexa conversion
    ├── convert_adamcodd.py            # AdamCodd conversion
    ├── verify_model.py                # Conversion validation
    └── requirements.txt               # Python dependencies
```

## Settings

| Setting | Default | Range | Description |
|---|---|---|---|
| Model | Marqo ViT-Tiny | 4 models | ML model used for classification |
| Album name | NSFW | any text | Name of the album for flagged content |
| Category toggles | Porn, Hentai, Sexy | 4 categories | Which categories to flag (viddexa 5-class only) |
| Confidence | 85% | 50–99% | How sure the AI must be before flagging |
| Frame interval | 5s | 1–30s | Seconds between video frame checks |
| Photo tasks | 20 | 1–128 | Concurrent photo classification tasks |
| Video tasks | 16 | 1–128 | Concurrent video classification tasks |

All settings are persisted across app launches.

## Privacy

- All processing happens on-device using CoreML
- No network requests are made for classification
- No data leaves your Mac
- Photos library access is required for scanning and album creation

## License

MIT License — see [LICENSE](LICENSE) for details.

The bundled ML model weights are licensed under their respective licenses by their authors (see Model Attribution above).
