# NSFWScanner v2.0 Design — Multi-Model, UI Polish, Settings

**Date:** 2026-02-24
**Status:** Approved

---

## 1. Multi-Model Architecture

### Models (4 total, all bundled as .mlpackage)

| Model | App ID | Size (F16) | Input | Classes | Best for |
|---|---|---|---|---|---|
| Marqo/nsfw-image-detection-384 | `marqo-vit-tiny` | ~11 MB | 384x384 | NSFW, SFW | Speed + accuracy (default) |
| Falconsai/nsfw_image_detection | `falconsai-vit-base` | ~80 MB | 224x224 | nsfw, normal | Most popular, well-tested |
| viddexa/nsfw-detection-2-mini | `viddexa-efficientnet` | ~20 MB | 224x224 | Porn, Hentai, Sexy, Drawing, Normal | Granular category control |
| AdamCodd/vit-base-nsfw-detector | `adamcodd-vit-base` | ~80 MB | 384x384 | NSFW, SFW | Strong on drawings/illustrations |

### ClassifierService Changes

- New `NSFWModel` enum with metadata (display name, input size, class labels, description)
- `loadModel(_ model: NSFWModel)` replaces single-model load
- Each model compiled to `.mlmodelc` on first use, cached in Caches directory
- Vision `VNCoreMLRequest` with `.scaleFill` handles different input sizes automatically

### Category Toggles (viddexa 5-class only)

- When viddexa model selected, show toggles: Porn, Hentai, Sexy, Drawing
- If top classification lands in a checked category above threshold, asset is flagged
- Other models use binary NSFW threshold as today

### Conversion Scripts

- Unified `convert_model.py` with model selector argument, or one script per model
- `verify_model.py` validates each conversion against PyTorch reference
- All 4 `.mlpackage` files committed to repo

---

## 2. Destination Options

### Two Actions

1. **"Move to Album"** — moves to user-named album (configurable, default "NSFW")
2. **"Hide in Photos"** — `PHAssetChangeRequest.isHidden = true` → system Hidden album

Both operate on selected items (or all if none selected).

### Configurable Album Name

- Text field in Settings, default "NSFW"
- `PhotoLibraryService.createAlbumIfNeeded(name:)` parameterized
- Skip-already-scanned logic uses configured album name

### Hidden Album Caveat

- Hidden items can't be detected on re-scan if Face ID lock enabled
- Acceptable trade-off — items may be re-flagged on subsequent scans

---

## 3. UI Polish — Clean macOS Native

### Toolbar

- Start/Cancel scan as toolbar buttons (`.toolbar { }`)
- Model picker as compact toolbar dropdown
- Remove scan button from sidebar body

### Sidebar

- Progress section (during scan) — dual progress bars
- Results section — flagged counts
- Scan Types — Photos/Videos toggles
- Model — picker + category toggles (viddexa only)
- Settings — Confidence, Frame interval, Album name
- Performance — Concurrency sliders

### Results Grid

- Slightly larger thumbnails, refined corner radius
- Hover state: subtle scale + shadow
- Selection: blue ring + checkmark with animation
- Bottom bar: "Move to Album" and "Hide" as separate buttons

### Detail Sheet

- Larger default size, proper padding/typography
- Video: player with scrubber, flagged frame timestamp

### Idle State

- Refined help text cards with model info card
- Clean typography and spacing

### Animations

- State transitions (idle → scanning → reviewing)
- Thumbnail grid entrance with `.transition`

---

## 4. Settings & New Features

### New Settings

- Album name (text field, default "NSFW")
- Model picker (dropdown, 4 models)
- Category toggles (viddexa 5-class only)

### Existing Settings (unchanged)

- Confidence threshold (50-99%)
- Frame interval (1-30s)
- Photo/Video concurrency (1-128)
- Scan type toggles (Photos/Videos)

### New Features

- **Scan complete summary** — inline banner: "Scanned X images and Y videos. Z flagged."
- **Keyboard shortcuts** — Cmd+A select all, Delete dismiss selected, Enter commit
- **Persist settings** — `@AppStorage` for model choice, thresholds, album name, concurrency

### Explicitly NOT Adding

- Folder/directory scanning
- Export/report
- Scheduled scans
- Per-photo model override
- Undo after commit
