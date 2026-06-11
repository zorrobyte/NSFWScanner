# Photo Library Sorter — Design

**Date**: 2026-02-24
**Status**: Approved

## Summary

Add a "Sort Library" mode to NSFWScanner that sends photos to the Qwen3-VL-8B vision model on 5090desktop for categorization, then creates Photos albums per category.

## Architecture

New mode alongside existing NSFW Scan. Entirely additive — no changes to existing scan flow.

### Flow

1. User selects "Sort Library" mode in toolbar
2. App fetches all photo assets via existing `PhotoLibraryService`
3. Each photo resized to 384x384, encoded as base64 JPEG
4. Sent to Qwen3-VL-8B at `http://100.83.43.98:1235/v1/chat/completions` (OpenAI-compatible)
5. Prompt asks model to classify into exactly ONE category
6. Results shown in review grid grouped by category
7. User reviews, then commits — app creates albums and adds photos

### Categories (default, editable in sidebar)

- Landscapes
- Pets & Animals
- Food & Drinks
- People & Portraits
- Screenshots
- Nature & Outdoors
- Architecture
- Sports & Action
- Art & Design
- Documents
- Memes & Funny
- Other

### Vision API Request

```
POST http://100.83.43.98:1235/v1/chat/completions
{
  "model": "qwen3-vl-8b",
  "messages": [
    {"role": "system", "content": "You are a photo classifier. /no_think"},
    {"role": "user", "content": [
      {"type": "text", "text": "Classify this image into exactly ONE category: Landscapes, Pets & Animals, Food & Drinks, People & Portraits, Screenshots, Nature & Outdoors, Architecture, Sports & Action, Art & Design, Documents, Memes & Funny, Other. Reply with ONLY the category name."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}
  ],
  "max_tokens": 20,
  "temperature": 0
}
```

### Key Decisions

- **One album per photo** — single best-fit category, no duplicates
- **Album prefix**: "Sorted - " (e.g., "Sorted - Landscapes") to distinguish from manual albums
- **Skip already-sorted**: Check membership in any "Sorted - *" album on re-run
- **Concurrency**: Default 8 parallel requests (network-bound, configurable 1-32)
- **Entitlement**: Add `com.apple.security.network.client` for outbound HTTP

### UI Changes

- **Toolbar**: Mode picker — `[NSFW Scan] [Sort Library]`
- **Sidebar (Sort mode)**: Category list (editable), endpoint URL, concurrency slider
- **Results (Sort mode)**: Grid grouped by category section headers, thumbnails with category badges
- **Progress**: "X of Y classified" with per-category running counts

### New Files

- `Services/VisionClassifierService.swift` — HTTP client for Qwen3-VL-8B
- `Services/SortOrchestrator.swift` — Scan engine for sort mode
- `Models/SortCategory.swift` — Category enum/model
- `Models/SortResult.swift` — Classified asset data
- `Views/SortControlView.swift` — Sidebar for sort mode
- `Views/SortResultsGridView.swift` — Grouped results grid

### Modified Files

- `NSFWScannerApp.swift` — Add SortOrchestrator to environment
- `ContentView.swift` — Add mode picker, switch between scan/sort views
- `NSFWScanner.entitlements` — Add network client entitlement
