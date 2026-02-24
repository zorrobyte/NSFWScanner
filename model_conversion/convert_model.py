#!/usr/bin/env python3
"""Convert Marqo/nsfw-image-detection-384 (ViT-Tiny) to Core ML .mlpackage."""

import sys
from pathlib import Path

import coremltools as ct
import torch
import timm


def main():
    output_path = Path(__file__).parent.parent / "FamilySafe-NSFW-Finder" / "Resources" / "NSFWClassifier.mlpackage"

    # Step 1: Load model via timm
    print("1/6  Loading Marqo/nsfw-image-detection-384 via timm...")
    model = timm.create_model("hf_hub:Marqo/nsfw-image-detection-384", pretrained=True)
    model.eval()
    print(f"     Model class count: {model.num_classes}")

    # Step 2: Disable attention fast paths (CRITICAL for tracing)
    print("2/6  Disabling MHA fast path and SDPA...")
    torch.backends.mha.set_fastpath_enabled(False)
    sdpa_count = 0
    for module in model.modules():
        if hasattr(module, "use_sdpa"):
            module.use_sdpa = False
            sdpa_count += 1
    print(f"     Disabled SDPA on {sdpa_count} modules")

    # Step 3: Trace with fixed-shape input
    print("3/6  Tracing model with input shape (1, 3, 384, 384)...")
    example_input = torch.rand(1, 3, 384, 384)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    print("     Trace successful")

    # Step 4: Define Core ML image input with baked-in normalization
    # PyTorch expects: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1.0
    # CoreML applies:  pixel * scale + bias
    # So: scale = 1/127.5, bias = -1.0 per channel
    print("4/6  Defining Core ML input with baked-in normalization...")
    scale = 1.0 / 127.5
    bias = [-1.0, -1.0, -1.0]
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, 384, 384),
        scale=scale,
        bias=bias,
        color_layout=ct.colorlayout.RGB,
    )

    # Step 5: Convert to Core ML with classifier config
    print("5/6  Converting to Core ML mlprogram (float16)...")
    class_labels = ["NSFW", "SFW"]  # Index 0 = NSFW, Index 1 = SFW
    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(class_labels),
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    # Step 6: Set metadata and save
    print("6/6  Setting metadata and saving...")
    mlmodel.author = "FamilySafe (model: Marqo/nsfw-image-detection-384)"
    mlmodel.short_description = (
        "NSFW image classifier based on ViT-Tiny (384x384). "
        "98.56% accuracy. Classes: NSFW, SFW."
    )
    mlmodel.version = "1.0"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    # Summary
    size_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\nDone! Saved to: {output_path}")
    print(f"     Size: {size_mb:.1f} MB")
    print(f"     Class labels: {class_labels}")
    print(f"     Input: 384x384 RGB image")
    print(f"     Normalization: baked in (scale={scale:.6f}, bias={bias})")


if __name__ == "__main__":
    main()
