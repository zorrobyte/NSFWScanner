#!/usr/bin/env python3
"""Convert Falconsai/nsfw_image_detection (ViT-Base-224) to Core ML .mlpackage."""

import sys
from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn
from transformers import ViTForImageClassification


# ImageNet normalization constants (used by Falconsai/nsfw_image_detection processor)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class NormalizedModel(nn.Module):
    """Wraps a ViT classifier and bakes in ImageNet normalization.

    Accepts a float tensor in [0, 1] (after CoreML scales raw uint8 pixels by
    1/255) and applies (x - mean) / std before forwarding to the base model.
    """

    def __init__(self, base_model: nn.Module, mean: list, std: list):
        super().__init__()
        self.base_model = base_model
        # Register as buffers so they travel with the model state but are not
        # treated as learnable parameters.
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.base_model(x).logits


def main():
    output_path = (
        Path(__file__).parent.parent
        / "NSFWScanner"
        / "Resources"
        / "FalconsaiNSFW.mlpackage"
    )

    # ------------------------------------------------------------------ #
    # Step 1: Load the HuggingFace model
    # ------------------------------------------------------------------ #
    print("1/6  Loading Falconsai/nsfw_image_detection via transformers...")
    hf_model = ViTForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection"
    )
    hf_model.eval()
    num_classes = hf_model.config.num_labels
    id2label = hf_model.config.id2label          # e.g. {0: "normal", 1: "nsfw"}
    class_labels = [id2label[i] for i in range(num_classes)]
    print(f"     Classes ({num_classes}): {class_labels}")

    # ------------------------------------------------------------------ #
    # Step 2: Wrap with normalization & disable attention fast paths
    # ------------------------------------------------------------------ #
    print("2/6  Wrapping model with ImageNet normalization...")
    wrapped = NormalizedModel(hf_model, IMAGENET_MEAN, IMAGENET_STD)
    wrapped.eval()

    print("     Disabling MHA fast path and SDPA for tracing compatibility...")
    torch.backends.mha.set_fastpath_enabled(False)
    sdpa_count = 0
    for module in wrapped.modules():
        if hasattr(module, "use_sdpa"):
            module.use_sdpa = False
            sdpa_count += 1
    print(f"     Disabled SDPA on {sdpa_count} modules")

    # ------------------------------------------------------------------ #
    # Step 3: Trace with fixed-shape input
    # ------------------------------------------------------------------ #
    print("3/6  Tracing model with input shape (1, 3, 224, 224)...")
    # Input to NormalizedModel is in [0, 1] (CoreML will scale pixels by 1/255)
    example_input = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped, example_input)
    print("     Trace successful")

    # ------------------------------------------------------------------ #
    # Step 4: Define Core ML image input
    # CoreML applies:  pixel * scale + bias  before passing to the model.
    # We want the model to receive values in [0, 1] so NormalizedModel can
    # apply per-channel ImageNet normalization itself.
    # Therefore: scale = 1/255, bias = [0, 0, 0]
    # ------------------------------------------------------------------ #
    print("4/6  Defining Core ML image input (scale=1/255, bias=[0,0,0])...")
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, 224, 224),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Convert to Core ML mlprogram with float16 precision
    # ------------------------------------------------------------------ #
    print("5/6  Converting to Core ML mlprogram (float16)...")
    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(class_labels),
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    # ------------------------------------------------------------------ #
    # Step 6: Set metadata and save
    # ------------------------------------------------------------------ #
    print("6/6  Setting metadata and saving...")
    mlmodel.author = "NSFWScanner (model: Falconsai/nsfw_image_detection)"
    mlmodel.short_description = (
        "NSFW image classifier based on ViT-Base-patch16-224. "
        "Classes: normal, nsfw. "
        "ImageNet normalization baked in."
    )
    mlmodel.version = "1.0"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    size_mb = (
        sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        / (1024 * 1024)
    )
    print(f"\nDone! Saved to: {output_path}")
    print(f"     Size: {size_mb:.1f} MB")
    print(f"     Class labels: {class_labels}")
    print(f"     Input: 224x224 RGB image")
    print(f"     Normalization: baked in via NormalizedModel wrapper")
    print(f"       mean={IMAGENET_MEAN}, std={IMAGENET_STD}")


if __name__ == "__main__":
    main()
