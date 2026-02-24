#!/usr/bin/env python3
"""Convert AdamCodd/vit-base-nsfw-detector (ViT-Base-384, 2 classes) to Core ML .mlpackage."""

import sys
from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor


class NormalizedModel(nn.Module):
    """Wraps a HuggingFace image classification model and bakes in normalization.

    The CoreML ImageType input provides raw [0, 255] pixels scaled to [0.0, 1.0]
    via scale=1/255.  This wrapper then applies the per-channel mean/std
    normalization that the original image processor requires, so the host app
    never has to touch pixel values.
    """

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        # Register as buffers so they are preserved during tracing
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # x arrives in [0, 1] range (after CoreML scale=1/255)
        x = (x - self.mean) / self.std
        output = self.model(pixel_values=x)
        return output.logits


def main():
    output_path = (
        Path(__file__).parent.parent
        / "NSFWScanner"
        / "Resources"
        / "AdamCoddNSFW.mlpackage"
    )

    model_id = "AdamCodd/vit-base-nsfw-detector"

    # Step 1: Load model and image processor
    print(f"1/6  Loading {model_id} via transformers...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    hf_model = AutoModelForImageClassification.from_pretrained(model_id)
    hf_model.eval()

    # Extract normalization parameters from the image processor
    mean = processor.image_mean  # e.g. [0.5, 0.5, 0.5]
    std = processor.image_std    # e.g. [0.5, 0.5, 0.5]
    print(f"     Classes: {hf_model.config.id2label}")
    print(f"     Normalization mean: {mean}")
    print(f"     Normalization std:  {std}")

    # Step 2: Wrap model with normalization and disable attention fast paths
    print("2/6  Wrapping model with normalization and disabling MHA/SDPA fast paths...")
    wrapped_model = NormalizedModel(hf_model, mean, std)
    wrapped_model.eval()

    torch.backends.mha.set_fastpath_enabled(False)

    sdpa_count = 0
    for module in wrapped_model.modules():
        # HuggingFace ViT uses scaled_dot_product_attention which needs disabling
        if hasattr(module, "use_sdpa"):
            module.use_sdpa = False
            sdpa_count += 1

    # Also disable the attention implementation selection used in newer transformers
    for module in wrapped_model.modules():
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = "eager"

    print(f"     Disabled SDPA on {sdpa_count} modules")

    # Step 3: Trace with fixed-shape input
    print("3/6  Tracing model with input shape (1, 3, 384, 384)...")
    # Input is [0, 1] (after CoreML applies scale=1/255 to the raw pixel bytes)
    example_input = torch.rand(1, 3, 384, 384)
    with torch.no_grad():
        # Verify forward pass before tracing
        test_out = wrapped_model(example_input)
        print(f"     Forward-pass logits shape: {test_out.shape}")

        traced_model = torch.jit.trace(wrapped_model, example_input)
    print("     Trace successful")

    # Step 4: Define Core ML image input
    # CoreML applies:  output = pixel_uint8 * scale + bias
    # We want output in [0, 1], so: scale = 1/255, bias = [0, 0, 0]
    # The NormalizedModel wrapper then handles the mean/std normalization.
    print("4/6  Defining Core ML input with scale=1/255, bias=[0,0,0]...")
    scale = 1.0 / 255.0
    bias = [0.0, 0.0, 0.0]
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, 384, 384),
        scale=scale,
        bias=bias,
        color_layout=ct.colorlayout.RGB,
    )

    # Step 5: Build class labels from model config
    id2label = hf_model.config.id2label
    # id2label is a dict {0: "label0", 1: "label1", ...}; build ordered list
    class_labels = [id2label[i] for i in sorted(id2label.keys())]
    print(f"     Class labels (ordered): {class_labels}")

    # Step 6: Convert to Core ML
    print("5/6  Converting to Core ML mlprogram (float16)...")
    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(class_labels),
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    # Step 7: Set metadata and save
    print("6/6  Setting metadata and saving...")
    mlmodel.author = f"NSFWScanner (model: {model_id})"
    mlmodel.short_description = (
        "NSFW image classifier based on ViT-Base (384x384). "
        f"Classes: {', '.join(class_labels)}."
    )
    mlmodel.version = "1.0"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    # Summary
    size_mb = (
        sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        / (1024 * 1024)
    )
    print(f"\nDone! Saved to: {output_path}")
    print(f"     Size:          {size_mb:.1f} MB")
    print(f"     Class labels:  {class_labels}")
    print(f"     Input:         384x384 RGB image")
    print(f"     Normalization: baked in (mean={mean}, std={std})")


if __name__ == "__main__":
    main()
