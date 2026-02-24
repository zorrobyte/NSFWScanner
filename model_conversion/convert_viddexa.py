#!/usr/bin/env python3
"""Convert viddexa/nsfw-detection-2-mini (EfficientNet-B4, 5-class) to Core ML .mlpackage."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForImageClassification, AutoImageProcessor


class NormalizedModel(nn.Module):
    """Wraps a HuggingFace image classification model with baked-in normalization.

    CoreML ImageType delivers pixels in [0, 255] * scale + bias = [0, 1] range
    (when scale=1/255, bias=[0,0,0]).  This wrapper then applies ImageNet-style
    mean/std normalization so the backbone sees the expected input distribution.
    """

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        # Register as buffers so they move with the model and are visible to the tracer
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # x is already in [0, 1] (CoreML applied scale=1/255, bias=0)
        x = (x - self.mean) / self.std
        outputs = self.model(pixel_values=x)
        return torch.nn.functional.softmax(outputs.logits, dim=-1)


def main():
    output_path = (
        Path(__file__).parent.parent / "NSFWScanner" / "Resources" / "ViddexaNSFW.mlpackage"
    )

    # Step 1: Load model and image processor
    model_id = "viddexa/nsfw-detection-2-mini"
    print(f"1/7  Loading {model_id} via transformers...")
    hf_model = AutoModelForImageClassification.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    hf_model.eval()
    print(f"     Model loaded: {type(hf_model).__name__}")

    # Extract normalization params from the processor
    mean = processor.image_mean   # e.g. [0.485, 0.456, 0.406]
    std = processor.image_std     # e.g. [0.229, 0.224, 0.225]
    image_size = processor.size   # dict like {"height": 380, "width": 380}
    if isinstance(image_size, dict):
        h = image_size.get("height", image_size.get("shortest_edge", 380))
        w = image_size.get("width", image_size.get("shortest_edge", 380))
    else:
        h = w = int(image_size)
    print(f"     Image size: {h}x{w}")
    print(f"     Normalization mean: {mean}")
    print(f"     Normalization std:  {std}")

    # Step 2: Build class labels in index order (CRITICAL for Swift app)
    print("2/7  Building class label map...")
    id2label = hf_model.config.id2label
    num_classes = len(id2label)
    class_labels = [id2label[i] for i in range(num_classes)]
    print(f"     {num_classes} classes in index order:")
    for i, label in enumerate(class_labels):
        print(f"       [{i}] {label}")

    # Step 3: Wrap model with normalization baked in
    print("3/7  Wrapping model with baked-in normalization...")
    wrapped = NormalizedModel(hf_model, mean, std)
    wrapped.eval()

    # Step 4: Disable attention fast paths (CRITICAL for tracing)
    print("4/7  Disabling MHA fast path and SDPA...")
    torch.backends.mha.set_fastpath_enabled(False)
    sdpa_count = 0
    for module in wrapped.modules():
        if hasattr(module, "use_sdpa"):
            module.use_sdpa = False
            sdpa_count += 1
    print(f"     Disabled SDPA on {sdpa_count} modules")

    # Step 5: Trace the wrapped model
    print(f"5/7  Tracing model with input shape (1, 3, {h}, {w})...")
    example_input = torch.rand(1, 3, h, w)

    traced_model = None
    with torch.no_grad():
        try:
            traced_model = torch.jit.trace(wrapped, example_input)
            print("     Trace successful (torch.jit.trace)")
        except Exception as e:
            print(f"     torch.jit.trace failed: {e}")
            print("     Falling back to ONNX export path...")

    if traced_model is None:
        # ONNX fallback path
        onnx_path = "/tmp/viddexa.onnx"
        print(f"     Exporting to ONNX: {onnx_path}")
        with torch.no_grad():
            torch.onnx.export(
                wrapped,
                example_input,
                onnx_path,
                opset_version=13,
                input_names=["image"],
                output_names=["logits"],
            )
        print("     ONNX export successful, converting to CoreML...")

        # CoreML image input for ONNX path
        image_input = ct.ImageType(
            name="image",
            shape=(1, 3, h, w),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        )
        mlmodel = ct.convert(
            onnx_path,
            convert_to="mlprogram",
            inputs=[image_input],
            classifier_config=ct.ClassifierConfig(class_labels),
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )
    else:
        # Step 6: Define CoreML image input with baked-in [0,1] scaling
        # CoreML applies: pixel * scale + bias  =>  pixel / 255 + 0 = [0, 1]
        # NormalizedModel then applies ImageNet mean/std
        print("6/7  Defining Core ML input (scale=1/255, bias=[0,0,0])...")
        image_input = ct.ImageType(
            name="image",
            shape=(1, 3, h, w),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        )

        # Convert to CoreML
        print("     Converting to Core ML mlprogram (float16)...")
        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[image_input],
            classifier_config=ct.ClassifierConfig(class_labels),
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )

    # Step 7: Set metadata and save
    print("7/7  Setting metadata and saving...")
    mlmodel.author = f"NSFWScanner (model: {model_id})"
    mlmodel.short_description = (
        f"NSFW image classifier based on EfficientNet-B4. "
        f"5-class: {', '.join(class_labels)}. Input: {h}x{w} RGB."
    )
    mlmodel.version = "1.0"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    # Summary
    size_mb = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"\nDone! Saved to: {output_path}")
    print(f"     Size: {size_mb:.1f} MB")
    print(f"     Class labels (index order): {class_labels}")
    print(f"     Input: {h}x{w} RGB image")
    print(f"     CoreML normalization: scale=1/255, bias=[0,0,0]")
    print(f"     ImageNet normalization baked in: mean={mean}, std={std}")


if __name__ == "__main__":
    main()
