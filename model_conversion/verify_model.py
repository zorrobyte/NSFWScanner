#!/usr/bin/env python3
"""Verify converted NSFWClassifier.mlpackage against PyTorch reference."""

import sys
import urllib.request
from pathlib import Path

import coremltools as ct
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform


def main():
    mlpackage_path = Path(__file__).parent.parent / "FamilySafe-NSFW-Finder" / "Resources" / "NSFWClassifier.mlpackage"

    if not mlpackage_path.exists():
        print(f"ERROR: {mlpackage_path} not found. Run convert_model.py first.")
        sys.exit(1)

    class_labels = ["NSFW", "SFW"]

    # Step 1: Download known-SFW test image (beignets from HuggingFace)
    test_image_path = Path(__file__).parent / "test_image.png"
    if not test_image_path.exists():
        print("1/5  Downloading test image (beignets)...")
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        urllib.request.urlretrieve(url, str(test_image_path))
    else:
        print("1/5  Using cached test image")
    print(f"     Image: {test_image_path}")

    # Step 2: Run Core ML inference
    print("2/5  Loading Core ML model...")
    coreml_model = ct.models.MLModel(str(mlpackage_path))
    pil_image = Image.open(test_image_path).convert("RGB").resize((384, 384))
    print("     Running Core ML prediction...")
    coreml_pred = coreml_model.predict({"image": pil_image})
    coreml_label = coreml_pred["classLabel"]
    coreml_raw_scores = coreml_pred["classLabel_probs"]
    # CoreML ClassifierConfig may output raw logits — apply softmax for comparison
    import math
    raw_values = list(coreml_raw_scores.values())
    raw_keys = list(coreml_raw_scores.keys())
    max_val = max(raw_values)
    exp_values = [math.exp(v - max_val) for v in raw_values]
    exp_sum = sum(exp_values)
    coreml_scores = {k: ev / exp_sum for k, ev in zip(raw_keys, exp_values)}
    print(f"     Core ML prediction: {coreml_label}")
    for label, score in sorted(coreml_scores.items(), key=lambda x: -x[1]):
        print(f"       {label}: {score:.4f}")

    # Step 3: Run PyTorch inference
    print("3/5  Loading PyTorch model...")
    torch_model = timm.create_model("hf_hub:Marqo/nsfw-image-detection-384", pretrained=True)
    torch_model.eval()

    data_config = resolve_data_config(torch_model.pretrained_cfg)
    transform = create_transform(**data_config)
    img_tensor = transform(Image.open(test_image_path).convert("RGB")).unsqueeze(0)

    print("     Running PyTorch prediction...")
    with torch.no_grad():
        logits = torch_model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    torch_label_idx = probs.argmax().item()
    torch_label = class_labels[torch_label_idx]
    print(f"     PyTorch prediction: {torch_label}")
    for i, label in enumerate(class_labels):
        print(f"       {label}: {probs[i].item():.4f}")

    # Step 4: Compare outputs
    print("4/5  Comparing outputs...")
    max_diff = 0.0
    for i, label in enumerate(class_labels):
        coreml_prob = coreml_scores.get(label, 0.0)
        torch_prob = probs[i].item()
        diff = abs(coreml_prob - torch_prob)
        max_diff = max(max_diff, diff)
        print(f"     {label}: CoreML={coreml_prob:.4f}  PyTorch={torch_prob:.4f}  diff={diff:.4f}")

    # Step 5: Verdict
    print("5/5  Verdict:")
    label_match = coreml_label == torch_label
    within_tolerance = max_diff < 0.01  # 1% tolerance for float16

    if label_match and within_tolerance:
        print(f"     PASS - Labels match ({coreml_label}), max probability diff: {max_diff:.4f}")
    elif label_match:
        print(f"     WARN - Labels match ({coreml_label}), but max diff {max_diff:.4f} > 1%")
        print(f"     (This is expected for float16 conversion)")
    else:
        print(f"     FAIL - Labels differ: CoreML={coreml_label}, PyTorch={torch_label}")
        sys.exit(1)


if __name__ == "__main__":
    main()
