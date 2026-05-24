#!/usr/bin/env python3
"""Quick local inference test — verifies the downloaded adapter loads and predicts correctly."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ── image transform (must match training) ─────────────────────────────────────
def build_transform(image_size=448):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="trained_models/mic2_internvl_v1",
                        help="Path to adapter directory")
    parser.add_argument("--image", required=True, help="Image file to classify")
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()

    adapter_path = str(Path(args.adapter).expanduser().resolve())
    image_path   = str(Path(args.image).expanduser().resolve())

    # Load meta
    with open(Path(adapter_path) / "model_meta.json") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    print(f"Model:   {meta['model_id']}")
    print(f"Classes: {class_names}\n")

    device = get_device()
    print(f"Device:  {device}")

    # Load model — use float32 for CPU/MPS compatibility
    print("Loading adapter …")
    sys.path.insert(0, str(Path(__file__).parent))
    from model import InternVLClassifier

    t0 = time.time()
    model = InternVLClassifier.load_adapter(
        adapter_path,
        dtype=torch.float32,
    )
    model = model.to(device)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # Load + preprocess image
    img = Image.open(image_path).convert("RGB")
    tf  = build_transform(args.image_size)
    pixel_values = tf(img).unsqueeze(0).to(device)

    # Inference
    print(f"Running inference on: {image_path}")
    t1 = time.time()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    elapsed = time.time() - t1

    probs = F.softmax(outputs.logits, dim=-1)[0]
    top5  = probs.topk(min(5, len(class_names)))

    pred_idx   = probs.argmax().item()
    pred_class = class_names[pred_idx]
    pred_conf  = probs[pred_idx].item()

    print(f"\n{'='*50}")
    print(f"  PREDICTION:  {pred_class}")
    print(f"  CONFIDENCE:  {pred_conf:.1%}")
    print(f"  LATENCY:     {elapsed*1000:.0f} ms")
    print(f"{'='*50}")
    print("\nTop-5 predictions:")
    for i, (idx, prob) in enumerate(zip(top5.indices.tolist(), top5.values.tolist()), 1):
        bar = "█" * int(prob * 30)
        print(f"  {i}. {class_names[idx]:25s} {prob:6.1%}  {bar}")

    print("\nInference test PASSED." if pred_conf > 0.5 else "\nWARNING: low confidence prediction.")


if __name__ == "__main__":
    main()
