#!/usr/bin/env python3
"""
Scan all images in Dataset/nn_mlcs with the trained classifier.
Produces:
  - nn_mlcs_results.json  : full per-image results
  - nn_mlcs_report.md     : human-readable summary + malicious detections
"""
import sys, json, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

HERE    = Path(__file__).parent
ADAPTER = HERE / "trained_models/mic2_internvl_v1"
FOLDER  = HERE.parent / "Dataset/nn_mlcs"
OUT_JSON = HERE / "nn_mlcs_results.json"
OUT_MD   = HERE / "nn_mlcs_report.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 32

CLASSES = [
    "Benign","fake_av","financial_scam","misleading_offers","fake_appstore",
    "tech_support_scam","gift_card_scan","forced_notification","suspicious_vpn",
    "malicious_extension","fake_updates","blank_LP","fake_downloader",
]
BENIGN_CLASSES = {"Benign", "blank_LP"}

# Per-class thresholds (FPR < 0.1%) from threshold analysis
THRESHOLDS = {
    "fake_appstore":        0.050,
    "blank_LP":             0.050,
    "gift_card_scan":       0.350,
    "malicious_extension":  0.800,
    "tech_support_scam":    0.990,
    "misleading_offers":    0.990,
    "suspicious_vpn":       0.990,
    "fake_downloader":      0.990,
    "fake_av":              0.999,
    "financial_scam":       0.999,
    "forced_notification":  0.999,
    "fake_updates":         0.999,
}

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def run_inference(model, dev, tf, image_paths):
    results = []
    batch_imgs, batch_paths = [], []
    total = len(image_paths)
    done  = 0

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(dev)
        with torch.no_grad():
            logits = model(pixel_values=x).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        for p, path in zip(probs, batch_paths):
            pred_idx  = int(np.argmax(p))
            pred_cls  = CLASSES[pred_idx]
            pred_conf = float(p[pred_idx])
            is_benign = pred_cls in BENIGN_CLASSES
            # two-stage: stage2 only if malicious
            threshold = THRESHOLDS.get(pred_cls, 0.500)
            if is_benign:
                stage1 = "BENIGN"
                stage2 = None
                decision = "pass"
            elif pred_conf >= threshold:
                stage1 = "MALICIOUS"
                stage2 = pred_cls
                decision = "block"
            else:
                stage1 = "MALICIOUS"
                stage2 = pred_cls
                decision = "review"

            # load source_url from json if available
            json_path = path.with_suffix(".json")
            source_url = ""
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text())
                    source_url = meta.get("metadata", {}).get("source_url", "")
                except Exception:
                    pass

            results.append({
                "file":       path.name,
                "source_url": source_url,
                "pred_cls":   pred_cls,
                "pred_conf":  round(pred_conf, 6),
                "stage1":     stage1,
                "stage2":     stage2,
                "decision":   decision,
                "scores":     {c: round(float(p[i]), 6) for i, c in enumerate(CLASSES)},
            })
        batch_imgs.clear()
        batch_paths.clear()

    for path in image_paths:
        try:
            img = Image.open(path)
            batch_imgs.append(tf(img))
        except Exception:
            batch_imgs.append(torch.zeros(3, 448, 448))
        batch_paths.append(path)
        done += 1
        if len(batch_imgs) == BATCH_SIZE:
            flush()
            print(f"\r  {done}/{total} ({done/total:.0%})", end="", flush=True)
    flush()
    print(f"\r  {done}/{total} — done          ")
    return results

def build_report(results):
    total    = len(results)
    benign   = [r for r in results if r["stage1"] == "BENIGN"]
    malicious = [r for r in results if r["stage1"] == "MALICIOUS"]
    block    = [r for r in malicious if r["decision"] == "block"]
    review   = [r for r in malicious if r["decision"] == "review"]

    # count by class
    class_counts = Counter(r["pred_cls"] for r in malicious)

    lines = []
    lines.append("# nn_mlcs Folder — Inference Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Count | % |")
    lines.append(f"|--------|-------|---|")
    lines.append(f"| Total images scanned | {total} | 100% |")
    lines.append(f"| Classified BENIGN | {len(benign)} | {len(benign)/total:.1%} |")
    lines.append(f"| Classified MALICIOUS | {len(malicious)} | {len(malicious)/total:.1%} |")
    lines.append(f"| → Block (high-confidence) | {len(block)} | {len(block)/total:.1%} |")
    lines.append(f"| → Review (below threshold) | {len(review)} | {len(review)/total:.1%} |")
    lines.append("")

    if malicious:
        lines.append("## Malicious Detections by Class")
        lines.append("")
        lines.append("| Class | Count | Avg Conf | Block | Review |")
        lines.append("|-------|-------|----------|-------|--------|")
        for cls, cnt in class_counts.most_common():
            cls_items = [r for r in malicious if r["pred_cls"] == cls]
            avg_conf  = np.mean([r["pred_conf"] for r in cls_items])
            n_block   = sum(1 for r in cls_items if r["decision"] == "block")
            n_review  = sum(1 for r in cls_items if r["decision"] == "review")
            lines.append(f"| {cls} | {cnt} | {avg_conf:.3f} | {n_block} | {n_review} |")
        lines.append("")

    # Per-decision detail
    for section_label, items in [("### BLOCK — High-Confidence Malicious", block),
                                   ("### REVIEW — Below Threshold (Needs Human Check)", review)]:
        if not items:
            continue
        lines.append(section_label)
        lines.append("")
        lines.append("| # | File | Source URL | Class | Conf | Threshold | Top-2 classes |")
        lines.append("|---|------|-----------|-------|------|-----------|---------------|")
        for i, r in enumerate(sorted(items, key=lambda x: -x["pred_conf"]), 1):
            top2 = sorted(r["scores"].items(), key=lambda x: -x[1])[:2]
            top2_str = " | ".join(f"{c}: {v:.3f}" for c, v in top2)
            thresh = THRESHOLDS.get(r["pred_cls"], 0.500)
            lines.append(
                f"| {i} | `{r['file']}` | {r['source_url']} "
                f"| {r['pred_cls']} | {r['pred_conf']:.4f} | {thresh:.3f} | {top2_str} |"
            )
        lines.append("")

    lines.append("## Benign Detections")
    lines.append("")
    benign_cls_counts = Counter(r["pred_cls"] for r in benign)
    lines.append("| Class | Count |")
    lines.append("|-------|-------|")
    for cls, cnt in benign_cls_counts.most_common():
        lines.append(f"| {cls} | {cnt} |")
    lines.append("")

    return "\n".join(lines)

def main():
    print("=== Loading model ===")
    sys.path.insert(0, str(HERE))
    from model import InternVLClassifier
    dev = get_device()
    print(f"Device: {dev}")
    model = InternVLClassifier.load_adapter(str(ADAPTER), dtype=torch.float32)
    model = model.to(dev).eval()

    print("\n=== Scanning nn_mlcs folder ===")
    image_paths = sorted(
        p for p in FOLDER.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Found {len(image_paths)} images")

    print("\n=== Running inference ===")
    tf = build_transform()
    t0 = time.time()
    results = run_inference(model, dev, tf, image_paths)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({elapsed/len(results)*1000:.1f} ms/image)")

    # Save JSON
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results → {OUT_JSON}")

    # Build and save report
    report = build_report(results)
    OUT_MD.write_text(report)
    print(f"Saved report  → {OUT_MD}")

    # Quick console summary
    benign   = sum(1 for r in results if r["stage1"] == "BENIGN")
    malicious = sum(1 for r in results if r["stage1"] == "MALICIOUS")
    block    = sum(1 for r in results if r["decision"] == "block")
    review   = sum(1 for r in results if r["decision"] == "review")
    print(f"\n{'='*50}")
    print(f"Total:     {len(results)}")
    print(f"BENIGN:    {benign}  ({benign/len(results):.1%})")
    print(f"MALICIOUS: {malicious}  ({malicious/len(results):.1%})")
    print(f"  → BLOCK:  {block}")
    print(f"  → REVIEW: {review}")

if __name__ == "__main__":
    main()
