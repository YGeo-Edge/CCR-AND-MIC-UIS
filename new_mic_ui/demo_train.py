#!/usr/bin/env python3
"""
Demo training run — local machine, no EC2, no S3.
Uses 20 images per class, 1 epoch to verify the full pipeline end-to-end.

Usage:
    cd aws_training
    python demo_train.py --dataset ../Dataset/Final_dataset
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from dataset import ImageFolderDataset
from model import InternVLClassifier

CONFIG_PATH = Path(__file__).parent / "configs" / "mic2_internvl.yaml"
OUTPUT_DIR = Path(__file__).parent / "demo_output"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def run_classification_report(trainer: Trainer, dataset, classes: list, save_path: Path):
    print("\n=== Classification report on test set ===")
    output = trainer.predict(dataset)
    preds = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids

    report_str = classification_report(labels, preds, target_names=classes, digits=4, zero_division=0)
    report_dict = classification_report(labels, preds, target_names=classes, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()

    print(report_str)

    (save_path / "classification_report.txt").write_text(report_str)
    with open(save_path / "classification_report.json", "w") as f:
        json.dump({"report": report_dict, "confusion_matrix": cm, "classes": classes}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to local dataset root (e.g. ../Dataset/Final_dataset)")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--max-per-class", type=int, default=20, help="Images per class (default: 20)")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    classes = cfg["labels"]
    image_size = cfg["data"].get("image_size", 448)
    dataset_root = Path(args.dataset)

    device = pick_device()
    # bfloat16 on CUDA/MPS; float32 on CPU
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    use_bf16 = device.type in ("cuda", "mps")

    print(f"Device: {device}  dtype: {dtype}")
    print(f"Dataset: {dataset_root}")
    print(f"Samples per class: {args.max_per_class}  Epochs: {args.epochs}")
    print(f"Classes ({len(classes)}): {classes}\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_ds = ImageFolderDataset(
        root=str(dataset_root),
        classes=classes,
        image_size=image_size,
        max_per_class=args.max_per_class,
    )

    indices = list(range(len(full_ds)))
    all_labels = [full_ds.samples[i][1] for i in indices]

    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.10, stratify=all_labels, random_state=42
    )
    trainval_labels = [full_ds.samples[i][1] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.15, stratify=trainval_labels, random_state=42
    )
    print(f"\nSplit — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    m_cfg = cfg["model"]
    model = InternVLClassifier(
        model_id=m_cfg["hf_id"],
        num_classes=len(classes),
        lora_rank=m_cfg.get("lora_rank", 16),
        lora_alpha=m_cfg.get("lora_alpha", 32),
        lora_dropout=m_cfg.get("lora_dropout", 0.05),
        lora_target_modules=m_cfg.get("lora_target_modules", ["qkv", "proj"]),
        dtype=dtype,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        dataloader_num_workers=0,   # 0 for local/MPS compatibility
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Subset(full_ds, train_idx),
        eval_dataset=Subset(full_ds, val_idx),
        compute_metrics=compute_metrics,
    )

    print("\n=== Training ===")
    trainer.train()

    print("\n=== Val evaluation ===")
    val_result = trainer.evaluate()
    print(f"Val accuracy: {val_result.get('eval_accuracy', 'n/a'):.4f}")

    print("\n=== Test evaluation ===")
    test_result = trainer.evaluate(eval_dataset=Subset(full_ds, test_idx), metric_key_prefix="test")
    print(f"Test accuracy: {test_result.get('test_accuracy', 'n/a'):.4f}")

    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump({"val": val_result, "test": test_result}, f, indent=2)

    run_classification_report(trainer, Subset(full_ds, test_idx), classes, OUTPUT_DIR)

    print(f"\n=== Demo complete. Results saved to {OUTPUT_DIR}/ ===")


if __name__ == "__main__":
    main()
