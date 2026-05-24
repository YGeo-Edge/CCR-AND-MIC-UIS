#!/usr/bin/env python3
"""
Training script — runs on EC2.
Downloads dataset from S3, fine-tunes InternVLClassifier, uploads adapter to S3.

Resilience features:
  - Checkpoints synced to S3 after every save → survives spot interruption
  - Auto-resumes from latest S3 checkpoint if a previous run was cut short
  - SIGTERM handler saves a final checkpoint before the instance is reclaimed
"""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import boto3
import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from dataset import ImageFolderDataset
from model import InternVLClassifier

# Global trainer ref so the SIGTERM handler can reach it
_trainer: Trainer = None


# ── Spot-interruption / SIGTERM handler ──────────────────────────────────────

def _sigterm_handler(signum, frame):
    print("\n=== SIGTERM received — spot interruption imminent. Saving checkpoint… ===")
    if _trainer is not None:
        try:
            _trainer.save_model()
            print("Emergency checkpoint saved to local disk.")
        except Exception as e:
            print(f"Warning: could not save checkpoint: {e}")
    sys.exit(0)

signal.signal(signal.SIGTERM, _sigterm_handler)


# ── S3 checkpoint sync callback ───────────────────────────────────────────────

class S3CheckpointCallback(TrainerCallback):
    """Uploads each trainer checkpoint to S3 immediately after it is saved."""

    def __init__(self, s3_client, bucket: str, s3_prefix: str):
        self.s3 = s3_client
        self.bucket = bucket
        self.s3_prefix = s3_prefix.rstrip("/")

    def on_save(self, args, state, control, **kwargs):
        ckpt_name = f"checkpoint-{state.global_step}"
        local_ckpt = Path(args.output_dir) / ckpt_name
        if not local_ckpt.exists():
            return
        s3_ckpt = f"{self.s3_prefix}/{ckpt_name}"
        print(f"\n  → Syncing {ckpt_name} to s3://{self.bucket}/{s3_ckpt}/")
        for f in local_ckpt.rglob("*"):
            if f.is_file():
                self.s3.upload_file(str(f), self.bucket,
                                    f"{s3_ckpt}/{f.relative_to(local_ckpt)}")
        # Update pointer so resume knows which checkpoint is latest
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.s3_prefix}/latest.txt",
            Body=ckpt_name.encode(),
        )
        print(f"  → {ckpt_name} synced.")


# ── S3 helpers ────────────────────────────────────────────────────────────────

def s3_sync_dataset(bucket: str, prefix: str, classes: list, local_dir: Path):
    """Download all class folders in parallel using aws s3 sync (much faster than boto3 per-file)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    s3_root = f"s3://{bucket}/{prefix.rstrip('/')}"
    for cls in classes:
        local_cls = local_dir / cls
        local_cls.mkdir(parents=True, exist_ok=True)
        src = f"{s3_root}/{cls}/"
        print(f"  syncing {cls} …", flush=True)
        subprocess.run(
            ["aws", "s3", "sync", src, str(local_cls), "--quiet"],
            check=True,
        )
        count = sum(1 for _ in local_cls.iterdir())
        print(f"  {cls}: {count} files", flush=True)


def s3_upload_dir(s3, local_dir: str, bucket: str, prefix: str):
    for f in Path(local_dir).rglob("*"):
        if f.is_file():
            key = f"{prefix.rstrip('/')}/{f.relative_to(local_dir)}"
            print(f"  uploading {f.name} …")
            s3.upload_file(str(f), bucket, key)


def s3_verify_uploads(s3, bucket: str, prefix: str, local_dir: str) -> bool:
    """Verify every local file was successfully uploaded to S3; print a manifest."""
    print("=== Verifying S3 uploads ===")
    ok = True
    for f in sorted(Path(local_dir).rglob("*")):
        if not f.is_file():
            continue
        key = f"{prefix.rstrip('/')}/{f.relative_to(local_dir)}"
        try:
            resp = s3.head_object(Bucket=bucket, Key=key)
            size = resp["ContentLength"]
            print(f"  OK  {f.name:45s}  {size:>12,} bytes")
        except Exception as e:
            print(f"  MISSING  {f.name} — {e}")
            ok = False
    if ok:
        print("=== All artifacts verified on S3 ===")
    else:
        print("=== WARNING: some artifacts missing from S3 ===")
    return ok


def s3_download_checkpoint(s3, bucket: str, s3_prefix: str, local_dir: Path) -> str | None:
    """Download the latest S3 checkpoint if one exists. Returns local path or None."""
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"{s3_prefix}/latest.txt")
        ckpt_name = resp["Body"].read().decode().strip()
    except Exception:
        return None

    s3_ckpt = f"{s3_prefix}/{ckpt_name}"
    local_ckpt = local_dir / ckpt_name
    local_ckpt.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    objects = [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_ckpt + "/")
        for obj in page.get("Contents", [])
    ]
    if not objects:
        return None

    print(f"=== Resuming from {ckpt_name} ({len(objects)} files) ===")
    for key in objects:
        rel = key[len(s3_ckpt) + 1:]
        dest = local_ckpt / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))

    return str(local_ckpt)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def run_classification_report(trainer: Trainer, dataset, classes: list, save_path: str):
    print("=== Classification report on test set ===")
    output = trainer.predict(dataset)
    preds = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids

    report_str = classification_report(labels, preds, target_names=classes, digits=4, zero_division=0)
    report_dict = classification_report(labels, preds, target_names=classes, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()

    print(report_str)

    with open(os.path.join(save_path, "classification_report.txt"), "w") as f:
        f.write(report_str)
    with open(os.path.join(save_path, "classification_report.json"), "w") as f:
        json.dump({"report": report_dict, "confusion_matrix": cm, "classes": classes}, f, indent=2)

    print(f"Classification report saved to {save_path}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _trainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bucket", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bucket = args.bucket or cfg["data"]["bucket"]
    job_name = cfg["job_name"]
    classes = cfg["labels"]
    model_id = cfg["model"]["hf_id"]
    image_size = cfg["data"].get("image_size", 448)
    test_split = cfg["data"].get("test_split", 0.10)
    val_split = cfg["data"].get("val_split", 0.15)
    data_prefix = cfg["data"]["dataset_prefix"]
    train_cfg = cfg["training"]
    output_prefix = cfg["output"]["prefix"].rstrip("/")

    local_data = Path("/tmp/dataset")
    local_model = Path(f"/tmp/models/{job_name}")
    ckpt_s3_prefix = f"jobs/{job_name}/checkpoints"
    s3 = boto3.client("s3")

    # ── 1. Download dataset ──────────────────────────────────────────────────
    print("=== Downloading dataset ===")
    s3_sync_dataset(bucket, data_prefix, classes, local_data)

    # ── 2. Build split datasets ──────────────────────────────────────────────
    print("=== Building datasets ===")
    full_ds = ImageFolderDataset(
        root=str(local_data),
        classes=classes,
        image_size=image_size,
        max_per_class=cfg["data"].get("max_per_class"),
    )
    indices = list(range(len(full_ds)))
    all_labels = [full_ds.samples[i][1] for i in indices]

    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_split, stratify=all_labels, random_state=42
    )
    trainval_labels = [full_ds.samples[i][1] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_split, stratify=trainval_labels, random_state=42
    )
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    # ── 3. Check for existing S3 checkpoint (resume after spot interruption) ──
    resume_from = s3_download_checkpoint(s3, bucket, ckpt_s3_prefix, local_model)

    # ── 4. Build model ───────────────────────────────────────────────────────
    print("=== Loading model ===")
    m_cfg = cfg["model"]
    model = InternVLClassifier(
        model_id=model_id,
        num_classes=len(classes),
        lora_rank=m_cfg.get("lora_rank", 16),
        lora_alpha=m_cfg.get("lora_alpha", 32),
        lora_dropout=m_cfg.get("lora_dropout", 0.05),
        lora_target_modules=m_cfg.get("lora_target_modules", ["qkv", "proj"]),
    )

    if train_cfg.get("gradient_checkpointing", True):
        try:
            model.vision_model.gradient_checkpointing_enable()
        except Exception:
            pass

    # ── 5. Train ─────────────────────────────────────────────────────────────
    print("=== Training ===")
    training_args = TrainingArguments(
        output_dir=str(local_model),
        num_train_epochs=train_cfg.get("epochs", 5),
        per_device_train_batch_size=train_cfg.get("batch_size", 8),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 16),
        gradient_accumulation_steps=train_cfg.get("grad_accum", 1),
        learning_rate=train_cfg.get("lr", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        bf16=True,
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=train_cfg.get("logging_steps", 50),
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        report_to="none",
        remove_unused_columns=False,
    )

    _trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Subset(full_ds, train_idx),
        eval_dataset=Subset(full_ds, val_idx),
        compute_metrics=compute_metrics,
        callbacks=[S3CheckpointCallback(s3, bucket, ckpt_s3_prefix)],
    )
    _trainer.train(resume_from_checkpoint=resume_from)

    # ── 6. Save adapter ──────────────────────────────────────────────────────
    print("=== Saving adapter ===")
    adapter_path = str(local_model / "best_adapter")
    model.save_adapter(adapter_path, classes)

    val_result = _trainer.evaluate()
    print(f"Val:  {val_result}")

    print("=== Test evaluation ===")
    test_result = _trainer.evaluate(eval_dataset=Subset(full_ds, test_idx), metric_key_prefix="test")
    print(f"Test: {test_result}")

    results = {"val": val_result, "test": test_result}
    with open(os.path.join(adapter_path, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    run_classification_report(_trainer, Subset(full_ds, test_idx), classes, adapter_path)

    # ── 7. Upload final model to S3 ───────────────────────────────────────────
    print("=== Uploading model ===")
    s3_output_prefix = f"{output_prefix}/{job_name}"
    s3_upload_dir(s3, adapter_path, bucket, s3_output_prefix)

    # Verify every file landed on S3 before the instance can shut down
    verified = s3_verify_uploads(s3, bucket, s3_output_prefix, adapter_path)
    if not verified:
        raise RuntimeError("S3 upload verification failed — NOT shutting down.")

    print(f"\n=== Done. Model at s3://{bucket}/{s3_output_prefix}/ ===")
    print(f"Download with:  python train_cli.py download --config <config.yaml>")


if __name__ == "__main__":
    main()
