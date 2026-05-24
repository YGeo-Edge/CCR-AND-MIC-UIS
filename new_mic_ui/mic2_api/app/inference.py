"""
Inference engine: model loading, batch worker, classification logic.

Design
------
- One model instance, loaded once at startup.
- Async batch worker: collects pending requests up to BATCH_SIZE or
  BATCH_TIMEOUT_MS, then runs a single forward pass for the whole batch.
- Inference runs in a ThreadPoolExecutor so it never blocks the event loop.
- Each submitted image is a Future; the worker resolves it when done.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CLASSES = [
    "Benign", "fake_av", "financial_scam", "misleading_offers", "fake_appstore",
    "tech_support_scam", "gift_card_scan", "forced_notification", "suspicious_vpn",
    "malicious_extension", "fake_updates", "blank_LP", "fake_downloader",
]
BENIGN_CLASSES = {"Benign", "blank_LP"}

# Per-class auto-block thresholds (FPR < 0.1% from threshold analysis)
THRESHOLDS: dict[str, float] = {
    "fake_appstore":       0.050,
    "blank_LP":            0.050,
    "gift_card_scan":      0.350,
    "malicious_extension": 0.800,
    "tech_support_scam":   0.990,
    "misleading_offers":   0.990,
    "suspicious_vpn":      0.990,
    "fake_downloader":     0.990,
    "fake_av":             0.999,
    "financial_scam":      0.999,
    "forced_notification": 0.999,
    "fake_updates":        0.999,
}

BATCH_SIZE      = int(os.getenv("BATCH_SIZE",       "8"))
BATCH_TIMEOUT_S = float(os.getenv("BATCH_TIMEOUT_MS", "50")) / 1000.0


@dataclass
class PendingItem:
    image_bytes: bytes
    future:      asyncio.Future
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _classify(probs: np.ndarray) -> dict:
    pred_idx  = int(np.argmax(probs))
    pred_cls  = CLASSES[pred_idx]
    pred_conf = float(probs[pred_idx])
    is_benign = pred_cls in BENIGN_CLASSES

    threshold = THRESHOLDS.get(pred_cls, 0.500)
    if is_benign:
        verdict  = "BENIGN"
        decision = "pass"
    elif pred_conf >= threshold:
        verdict  = "MALICIOUS"
        decision = "block"
    else:
        verdict  = "MALICIOUS"
        decision = "review"

    top3 = sorted(
        [{"label": c, "confidence": round(float(probs[i]), 6)} for i, c in enumerate(CLASSES)],
        key=lambda x: -x["confidence"],
    )[:3]

    all_scores = {c: round(float(probs[i]), 6) for i, c in enumerate(CLASSES)}

    return {
        "binary": {
            "verdict":    verdict,
            "confidence": round(pred_conf, 6),
        },
        "multiclass": {
            "label":      pred_cls,
            "confidence": round(pred_conf, 6),
            "decision":   decision,
            "threshold":  threshold,
            "top3":       top3,
        },
        "all_scores": all_scores,
    }


class InferenceEngine:
    def __init__(self, adapter_path: str):
        self.adapter_path = adapter_path
        self.model        = None
        self.device       = _get_device()
        self.transform    = _build_transform()
        self._executor    = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._queue: asyncio.Queue[PendingItem] = None   # set in start()
        self._worker_task = None
        self.loaded       = False

        # Metrics
        self._start_time      = time.monotonic()
        self._total           = 0
        self._completed       = 0
        self._failed          = 0
        self._latencies: deque = deque(maxlen=10_000)
        self._batch_sizes: deque = deque(maxlen=1_000)
        self._recent_completions: deque = deque(maxlen=10_000)  # timestamps

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load_model(self):
        """Blocking — call from a thread at startup."""
        adapter_dir = os.path.dirname(os.path.abspath(__file__))
        # model.py lives one level up (copied there during Docker build)
        sys.path.insert(0, os.path.join(adapter_dir, ".."))
        from model import InternVLClassifier  # noqa: PLC0415

        dtype = torch.float32  # float16 not supported on MPS/CPU reliably
        print(f"Loading model from {self.adapter_path} on {self.device} …")
        self.model = InternVLClassifier.load_adapter(self.adapter_path, dtype=dtype)
        self.model = self.model.to(self.device).eval()
        self.loaded = True
        print("Model ready.")

    async def start(self):
        loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        # Load model in thread so startup doesn't block
        await loop.run_in_executor(self._executor, self.load_model)
        self._worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    # ── public API ────────────────────────────────────────────────────────────

    async def submit(self, image_bytes: bytes) -> asyncio.Future:
        loop  = asyncio.get_event_loop()
        fut   = loop.create_future()
        item  = PendingItem(image_bytes=image_bytes, future=fut)
        self._total += 1
        await self._queue.put(item)
        return fut

    @property
    def queue_size(self) -> int:
        return self._queue.qsize() if self._queue else 0

    def metrics(self) -> dict:
        uptime   = time.monotonic() - self._start_time
        lats     = list(self._latencies)
        batches  = list(self._batch_sizes)

        # throughput: completions in last 60s
        now   = time.monotonic()
        recent = [t for t in self._recent_completions if now - t < 60]
        rps    = len(recent) / min(uptime, 60) if recent else 0.0

        def pct(data, p):
            return float(np.percentile(data, p)) if data else None

        return {
            "uptime_s":           round(uptime, 1),
            "requests_total":     self._total,
            "requests_completed": self._completed,
            "requests_failed":    self._failed,
            "avg_latency_ms":     round(float(np.mean(lats)), 2) if lats else None,
            "p50_latency_ms":     pct(lats, 50),
            "p95_latency_ms":     pct(lats, 95),
            "p99_latency_ms":     pct(lats, 99),
            "avg_batch_size":     round(float(np.mean(batches)), 2) if batches else None,
            "throughput_rps":     round(rps, 2),
            "device":             str(self.device),
            "queue_size":         self.queue_size,
        }

    # ── internal ──────────────────────────────────────────────────────────────

    async def _batch_worker(self):
        loop = asyncio.get_event_loop()
        while True:
            # Wait for first item
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            batch = [first]
            deadline = loop.time() + BATCH_TIMEOUT_S

            # Fill batch up to BATCH_SIZE within the timeout window
            while len(batch) < BATCH_SIZE:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            self._batch_sizes.append(len(batch))

            # Run synchronous inference in thread executor
            try:
                results = await loop.run_in_executor(
                    self._executor, self._sync_infer, batch
                )
                for item, result in zip(batch, results):
                    if not item.future.done():
                        item.future.set_result(result)
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    def _sync_infer(self, batch: list[PendingItem]) -> list[dict]:
        """Runs in the executor thread — synchronous."""
        t0 = time.monotonic()

        tensors = []
        for item in batch:
            try:
                pil = Image.open(io.BytesIO(item.image_bytes))
                tensors.append(self.transform(pil))
            except Exception:
                tensors.append(torch.zeros(3, 448, 448))

        x = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values=x).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        elapsed_ms = (time.monotonic() - t0) * 1000
        per_image_ms = elapsed_ms / len(batch)

        now = time.monotonic()
        results = []
        for p in probs:
            result = _classify(p)
            result["processing_ms"] = round(per_image_ms, 2)
            results.append(result)
            self._latencies.append(per_image_ms)
            self._recent_completions.append(now)
            self._completed += 1

        return results
