"""
MIC2 VLM Classifier — REST API
================================
Two usage patterns:

  Async (high-throughput):
    POST /v1/analyze          → { job_id, status: "pending" }
    GET  /v1/result/{job_id}  → { status: "ready", binary: {...}, multiclass: {...} }

  Sync (simple / low-latency):
    POST /v1/analyze/sync     → full result immediately (waits for inference)

OpenAPI docs: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .inference import InferenceEngine
from .schemas import (
    AnalysisResult,
    BenchmarkResponse,
    HealthResponse,
    JobStatus,
    MetricsResponse,
    SubmitResponse,
)

# ── config ────────────────────────────────────────────────────────────────────
ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH",
    os.path.join(os.path.dirname(__file__), "..", "weights", "mic2_internvl_v1"),
)
JOB_TTL_S  = int(os.getenv("JOB_TTL_S",  "300"))   # keep results 5 min
MAX_UPLOAD  = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1_048_576

# ── in-memory job store ───────────────────────────────────────────────────────
_jobs: dict[str, AnalysisResult] = {}

engine = InferenceEngine(adapter_path=ADAPTER_PATH)


# ── lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    asyncio.create_task(_cleanup_loop())
    yield
    await engine.stop()


async def _cleanup_loop():
    """Remove job results older than JOB_TTL_S."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now(timezone.utc)
        stale = [
            jid for jid, r in _jobs.items()
            if r.completed_at and (now - r.completed_at).total_seconds() > JOB_TTL_S
        ]
        for jid in stale:
            del _jobs[jid]


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MIC2 VLM Classifier API",
    description=(
        "Malicious ad detection using InternVL2.5-1B vision encoder + LoRA.\n\n"
        "**Two-stage pipeline:**\n"
        "- Stage 1 (binary): Benign+blank_LP vs Malicious — 99.3% accuracy\n"
        "- Stage 2 (multiclass): 11 malicious categories with per-class confidence thresholds\n\n"
        "**Decisions:** `pass` (benign), `block` (high-confidence malicious), "
        "`review` (below auto-block threshold)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────
def _make_result(job_id: str, raw: dict, submitted_at: datetime) -> AnalysisResult:
    from .schemas import (BinaryResult, ClassScore, Decision, MulticlassResult, Verdict)

    binary = BinaryResult(
        verdict    = Verdict(raw["binary"]["verdict"]),
        confidence = raw["binary"]["confidence"],
    )
    mc_raw = raw["multiclass"]
    multiclass = MulticlassResult(
        label      = mc_raw["label"],
        confidence = mc_raw["confidence"],
        decision   = Decision(mc_raw["decision"]),
        threshold  = mc_raw["threshold"],
        top3       = [ClassScore(**s) for s in mc_raw["top3"]],
    )
    return AnalysisResult(
        job_id        = job_id,
        status        = JobStatus.READY,
        submitted_at  = submitted_at,
        completed_at  = datetime.now(timezone.utc),
        processing_ms = raw["processing_ms"],
        binary        = binary,
        multiclass    = multiclass,
        all_scores    = raw["all_scores"],
    )


async def _run_job(job_id: str, image_bytes: bytes, submitted_at: datetime):
    """Background task: run inference and store result."""
    try:
        fut    = await engine.submit(image_bytes)
        raw    = await fut
        result = _make_result(job_id, raw, submitted_at)
    except Exception as exc:
        result = AnalysisResult(
            job_id       = job_id,
            status       = JobStatus.FAILED,
            submitted_at = submitted_at,
            completed_at = datetime.now(timezone.utc),
            error        = str(exc),
        )
    _jobs[job_id] = result


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Service health check."""
    m = engine.metrics()
    return HealthResponse(
        status       = "ok" if engine.loaded else "loading",
        model_loaded = engine.loaded,
        device       = m["device"],
        uptime_s     = m["uptime_s"],
        queue_size   = m["queue_size"],
    )


@app.get("/v1/metrics", response_model=MetricsResponse, tags=["System"])
async def metrics():
    """Runtime performance metrics."""
    return MetricsResponse(**engine.metrics())


@app.post(
    "/v1/analyze",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Inference"],
    summary="Submit image for async analysis",
)
async def analyze_async(image: Annotated[UploadFile, File(description="Screenshot image (JPEG/PNG/WEBP)")]):
    """
    Submit an image for analysis. Returns a `job_id` immediately.
    Poll `GET /v1/result/{job_id}` until `status == "ready"`.
    """
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly")

    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD // 1_048_576} MB limit")

    job_id       = str(uuid.uuid4())
    submitted_at = datetime.now(timezone.utc)
    _jobs[job_id] = AnalysisResult(
        job_id       = job_id,
        status       = JobStatus.PENDING,
        submitted_at = submitted_at,
    )
    asyncio.create_task(_run_job(job_id, data, submitted_at))

    return SubmitResponse(
        job_id         = job_id,
        status         = JobStatus.PENDING,
        queue_position = engine.queue_size,
    )


@app.get(
    "/v1/result/{job_id}",
    response_model=AnalysisResult,
    tags=["Inference"],
    summary="Poll for analysis result",
)
async def get_result(job_id: str):
    """Retrieve the result for a previously submitted job."""
    result = _jobs.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found (may have expired)")
    return result


@app.post(
    "/v1/analyze/sync",
    response_model=AnalysisResult,
    tags=["Inference"],
    summary="Synchronous analysis (waits for result)",
)
async def analyze_sync(image: Annotated[UploadFile, File(description="Screenshot image (JPEG/PNG/WEBP)")]):
    """
    Submit an image and wait for the result in a single request.
    Simpler than the async flow but ties up the connection during inference.
    """
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly")

    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD // 1_048_576} MB limit")

    job_id       = str(uuid.uuid4())
    submitted_at = datetime.now(timezone.utc)

    try:
        fut = await engine.submit(data)
        raw = await fut
        return _make_result(job_id, raw, submitted_at)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/v1/benchmark",
    response_model=BenchmarkResponse,
    tags=["System"],
    summary="Internal throughput benchmark",
)
async def benchmark(
    image:       Annotated[UploadFile, File(description="Image to repeat for the benchmark")],
    n_requests:  int = 50,
    concurrency: int = 8,
):
    """
    Runs `n_requests` inferences using `concurrency` parallel coroutines
    and reports throughput and latency percentiles.
    Useful for capacity planning — does NOT store results.
    """
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if n_requests < 1 or n_requests > 500:
        raise HTTPException(status_code=400, detail="n_requests must be 1–500")
    if concurrency < 1 or concurrency > 64:
        raise HTTPException(status_code=400, detail="concurrency must be 1–64")

    data = await image.read()
    sem  = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async def one_request():
        nonlocal errors
        async with sem:
            t0 = time.monotonic()
            try:
                fut = await engine.submit(data)
                await fut
                latencies.append((time.monotonic() - t0) * 1000)
            except Exception:
                errors += 1

    t_start = time.monotonic()
    await asyncio.gather(*[one_request() for _ in range(n_requests)])
    total_s = time.monotonic() - t_start

    if not latencies:
        raise HTTPException(status_code=500, detail="All benchmark requests failed")

    import numpy as np
    lats = np.array(latencies)
    return BenchmarkResponse(
        n_requests     = n_requests,
        concurrency    = concurrency,
        total_time_s   = round(total_s, 3),
        throughput_rps = round(len(latencies) / total_s, 2),
        avg_latency_ms = round(float(np.mean(lats)), 2),
        p50_latency_ms = round(float(np.percentile(lats, 50)), 2),
        p95_latency_ms = round(float(np.percentile(lats, 95)), 2),
        p99_latency_ms = round(float(np.percentile(lats, 99)), 2),
        min_latency_ms = round(float(np.min(lats)), 2),
        max_latency_ms = round(float(np.max(lats)), 2),
        errors         = errors,
    )
