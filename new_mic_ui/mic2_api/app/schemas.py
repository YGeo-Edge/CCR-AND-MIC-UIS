from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    BENIGN   = "BENIGN"
    MALICIOUS = "MALICIOUS"


class Decision(str, Enum):
    PASS    = "pass"     # BENIGN — no action
    REVIEW  = "review"   # malicious but below auto-block threshold
    BLOCK   = "block"    # malicious above threshold → auto-block


class JobStatus(str, Enum):
    PENDING = "pending"
    READY   = "ready"
    FAILED  = "failed"


class ClassScore(BaseModel):
    label:      str
    confidence: float = Field(ge=0.0, le=1.0)


class BinaryResult(BaseModel):
    verdict:    Verdict
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the top predicted class")


class MulticlassResult(BaseModel):
    label:      str   = Field(description="Top predicted class name")
    confidence: float = Field(ge=0.0, le=1.0)
    decision:   Decision
    threshold:  float = Field(description="Per-class auto-block threshold used")
    top3:       list[ClassScore]


class AnalysisResult(BaseModel):
    job_id:        str
    status:        JobStatus
    submitted_at:  datetime
    completed_at:  datetime | None = None
    processing_ms: float | None    = None
    binary:        BinaryResult | None    = None
    multiclass:    MulticlassResult | None = None
    all_scores:    dict[str, float] | None = None
    error:         str | None = None


class SubmitResponse(BaseModel):
    job_id:         str
    status:         JobStatus = JobStatus.PENDING
    queue_position: int


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    device:       str
    uptime_s:     float
    queue_size:   int


class MetricsResponse(BaseModel):
    uptime_s:            float
    requests_total:      int
    requests_completed:  int
    requests_failed:     int
    avg_latency_ms:      float | None
    p50_latency_ms:      float | None
    p95_latency_ms:      float | None
    p99_latency_ms:      float | None
    avg_batch_size:      float | None
    throughput_rps:      float | None   # over last 60 s
    device:              str
    queue_size:          int


class BenchmarkResponse(BaseModel):
    n_requests:      int
    concurrency:     int
    total_time_s:    float
    throughput_rps:  float
    avg_latency_ms:  float
    p50_latency_ms:  float
    p95_latency_ms:  float
    p99_latency_ms:  float
    min_latency_ms:  float
    max_latency_ms:  float
    errors:          int
