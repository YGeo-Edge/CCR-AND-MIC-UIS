#!/usr/bin/env python3
"""
Standalone throughput benchmark for the MIC2 API.

Usage:
    python benchmark.py --image path/to/test.jpg
    python benchmark.py --image path/to/test.jpg --url http://localhost:8000 --n 100 --concurrency 16
    python benchmark.py --image path/to/test.jpg --mode sync    # test sync endpoint
    python benchmark.py --image path/to/test.jpg --mode async   # test async submit+poll
    python benchmark.py --image path/to/test.jpg --mode server  # use /v1/benchmark endpoint

Results printed as a table + per-request latency distribution.
"""
import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import httpx

BASE_URL    = "http://localhost:8000"
POLL_INTERVAL = 0.05   # 50ms polling interval for async mode


# ── helpers ───────────────────────────────────────────────────────────────────

async def request_sync(client: httpx.AsyncClient, image_bytes: bytes, url: str) -> float:
    t0 = time.monotonic()
    r = await client.post(
        f"{url}/v1/analyze/sync",
        files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=120,
    )
    r.raise_for_status()
    return (time.monotonic() - t0) * 1000


async def request_async(client: httpx.AsyncClient, image_bytes: bytes, url: str) -> float:
    t0 = time.monotonic()
    # Submit
    r = await client.post(
        f"{url}/v1/analyze",
        files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]

    # Poll
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        r = await client.get(f"{url}/v1/result/{job_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if data["status"] in ("ready", "failed"):
            break

    return (time.monotonic() - t0) * 1000


async def run_benchmark(url: str, image_bytes: bytes, n: int, concurrency: int, mode: str):
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async def one(client):
        nonlocal errors
        async with sem:
            try:
                if mode == "sync":
                    lat = await request_sync(client, image_bytes, url)
                else:
                    lat = await request_async(client, image_bytes, url)
                latencies.append(lat)
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                errors += 1

    print(f"Warming up (2 requests) …")
    async with httpx.AsyncClient() as client:
        await asyncio.gather(one(client), one(client))
        latencies.clear()

        print(f"Running {n} requests  concurrency={concurrency}  mode={mode} …")
        t_start = time.monotonic()
        await asyncio.gather(*[one(client) for _ in range(n)])
        total_s = time.monotonic() - t_start

    return latencies, errors, total_s


async def run_server_benchmark(url: str, image_bytes: bytes, n: int, concurrency: int):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{url}/v1/benchmark",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            params={"n_requests": n, "concurrency": concurrency},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()


# ── report ────────────────────────────────────────────────────────────────────

def print_report(latencies, errors, total_s, n, concurrency, mode):
    ok = len(latencies)
    if not latencies:
        print("All requests failed.")
        return

    lats = sorted(latencies)
    throughput = ok / total_s

    def pct(p):
        idx = min(int(len(lats) * p / 100), len(lats) - 1)
        return round(lats[idx], 1)

    print()
    print("=" * 52)
    print("  MIC2 API Benchmark Results")
    print("=" * 52)
    print(f"  Mode:          {mode}")
    print(f"  Requests:      {n}  ({ok} ok, {errors} errors)")
    print(f"  Concurrency:   {concurrency}")
    print(f"  Total time:    {total_s:.2f}s")
    print(f"  Throughput:    {throughput:.2f} req/s")
    print(f"  Avg latency:   {statistics.mean(lats):.1f} ms")
    print(f"  Min latency:   {lats[0]:.1f} ms")
    print(f"  p50 latency:   {pct(50):.1f} ms")
    print(f"  p90 latency:   {pct(90):.1f} ms")
    print(f"  p95 latency:   {pct(95):.1f} ms")
    print(f"  p99 latency:   {pct(99):.1f} ms")
    print(f"  Max latency:   {lats[-1]:.1f} ms")
    print("=" * 52)
    print()

    # Latency histogram (ASCII)
    bucket_ms = [0, 100, 250, 500, 1000, 2000, 5000, float("inf")]
    labels = ["<100ms", "100-250ms", "250-500ms", "500ms-1s", "1-2s", "2-5s", ">5s"]
    print("  Latency distribution:")
    for i, label in enumerate(labels):
        lo, hi = bucket_ms[i], bucket_ms[i + 1]
        count = sum(1 for l in lats if lo <= l < hi)
        bar = "█" * int(count / len(lats) * 30)
        print(f"  {label:12s} {bar:30s} {count:4d} ({count/len(lats)*100:4.1f}%)")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MIC2 API benchmark")
    parser.add_argument("--image",       required=True, help="Test image path")
    parser.add_argument("--url",         default=BASE_URL, help="API base URL")
    parser.add_argument("--n",           type=int, default=50, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=8,  help="Max concurrent requests")
    parser.add_argument("--mode",        choices=["sync", "async", "server"],
                        default="sync", help="sync=one-shot, async=submit+poll, server=server-side bench")
    args = parser.parse_args()

    image_path  = Path(args.image)
    image_bytes = image_path.read_bytes()
    print(f"Image: {image_path.name} ({len(image_bytes)//1024} KB)")

    # Check health first
    import urllib.request
    try:
        with urllib.request.urlopen(f"{args.url}/v1/health", timeout=5) as r:
            import json
            h = json.loads(r.read())
            if not h["model_loaded"]:
                print("WARNING: Model not yet loaded — results may be unreliable")
            print(f"Server: {h['status']}  device={h['device']}  uptime={h['uptime_s']}s")
    except Exception as e:
        print(f"Could not reach {args.url}/v1/health: {e}")
        sys.exit(1)

    if args.mode == "server":
        result = asyncio.run(run_server_benchmark(args.url, image_bytes, args.n, args.concurrency))
        print()
        print("=" * 52)
        print("  MIC2 API Benchmark Results  (server-side)")
        print("=" * 52)
        for k, v in result.items():
            print(f"  {k:<22s} {v}")
        print("=" * 52)
    else:
        latencies, errors, total_s = asyncio.run(
            run_benchmark(args.url, image_bytes, args.n, args.concurrency, args.mode)
        )
        print_report(latencies, errors, total_s, args.n, args.concurrency, args.mode)


if __name__ == "__main__":
    main()
