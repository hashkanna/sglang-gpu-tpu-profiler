#!/usr/bin/env python3
"""Run production-shaped /v1/score benchmark against a GPU endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests
import yaml


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    idx = (len(arr) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    if lo == hi:
        return arr[lo]
    return arr[lo] * (hi - idx) + arr[hi] * (idx - lo)


def build_payload(model: str, workload: dict[str, Any]) -> dict[str, Any]:
    query_tokens = int(workload["query_tokens"])
    num_items = int(workload["num_items"])
    item_tokens = int(workload["item_tokens"])
    label_token_ids = [int(x) for x in workload.get("label_token_ids", [198])]
    apply_softmax = bool(workload.get("apply_softmax", False))
    query = "hello " * query_tokens
    item = "hello " * item_tokens
    return {
        "model": model,
        "query": query,
        "items": [item for _ in range(num_items)],
        "label_token_ids": label_token_ids,
        "apply_softmax": apply_softmax,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /v1/score on a GPU endpoint.")
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--url", required=True, help="GPU server base URL, e.g. http://127.0.0.1:30000")
    parser.add_argument("--endpoint", default="/v1/score")
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--timed-requests", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    baseline = yaml.safe_load(args.baseline_config.read_text())
    model = str(baseline["experiment"]["model"])
    workloads = list(baseline["workloads"])
    url = args.url.rstrip("/") + args.endpoint

    output: dict[str, Any] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "url": url,
        "benchmark": {
            "warmup_requests": int(args.warmup_requests),
            "timed_requests": int(args.timed_requests),
            "concurrency": 1,
            "timeout_sec": int(args.timeout_sec),
        },
        "workloads": {},
    }

    for wl in workloads:
        name = str(wl["name"])
        payload = build_payload(model, wl)

        # Warmup phase.
        for _ in range(int(args.warmup_requests)):
            try:
                requests.post(url, json=payload, timeout=int(args.timeout_sec))
            except Exception:
                pass

        latencies_ms: list[float] = []
        failures = 0
        error_counts: dict[str, int] = {}
        t_start = time.perf_counter()
        for _ in range(int(args.timed_requests)):
            req_start = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, timeout=int(args.timeout_sec))
                if resp.status_code == 200:
                    latencies_ms.append((time.perf_counter() - req_start) * 1000.0)
                else:
                    failures += 1
                    key = f"http_{resp.status_code}"
                    error_counts[key] = error_counts.get(key, 0) + 1
            except Exception as e:  # noqa: BLE001 - benchmark should classify all request failures.
                failures += 1
                key = type(e).__name__
                error_counts[key] = error_counts.get(key, 0) + 1
        elapsed_sec = max(1e-9, time.perf_counter() - t_start)
        success = len(latencies_ms)
        num_items = int(wl["num_items"])

        output["workloads"][name] = {
            "query_tokens": int(wl["query_tokens"]),
            "num_items": num_items,
            "item_tokens": int(wl["item_tokens"]),
            "throughput_items_per_sec": (success * num_items) / elapsed_sec,
            "latency_p50_ms": statistics.median(latencies_ms) if latencies_ms else None,
            "latency_p99_ms": percentile(latencies_ms, 0.99),
            "num_failures": failures,
            "error_counts": error_counts,
            "successful_requests": success,
            "timed_elapsed_sec": elapsed_sec,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
