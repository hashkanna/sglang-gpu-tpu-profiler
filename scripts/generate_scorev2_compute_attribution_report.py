#!/usr/bin/env python3
"""Generate score-v2 compute-attribution report (padding + dispatch-shape waste)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return obj


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return obj


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0
    return num / den


def _bucket_up(value: int, buckets: list[int]) -> int:
    for bucket in buckets:
        if value <= bucket:
            return int(bucket)
    return int(value)


def _parse_static_arg_value(static_args: list[str], key: str, default: int) -> int:
    for idx, token in enumerate(static_args):
        if token == key and idx + 1 < len(static_args):
            try:
                return int(static_args[idx + 1])
            except ValueError:
                return default
    return default


def _summary_by_workload(
    summary: dict[str, Any],
    items_per_step: int,
) -> dict[str, dict[str, float]]:
    rows = [
        run
        for run in summary.get("runs", [])
        if int(run.get("items_per_step", -1)) == int(items_per_step) and not run.get("run_error")
    ]
    if not rows:
        raise ValueError(f"No successful runs for items_per_step={items_per_step}")

    out: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        for name, metrics in row.get("workloads", {}).items():
            sp = metrics.get("score_phase", {})
            out.setdefault(
                name,
                {
                    "throughput_items_per_sec": [],
                    "latency_p50_ms": [],
                    "latency_p99_ms": [],
                    "device_compute_median_ms": [],
                    "host_orchestration_median_ms": [],
                    "queue_wait_median_ms": [],
                    "dispatches_median": [],
                },
            )
            out[name]["throughput_items_per_sec"].append(
                float(metrics.get("throughput_items_per_sec", 0.0))
            )
            out[name]["latency_p50_ms"].append(float(metrics.get("latency_p50_ms", 0.0)))
            out[name]["latency_p99_ms"].append(float(metrics.get("latency_p99_ms", 0.0)))
            out[name]["device_compute_median_ms"].append(
                float(sp.get("device_compute_median_ms", 0.0))
            )
            out[name]["host_orchestration_median_ms"].append(
                float(sp.get("host_orchestration_median_ms", 0.0))
            )
            out[name]["queue_wait_median_ms"].append(float(sp.get("queue_wait_median_ms", 0.0)))
            out[name]["dispatches_median"].append(float(sp.get("dispatches_median", 0.0)))

    means: dict[str, dict[str, float]] = {}
    for name, cols in out.items():
        means[name] = {k: float(statistics.mean(v)) for k, v in cols.items()}
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate score-v2 compute-attribution report.")
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--tpu-summary", required=True, type=Path)
    parser.add_argument("--items-per-step", type=int, default=None)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    config = _load_yaml(args.baseline_config)
    summary = _load_json(args.tpu_summary)

    selected_items_per_step = args.items_per_step
    if selected_items_per_step is None:
        selected_items_per_step = int(summary.get("recommended", {}).get("items_per_step", 0))
    if selected_items_per_step <= 0:
        selected_items_per_step = int(
            config.get("tpu_matrix", {}).get("items_per_step_candidates", [96])[0]
        )

    matrix_cfg = config.get("tpu_matrix", {})
    shape_cfg = matrix_cfg.get("shape_contract", {})
    query_buckets = [int(x) for x in shape_cfg.get("query_token_buckets", [])]
    item_buckets = [int(x) for x in shape_cfg.get("item_token_buckets", [])]
    timed_requests = int(config.get("benchmark", {}).get("timed_requests", 10))

    static_args = config.get("tpu_server", {}).get("static_args", [])
    long_threshold = _parse_static_arg_value(
        static_args,
        "--multi-item-score-from-cache-v2-long-seq-threshold",
        0,
    )
    long_lane_items_per_step = _parse_static_arg_value(
        static_args,
        "--multi-item-score-from-cache-v2-long-seq-items-per-step",
        selected_items_per_step,
    )
    ultra_threshold = _parse_static_arg_value(
        static_args,
        "--multi-item-score-from-cache-v2-ultra-long-seq-threshold",
        0,
    )
    ultra_lane_items_per_step = _parse_static_arg_value(
        static_args,
        "--multi-item-score-from-cache-v2-ultra-long-seq-items-per-step",
        selected_items_per_step,
    )

    wl_metrics = _summary_by_workload(summary, selected_items_per_step)
    workloads = config.get("workloads", [])

    dispatch_size_histogram: dict[str, int] = defaultdict(int)
    dispatch_shape_histogram: dict[str, int] = defaultdict(int)
    dispatch_size_histogram_timed: dict[str, int] = defaultdict(int)
    dispatch_shape_histogram_timed: dict[str, int] = defaultdict(int)

    report_rows: list[dict[str, Any]] = []
    total_actual_tokens = 0.0
    total_processed_tokens = 0.0
    total_device_ms = 0.0
    total_upper_savings_ms = 0.0

    for workload in workloads:
        name = str(workload["name"])
        query_tokens = int(workload["query_tokens"])
        num_items = int(workload["num_items"])
        item_tokens = int(workload["item_tokens"])
        max_total_tokens = query_tokens + item_tokens

        query_bucket = _bucket_up(query_tokens, query_buckets)
        item_bucket = _bucket_up(item_tokens, item_buckets)

        effective_items_per_step = int(selected_items_per_step)
        long_lane_applied = False
        ultra_lane_applied = False
        if long_threshold > 0 and max_total_tokens >= long_threshold:
            effective_items_per_step = max(
                1, min(effective_items_per_step, int(long_lane_items_per_step))
            )
            long_lane_applied = True
        if ultra_threshold > 0 and max_total_tokens >= ultra_threshold:
            effective_items_per_step = max(
                1, min(effective_items_per_step, int(ultra_lane_items_per_step))
            )
            ultra_lane_applied = True

        dispatches = int(math.ceil(_safe_div(float(num_items), float(effective_items_per_step))))
        full_dispatches = num_items // effective_items_per_step
        tail_items = num_items % effective_items_per_step
        if tail_items == 0:
            tail_items = effective_items_per_step
            tail_dispatches = 0
        else:
            tail_dispatches = 1

        if full_dispatches > 0:
            dispatch_size_histogram[str(effective_items_per_step)] += full_dispatches
            dispatch_size_histogram_timed[str(effective_items_per_step)] += (
                full_dispatches * timed_requests
            )
            shape_key = (
                f"items={effective_items_per_step}|q_bucket={query_bucket}|"
                f"item_bucket={item_bucket}|lane=full"
            )
            dispatch_shape_histogram[shape_key] += full_dispatches
            dispatch_shape_histogram_timed[shape_key] += full_dispatches * timed_requests
        if tail_dispatches > 0:
            dispatch_size_histogram[str(tail_items)] += 1
            dispatch_size_histogram_timed[str(tail_items)] += timed_requests
            shape_key = (
                f"items={tail_items}|q_bucket={query_bucket}|"
                f"item_bucket={item_bucket}|lane=tail"
            )
            dispatch_shape_histogram[shape_key] += 1
            dispatch_shape_histogram_timed[shape_key] += timed_requests

        padded_item_slots = dispatches * effective_items_per_step - num_items
        processed_tokens = float(dispatches * effective_items_per_step * (query_bucket + item_bucket))
        actual_tokens = float(num_items * (query_tokens + item_tokens))
        wasted_tokens = max(0.0, processed_tokens - actual_tokens)
        wasted_slot_ratio = _safe_div(float(padded_item_slots), float(dispatches * effective_items_per_step))
        wasted_token_ratio = _safe_div(wasted_tokens, processed_tokens)

        observed = wl_metrics.get(name, {})
        device_ms = float(observed.get("device_compute_median_ms", 0.0))
        throughput = float(observed.get("throughput_items_per_sec", 0.0))

        # Upper-bound envelope assuming device compute scales with effective processed tokens.
        device_savings_upper_ms = device_ms * wasted_token_ratio
        projected_best_throughput = (
            throughput / (1.0 - wasted_token_ratio) if wasted_token_ratio < 0.95 else throughput
        )
        throughput_gain_upper = max(0.0, projected_best_throughput - throughput)

        total_actual_tokens += actual_tokens
        total_processed_tokens += processed_tokens
        total_device_ms += device_ms
        total_upper_savings_ms += device_savings_upper_ms

        report_rows.append(
            {
                "workload": name,
                "shape": {
                    "query_tokens": query_tokens,
                    "num_items": num_items,
                    "item_tokens": item_tokens,
                    "max_total_tokens": max_total_tokens,
                },
                "buckets": {
                    "query_bucket": query_bucket,
                    "item_bucket": item_bucket,
                },
                "lane_policy": {
                    "selected_items_per_step": selected_items_per_step,
                    "effective_items_per_step": effective_items_per_step,
                    "long_lane_applied": long_lane_applied,
                    "ultra_lane_applied": ultra_lane_applied,
                },
                "dispatch": {
                    "dispatches": dispatches,
                    "full_dispatches": full_dispatches,
                    "tail_dispatches": tail_dispatches,
                    "tail_items": tail_items,
                    "padded_item_slots": padded_item_slots,
                    "padded_item_slot_ratio": wasted_slot_ratio,
                },
                "token_waste": {
                    "actual_tokens": actual_tokens,
                    "processed_tokens": processed_tokens,
                    "wasted_tokens": wasted_tokens,
                    "wasted_token_ratio": wasted_token_ratio,
                },
                "observed_runtime": {
                    "throughput_items_per_sec": throughput,
                    "latency_p50_ms": float(observed.get("latency_p50_ms", 0.0)),
                    "latency_p99_ms": float(observed.get("latency_p99_ms", 0.0)),
                    "device_compute_median_ms": device_ms,
                    "host_orchestration_median_ms": float(
                        observed.get("host_orchestration_median_ms", 0.0)
                    ),
                    "queue_wait_median_ms": float(observed.get("queue_wait_median_ms", 0.0)),
                    "dispatches_median": float(observed.get("dispatches_median", 0.0)),
                },
                "compute_savings_envelope": {
                    "device_savings_upper_ms": device_savings_upper_ms,
                    "throughput_gain_upper_items_per_sec": throughput_gain_upper,
                    "projected_best_throughput_items_per_sec": projected_best_throughput,
                },
            }
        )

    report_rows.sort(
        key=lambda row: row["compute_savings_envelope"]["device_savings_upper_ms"],
        reverse=True,
    )

    aggregate_wasted_ratio = _safe_div(total_processed_tokens - total_actual_tokens, total_processed_tokens)
    aggregate = {
        "total_actual_tokens": total_actual_tokens,
        "total_processed_tokens": total_processed_tokens,
        "total_wasted_tokens": max(0.0, total_processed_tokens - total_actual_tokens),
        "aggregate_wasted_token_ratio": aggregate_wasted_ratio,
        "total_device_compute_median_ms": total_device_ms,
        "aggregate_device_savings_upper_ms": total_upper_savings_ms,
        "aggregate_device_savings_upper_ratio": _safe_div(total_upper_savings_ms, total_device_ms),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline_config": str(args.baseline_config),
            "tpu_summary": str(args.tpu_summary),
            "selected_items_per_step": selected_items_per_step,
            "timed_requests_per_repeat": timed_requests,
        },
        "lane_policy": {
            "long_seq_threshold": long_threshold,
            "long_seq_items_per_step": long_lane_items_per_step,
            "ultra_long_seq_threshold": ultra_threshold,
            "ultra_long_seq_items_per_step": ultra_lane_items_per_step,
        },
        "dispatch_size_histogram_per_request": dict(sorted(dispatch_size_histogram.items())),
        "dispatch_shape_histogram_per_request": dict(sorted(dispatch_shape_histogram.items())),
        "dispatch_size_histogram_timed_window": dict(sorted(dispatch_size_histogram_timed.items())),
        "dispatch_shape_histogram_timed_window": dict(
            sorted(dispatch_shape_histogram_timed.items())
        ),
        "aggregate": aggregate,
        "by_workload": report_rows,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")

    lines: list[str] = []
    lines.append("# Score-v2 Compute Attribution Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- baseline_config: `{args.baseline_config}`")
    lines.append(f"- tpu_summary: `{args.tpu_summary}`")
    lines.append(f"- selected_items_per_step: `{selected_items_per_step}`")
    lines.append(f"- timed_requests_per_repeat: `{timed_requests}`")
    lines.append("")
    lines.append("## Aggregate Waste + Savings Envelope")
    lines.append("")
    lines.append(
        f"- aggregate_wasted_token_ratio: `{aggregate['aggregate_wasted_token_ratio']:.4f}`"
    )
    lines.append(
        f"- aggregate_device_savings_upper_ms: `{aggregate['aggregate_device_savings_upper_ms']:.3f}`"
    )
    lines.append(
        f"- aggregate_device_savings_upper_ratio: `{aggregate['aggregate_device_savings_upper_ratio']:.4f}`"
    )
    lines.append("")
    lines.append("## Dispatch Size Histogram (Timed Window)")
    lines.append("")
    for size, count in sorted(dispatch_size_histogram_timed.items(), key=lambda kv: int(kv[0])):
        lines.append(f"- items={size}: `{count}` dispatches")
    lines.append("")
    lines.append("## Per-workload Attribution")
    lines.append("")
    lines.append(
        "| workload | lane ips | dispatches | padded slot ratio | wasted token ratio | device_ms | device savings upper (ms) | throughput | projected best throughput |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report_rows:
        lines.append(
            f"| {row['workload']} | {row['lane_policy']['effective_items_per_step']} | "
            f"{row['dispatch']['dispatches']} | {row['dispatch']['padded_item_slot_ratio']:.4f} | "
            f"{row['token_waste']['wasted_token_ratio']:.4f} | "
            f"{row['observed_runtime']['device_compute_median_ms']:.3f} | "
            f"{row['compute_savings_envelope']['device_savings_upper_ms']:.3f} | "
            f"{row['observed_runtime']['throughput_items_per_sec']:.3f} | "
            f"{row['compute_savings_envelope']['projected_best_throughput_items_per_sec']:.3f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Savings are an upper-bound envelope assuming device compute scales with effective processed tokens."
    )
    lines.append(
        "- This attribution combines shape-bucket padding and dispatch tail-fragmentation into a single token-waste estimate."
    )

    args.md_out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
