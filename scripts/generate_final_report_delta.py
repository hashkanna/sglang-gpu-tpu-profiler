#!/usr/bin/env python3
"""Compare two final TPU-vs-GPU report JSON files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return obj


def _by_workload(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in report.get("per_workload", []):
        name = str(row.get("workload"))
        if name:
            out[name] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare before/after final reports")
    parser.add_argument("--before-final-report", required=True, type=Path)
    parser.add_argument("--after-final-report", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    before = _load(args.before_final_report)
    after = _load(args.after_final_report)

    before_agg_tpu = before.get("aggregate_means", {}).get("tpu", {})
    after_agg_tpu = after.get("aggregate_means", {}).get("tpu", {})
    before_gap = before.get("aggregate_means", {}).get("delta_tpu_minus_gpu", {})
    after_gap = after.get("aggregate_means", {}).get("delta_tpu_minus_gpu", {})

    delta = {
        "tpu_aggregate_after_minus_before": {
            "throughput_items_per_sec": float(after_agg_tpu.get("throughput_items_per_sec", 0.0))
            - float(before_agg_tpu.get("throughput_items_per_sec", 0.0)),
            "latency_p50_ms": float(after_agg_tpu.get("latency_p50_ms", 0.0))
            - float(before_agg_tpu.get("latency_p50_ms", 0.0)),
            "latency_p99_ms": float(after_agg_tpu.get("latency_p99_ms", 0.0))
            - float(before_agg_tpu.get("latency_p99_ms", 0.0)),
        },
        "tpu_minus_gpu_gap_after_minus_before": {
            "throughput_items_per_sec": float(after_gap.get("throughput_items_per_sec", 0.0))
            - float(before_gap.get("throughput_items_per_sec", 0.0)),
            "latency_p50_ms": float(after_gap.get("latency_p50_ms", 0.0))
            - float(before_gap.get("latency_p50_ms", 0.0)),
            "latency_p99_ms": float(after_gap.get("latency_p99_ms", 0.0))
            - float(before_gap.get("latency_p99_ms", 0.0)),
        },
    }

    before_wl = _by_workload(before)
    after_wl = _by_workload(after)
    workloads = sorted(set(before_wl.keys()) & set(after_wl.keys()))
    per_workload: list[dict[str, Any]] = []
    for wl in workloads:
        b = before_wl[wl]
        a = after_wl[wl]
        per_workload.append(
            {
                "workload": wl,
                "tpu_after_minus_before": {
                    "throughput_items_per_sec": float(a["tpu"]["throughput_items_per_sec"])
                    - float(b["tpu"]["throughput_items_per_sec"]),
                    "latency_p50_ms": float(a["tpu"]["latency_p50_ms"])
                    - float(b["tpu"]["latency_p50_ms"]),
                    "latency_p99_ms": float(a["tpu"]["latency_p99_ms"])
                    - float(b["tpu"]["latency_p99_ms"]),
                },
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "before_final_report": str(args.before_final_report),
        "after_final_report": str(args.after_final_report),
        "delta": delta,
        "per_workload": per_workload,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    lines = ["# Final Report Delta", "", "## Aggregate TPU Delta (after - before)", ""]
    tpu_d = delta["tpu_aggregate_after_minus_before"]
    lines.append(f"- throughput_items_per_sec: `{tpu_d['throughput_items_per_sec']:+.3f}`")
    lines.append(f"- latency_p50_ms: `{tpu_d['latency_p50_ms']:+.3f}`")
    lines.append(f"- latency_p99_ms: `{tpu_d['latency_p99_ms']:+.3f}`")

    lines.extend(["", "## TPU-minus-GPU Gap Delta (after - before)", ""])
    gap_d = delta["tpu_minus_gpu_gap_after_minus_before"]
    lines.append(f"- throughput_items_per_sec: `{gap_d['throughput_items_per_sec']:+.3f}`")
    lines.append(f"- latency_p50_ms: `{gap_d['latency_p50_ms']:+.3f}`")
    lines.append(f"- latency_p99_ms: `{gap_d['latency_p99_ms']:+.3f}`")

    lines.extend(["", "## Per-workload TPU Delta (after - before)", "", "| workload | throughput delta (items/s) | p50 delta (ms) | p99 delta (ms) |", "|---|---:|---:|---:|"])
    for row in report["per_workload"]:
        d = row["tpu_after_minus_before"]
        lines.append(
            f"| {row['workload']} | {d['throughput_items_per_sec']:+.3f} | {d['latency_p50_ms']:+.3f} | {d['latency_p99_ms']:+.3f} |"
        )

    args.md_out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
