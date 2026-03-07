#!/usr/bin/env python3
"""Generate side-by-side report from adaptive TPU summary and a GPU baseline run."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKLOAD_ORDER = ["pr28_hotshape", "small_batch", "medium_batch"]


def pct_delta(base: float | None, new: float | None) -> float | None:
    if base is None or new is None or base == 0:
        return None
    return ((new - base) / base) * 100.0


def fmt(v: float | None, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def fmt_pct(v: float | None, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}%"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def render_md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Adaptive TPU vs GPU Baseline")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- adaptive_summary: `{report['adaptive_summary']}`")
    lines.append(f"- gpu_runs: `{report['gpu_baseline']['run_count']}`")
    lines.append(f"- gpu_throughput_ref: `median over gpu runs`")
    for run_dir in report["gpu_baseline"]["run_dirs"]:
        lines.append(f"- gpu_run_dir: `{run_dir}`")
    lines.append("")
    lines.append("## Side-by-Side")
    lines.append("")
    lines.append(
        "| Workload | TPU recommended ips | GPU tput (median) | TPU tput | TPU vs GPU | GPU p50 (median) | TPU p50 | GPU p99 (median) | TPU p99 | TPU failure_rate | TPU top_error |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for wl in WORKLOAD_ORDER:
        row = report["workloads"][wl]
        lines.append(
            f"| {wl} | {row['recommended_items_per_step']} | "
            f"{fmt(row['gpu_throughput_items_per_sec'], 1)} | {fmt(row['tpu_throughput_items_per_sec'], 1)} | "
            f"{fmt_pct(row['delta_throughput_pct'], 1)} | "
            f"{fmt(row['gpu_p50_ms'], 1)} | {fmt(row['tpu_p50_ms'], 1)} | "
            f"{fmt(row['gpu_p99_ms'], 1)} | {fmt(row['tpu_p99_ms'], 1)} | "
            f"{fmt(row['tpu_failure_rate'] * 100.0, 1)}% | {row['tpu_top_error']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate adaptive TPU vs GPU report.")
    parser.add_argument("--adaptive-summary", required=True)
    parser.add_argument(
        "--gpu-run-dir",
        action="append",
        default=[],
        help="GPU run directory containing raw_results.json. Repeat this flag for multiple runs.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    adaptive_summary = Path(args.adaptive_summary).resolve()
    if not adaptive_summary.exists():
        raise FileNotFoundError(f"Missing adaptive summary: {adaptive_summary}")
    if not args.gpu_run_dir:
        raise ValueError("Provide at least one --gpu-run-dir.")

    gpu_run_dirs = [Path(x).resolve() for x in args.gpu_run_dir]
    gpu_raw_paths = [p / "raw_results.json" for p in gpu_run_dirs]
    for raw_path in gpu_raw_paths:
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing GPU raw results: {raw_path}")

    adaptive = json.loads(adaptive_summary.read_text())

    gpu_rows_by_workload: dict[str, list[dict[str, float]]] = {wl: [] for wl in WORKLOAD_ORDER}
    for raw_path in gpu_raw_paths:
        gpu = json.loads(raw_path.read_text()).get("gpu", {})
        for wl in WORKLOAD_ORDER:
            row = gpu.get(wl)
            if not row:
                continue
            gpu_rows_by_workload[wl].append(
                {
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"]),
                    "latency_p50_ms": float(row["latency_p50_ms"]),
                    "latency_p99_ms": float(row["latency_p99_ms"]),
                }
            )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "adaptive_summary": str(adaptive_summary),
        "gpu_baseline": {
            "run_count": len(gpu_run_dirs),
            "run_dirs": [str(p) for p in gpu_run_dirs],
            "by_workload": {},
        },
        "workloads": {},
    }

    for wl in WORKLOAD_ORDER:
        gpu_rows = gpu_rows_by_workload[wl]
        if not gpu_rows:
            raise ValueError(f"No GPU rows found for workload '{wl}' across provided runs.")

        gpu_tput_vals = [r["throughput_items_per_sec"] for r in gpu_rows]
        gpu_p50_vals = [r["latency_p50_ms"] for r in gpu_rows]
        gpu_p99_vals = [r["latency_p99_ms"] for r in gpu_rows]

        gpu_tput_median = _median(gpu_tput_vals)
        gpu_p50_median = _median(gpu_p50_vals)
        gpu_p99_median = _median(gpu_p99_vals)
        if gpu_tput_median is None or gpu_p50_median is None or gpu_p99_median is None:
            raise ValueError(f"Unable to compute GPU medians for workload '{wl}'.")

        report["gpu_baseline"]["by_workload"][wl] = {
            "n_runs": len(gpu_rows),
            "throughput_items_per_sec_median": gpu_tput_median,
            "throughput_items_per_sec_min": min(gpu_tput_vals),
            "throughput_items_per_sec_max": max(gpu_tput_vals),
            "latency_p50_ms_median": gpu_p50_median,
            "latency_p50_ms_min": min(gpu_p50_vals),
            "latency_p50_ms_max": max(gpu_p50_vals),
            "latency_p99_ms_median": gpu_p99_median,
            "latency_p99_ms_min": min(gpu_p99_vals),
            "latency_p99_ms_max": max(gpu_p99_vals),
        }

        t = adaptive["workloads"][wl]["recommended_row"]
        top_err = "none"
        top_err_list = t.get("top_error_counts") or []
        if top_err_list:
            err, cnt = top_err_list[0]
            top_err = f"{cnt}x {str(err)[:100]}"
        report["workloads"][wl] = {
            "recommended_items_per_step": int(adaptive["workloads"][wl]["recommended_items_per_step"]),
            "gpu_throughput_items_per_sec": gpu_tput_median,
            "tpu_throughput_items_per_sec": float(t["throughput_median_items_per_sec"]),
            "delta_throughput_pct": pct_delta(
                gpu_tput_median,
                float(t["throughput_median_items_per_sec"]),
            ),
            "gpu_p50_ms": gpu_p50_median,
            "tpu_p50_ms": float(t["latency_p50_median_ms"]),
            "gpu_p99_ms": gpu_p99_median,
            "tpu_p99_ms": float(t["latency_p99_median_ms"]),
            "tpu_failure_rate": float(t["failure_rate"]),
            "tpu_top_error": top_err,
        }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else adaptive_summary.parent / "adaptive_vs_gpu_report.md"
    )
    out_json = out_path.with_suffix(".json")
    out_path.write_text(render_md(report))
    out_json.write_text(json.dumps(report, indent=2))
    print(f"report_md={out_path}")
    print(f"report_json={out_json}")


if __name__ == "__main__":
    main()
