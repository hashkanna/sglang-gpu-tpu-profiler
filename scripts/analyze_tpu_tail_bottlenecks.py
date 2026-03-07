#!/usr/bin/env python3
"""Quantify good-vs-bad tail bottleneck split from matrix_summary artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    summary_path: str
    run_dir: str
    items_per_step: int
    repeat_idx: int
    p99_p50_ratio: float
    throughput_items_per_sec: float
    queue_wait_ms: float
    device_compute_ms: float
    host_orchestration_ms: float

    @property
    def total_phase_ms(self) -> float:
        return self.queue_wait_ms + self.device_compute_ms + self.host_orchestration_ms


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    idx = int(round((len(sorted_values) - 1) * q))
    return float(sorted_values[idx])


def load_records(summary_paths: list[Path], workload: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    for summary_path in summary_paths:
        raw = json.loads(summary_path.read_text())
        for run in raw.get("runs", []):
            wl = run.get("workloads", {}).get(workload, {})
            phase = wl.get("score_phase", {})
            ratio = wl.get("p99_p50_ratio")
            queue_wait = phase.get("queue_wait_median_ms")
            device = phase.get("device_compute_median_ms")
            host = phase.get("host_orchestration_median_ms")
            throughput = wl.get("throughput_items_per_sec")
            if None in (ratio, queue_wait, device, host, throughput):
                continue
            records.append(
                RunRecord(
                    summary_path=str(summary_path),
                    run_dir=str(run.get("run_dir", "")),
                    items_per_step=int(run.get("items_per_step", 0)),
                    repeat_idx=int(run.get("repeat_idx", 0)),
                    p99_p50_ratio=float(ratio),
                    throughput_items_per_sec=float(throughput),
                    queue_wait_ms=float(queue_wait),
                    device_compute_ms=float(device),
                    host_orchestration_ms=float(host),
                )
            )
    return records


def summarize(records: list[RunRecord]) -> dict[str, float]:
    queue = [r.queue_wait_ms for r in records]
    device = [r.device_compute_ms for r in records]
    host = [r.host_orchestration_ms for r in records]
    total = [r.total_phase_ms for r in records]
    ratio = [r.p99_p50_ratio for r in records]
    tput = [r.throughput_items_per_sec for r in records]

    queue_med = _median(queue)
    device_med = _median(device)
    host_med = _median(host)
    total_med = _median(total)
    return {
        "count": float(len(records)),
        "p99_p50_ratio_median": _median(ratio),
        "throughput_items_per_sec_median": _median(tput),
        "queue_wait_median_ms": queue_med,
        "device_compute_median_ms": device_med,
        "host_orchestration_median_ms": host_med,
        "total_phase_median_ms": total_med,
        "queue_share_pct": (queue_med / total_med * 100.0) if total_med > 0 else 0.0,
        "device_share_pct": (device_med / total_med * 100.0) if total_med > 0 else 0.0,
        "host_share_pct": (host_med / total_med * 100.0) if total_med > 0 else 0.0,
    }


def pick_fixes(good: dict[str, float], bad: dict[str, float], best_run: RunRecord) -> list[str]:
    delta_queue = bad["queue_wait_median_ms"] - good["queue_wait_median_ms"]
    delta_device = bad["device_compute_median_ms"] - good["device_compute_median_ms"]
    delta_host = bad["host_orchestration_median_ms"] - good["host_orchestration_median_ms"]

    ranked = sorted(
        [("queue", delta_queue), ("device", delta_device), ("host", delta_host)],
        key=lambda x: x[1],
        reverse=True,
    )

    fixes: list[str] = []
    for kind, delta in ranked:
        if kind == "host":
            fixes.append(
                "Use single-workload lane auto-bump (`max_running_requests` and "
                "`multi_item_extend_batch_size` >= workload items). "
                f"Observed host-orchestration delta: {delta:.3f} ms."
            )
        elif kind == "queue":
            fixes.append(
                "Reduce queueing by aligning lane capacity to workload shape and avoiding under-sized "
                "extend batches. "
                f"Observed queue-wait delta: {delta:.3f} ms."
            )
        else:
            fixes.append(
                "Reduce device-phase tail by preferring lanes with lower dispatch fragmentation and "
                "stable compile-free timed windows. "
                f"Observed device-compute delta: {delta:.3f} ms."
            )

    fixes.append(
        "Reference best observed run as baseline target: "
        f"items_per_step={best_run.items_per_step}, "
        f"p99/p50={best_run.p99_p50_ratio:.3f}, "
        f"run_dir={best_run.run_dir}"
    )
    return fixes[:3]


def render_markdown(
    *,
    workload: str,
    total_records: int,
    good_summary: dict[str, float],
    bad_summary: dict[str, float],
    good_threshold: float,
    bad_threshold: float,
    fixes: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"# TPU Tail Bottleneck Deep Dive ({workload})")
    lines.append("")
    lines.append("## Cohort Selection")
    lines.append("")
    lines.append(f"- total_runs_analyzed: `{total_records}`")
    lines.append(f"- good_tail_threshold_p99_p50: `<= {good_threshold:.4f}`")
    lines.append(f"- bad_tail_threshold_p99_p50: `>= {bad_threshold:.4f}`")
    lines.append("")
    lines.append("## Host/Device Split")
    lines.append("")
    lines.append("| Cohort | count | p99/p50 med | throughput med | queue med (ms) | device med (ms) | host med (ms) | total med (ms) | queue % | device % | host % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in (("Good tail", good_summary), ("Bad tail", bad_summary)):
        lines.append(
            f"| {name} | {int(s['count'])} | {s['p99_p50_ratio_median']:.4f} | "
            f"{s['throughput_items_per_sec_median']:.2f} | {s['queue_wait_median_ms']:.3f} | "
            f"{s['device_compute_median_ms']:.3f} | {s['host_orchestration_median_ms']:.3f} | "
            f"{s['total_phase_median_ms']:.3f} | {s['queue_share_pct']:.2f}% | "
            f"{s['device_share_pct']:.2f}% | {s['host_share_pct']:.2f}% |"
        )
    lines.append("")
    lines.append("## Top 3 Fixes")
    lines.append("")
    for idx, fix in enumerate(fixes, start=1):
        lines.append(f"{idx}. {fix}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze good-vs-bad tail bottlenecks.")
    parser.add_argument("--workload", default="medium_batch")
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Path to matrix_summary.json (can be passed multiple times).",
    )
    parser.add_argument("--good-quantile", type=float, default=0.30)
    parser.add_argument("--bad-quantile", type=float, default=0.70)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    summary_paths = [Path(p).resolve() for p in args.summary]
    missing = [str(p) for p in summary_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing summary paths: {missing}")

    records = load_records(summary_paths, args.workload)
    if len(records) < 4:
        raise RuntimeError(
            f"Need at least 4 run records for stable split; found {len(records)}."
        )

    ratios = sorted(r.p99_p50_ratio for r in records)
    good_threshold = _quantile(ratios, args.good_quantile)
    bad_threshold = _quantile(ratios, args.bad_quantile)

    good_records = [r for r in records if r.p99_p50_ratio <= good_threshold]
    bad_records = [r for r in records if r.p99_p50_ratio >= bad_threshold]
    if not good_records or not bad_records:
        raise RuntimeError("Failed to form good/bad cohorts.")

    good_summary = summarize(good_records)
    bad_summary = summarize(bad_records)
    best_run = min(records, key=lambda r: r.p99_p50_ratio)
    fixes = pick_fixes(good_summary, bad_summary, best_run)

    report = {
        "workload": args.workload,
        "total_records": len(records),
        "good_threshold": good_threshold,
        "bad_threshold": bad_threshold,
        "good_summary": good_summary,
        "bad_summary": bad_summary,
        "fixes": fixes,
        "best_run": {
            "summary_path": best_run.summary_path,
            "run_dir": best_run.run_dir,
            "items_per_step": best_run.items_per_step,
            "repeat_idx": best_run.repeat_idx,
            "p99_p50_ratio": best_run.p99_p50_ratio,
            "throughput_items_per_sec": best_run.throughput_items_per_sec,
        },
    }

    output_md = Path(args.output_md).resolve()
    output_json = Path(args.output_json).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    output_md.write_text(
        render_markdown(
            workload=args.workload,
            total_records=len(records),
            good_summary=good_summary,
            bad_summary=bad_summary,
            good_threshold=good_threshold,
            bad_threshold=bad_threshold,
            fixes=fixes,
        )
    )
    print(f"output_md={output_md}")
    print(f"output_json={output_json}")


if __name__ == "__main__":
    main()
