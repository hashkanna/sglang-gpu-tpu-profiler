#!/usr/bin/env python3
"""Generate TPU-vs-GPU phase attribution report for production scoring workloads."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return obj


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0
    return num / den


def _classify_bottleneck(host_share: float, device_share: float, queue_share: float) -> str:
    if queue_share >= 0.20:
        return "scheduling_queue"
    if host_share >= 0.22:
        return "host_orchestration"
    if device_share >= 0.70:
        return "device_compute"
    return "mixed"


def _action_for_bottleneck(kind: str) -> str:
    if kind == "device_compute":
        return "Prioritize device-path optimization (kernel/mask/data-layout tuning) for this lane."
    if kind == "host_orchestration":
        return "Prioritize host-side batching/dispatch orchestration reductions for this lane."
    if kind == "scheduling_queue":
        return "Prioritize scheduler queue/overlap adjustments to reduce wait time."
    return "Use combined host+scheduler tuning; no single phase dominates."


def _collect_tpu_metrics(
    matrix: dict[str, Any],
    items_per_step: int,
) -> dict[str, dict[str, float]]:
    runs = [
        run
        for run in matrix.get("runs", [])
        if int(run.get("items_per_step", -1)) == int(items_per_step) and not run.get("run_error")
    ]
    if not runs:
        raise ValueError(f"No successful runs found for items_per_step={items_per_step}")

    names = sorted(set.intersection(*[set(run.get("workloads", {}).keys()) for run in runs]))
    out: dict[str, dict[str, float]] = {}
    for name in names:
        wl_vals = [run["workloads"][name] for run in runs if name in run.get("workloads", {})]
        sp_vals = [v.get("score_phase", {}) for v in wl_vals]

        throughput = _median([float(v.get("throughput_items_per_sec", 0.0)) for v in wl_vals])
        p50 = _median([float(v.get("latency_p50_ms", 0.0)) for v in wl_vals])
        p99 = _median([float(v.get("latency_p99_ms", 0.0)) for v in wl_vals])

        host_ms = _median([float(s.get("host_orchestration_median_ms", 0.0)) for s in sp_vals])
        device_ms = _median([float(s.get("device_compute_median_ms", 0.0)) for s in sp_vals])
        queue_ms = _median([float(s.get("queue_wait_median_ms", 0.0)) for s in sp_vals])
        dispatches = _median([float(s.get("dispatches_median", 0.0)) for s in sp_vals])
        phase_util = _median([float(s.get("utilization_pct", 0.0)) for s in sp_vals])
        theoretical = _median([float(s.get("theoretical_items_per_sec", 0.0)) for s in sp_vals])

        phase_total = host_ms + device_ms + queue_ms
        host_share = _safe_div(host_ms, phase_total)
        device_share = _safe_div(device_ms, phase_total)
        queue_share = _safe_div(queue_ms, phase_total)

        out[name] = {
            "throughput_items_per_sec": throughput,
            "latency_p50_ms": p50,
            "latency_p99_ms": p99,
            "score_phase_host_orchestration_median_ms": host_ms,
            "score_phase_device_compute_median_ms": device_ms,
            "score_phase_queue_wait_median_ms": queue_ms,
            "score_phase_dispatches_median": dispatches,
            "score_phase_utilization_pct": phase_util,
            "score_phase_theoretical_items_per_sec": theoretical,
            "phase_host_share": host_share,
            "phase_device_share": device_share,
            "phase_queue_share": queue_share,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TPU-vs-GPU phase attribution report.")
    parser.add_argument("--tpu-matrix-summary", required=True, type=Path)
    parser.add_argument("--gpu-telemetry", required=True, type=Path)
    parser.add_argument("--tpu-items-per-step", type=int, default=None)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    tpu_matrix = _load_json(args.tpu_matrix_summary)
    gpu_obj = _load_json(args.gpu_telemetry)

    selected_ips = args.tpu_items_per_step
    if selected_ips is None:
        selected_ips = int(tpu_matrix.get("recommended", {}).get("items_per_step", 0))
    if selected_ips <= 0:
        raise ValueError("Unable to determine --tpu-items-per-step from matrix summary")

    tpu = _collect_tpu_metrics(tpu_matrix, selected_ips)
    gpu = {
        str(row["name"]): {
            "throughput_items_per_sec": float(row.get("throughput_items_per_sec", 0.0)),
            "latency_p50_ms": float(row.get("latency_p50_ms", 0.0)),
            "latency_p99_ms": float(row.get("latency_p99_ms", 0.0)),
            "gpu_util_median": float(row.get("gpu_util_median", 0.0)),
            "gpu_util_p95": float(row.get("gpu_util_p95", 0.0)),
            "mem_util_median": float(row.get("mem_util_median", 0.0)),
            "mem_used_median_mb": float(row.get("mem_used_median_mb", 0.0)),
            "power_median_w": float(row.get("power_median_w", 0.0)),
            "samples": int(row.get("samples", 0)),
        }
        for row in gpu_obj.get("workloads", [])
    }

    common = sorted(set(tpu.keys()) & set(gpu.keys()))
    if not common:
        raise ValueError("No overlapping workloads between TPU matrix summary and GPU telemetry")

    per_workload: list[dict[str, Any]] = []
    for name in common:
        t = tpu[name]
        g = gpu[name]
        tput_gap = float(t["throughput_items_per_sec"] - g["throughput_items_per_sec"])
        p50_gap = float(t["latency_p50_ms"] - g["latency_p50_ms"])
        p99_gap = float(t["latency_p99_ms"] - g["latency_p99_ms"])

        tput_gap_ratio = max(
            0.0,
            _safe_div(g["throughput_items_per_sec"] - t["throughput_items_per_sec"], g["throughput_items_per_sec"]),
        )
        p99_gap_ratio = max(0.0, _safe_div(t["latency_p99_ms"] - g["latency_p99_ms"], g["latency_p99_ms"]))
        p99_weight = 0.30 if tput_gap_ratio > 0.0 else 0.05
        opportunity_score = 0.85 * tput_gap_ratio + p99_weight * p99_gap_ratio

        bottleneck = _classify_bottleneck(
            host_share=float(t["phase_host_share"]),
            device_share=float(t["phase_device_share"]),
            queue_share=float(t["phase_queue_share"]),
        )
        per_workload.append(
            {
                "workload": name,
                "tpu": t,
                "gpu": g,
                "delta_tpu_minus_gpu": {
                    "throughput_items_per_sec": tput_gap,
                    "latency_p50_ms": p50_gap,
                    "latency_p99_ms": p99_gap,
                },
                "attribution": {
                    "dominant_phase": bottleneck,
                    "action": _action_for_bottleneck(bottleneck),
                    "opportunity_score": opportunity_score,
                    "throughput_gap_ratio": tput_gap_ratio,
                    "latency_p99_gap_ratio": p99_gap_ratio,
                },
            }
        )

    ranked = sorted(per_workload, key=lambda row: row["attribution"]["opportunity_score"], reverse=True)

    bottleneck_totals: dict[str, float] = {"device_compute": 0.0, "host_orchestration": 0.0, "scheduling_queue": 0.0, "mixed": 0.0}
    for row in ranked:
        bottleneck_totals[row["attribution"]["dominant_phase"]] += float(row["attribution"]["opportunity_score"])

    top = ranked[0]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "tpu_matrix_summary": str(args.tpu_matrix_summary),
            "gpu_telemetry": str(args.gpu_telemetry),
            "tpu_selected_items_per_step": selected_ips,
            "workload_count": len(common),
        },
        "attribution_ranked": ranked,
        "bottleneck_totals": bottleneck_totals,
        "top_intervention_candidate": {
            "workload": top["workload"],
            "dominant_phase": top["attribution"]["dominant_phase"],
            "action": top["attribution"]["action"],
            "opportunity_score": top["attribution"]["opportunity_score"],
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    lines: list[str] = []
    lines.append("# TPU vs GPU Phase Attribution Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- TPU matrix summary: `{args.tpu_matrix_summary}`")
    lines.append(f"- GPU telemetry: `{args.gpu_telemetry}`")
    lines.append(f"- TPU selected items_per_step: `{selected_ips}`")
    lines.append(f"- Overlapping workloads: `{len(common)}`")
    lines.append("")
    lines.append("## Ranked Bottleneck Opportunities")
    lines.append("")
    lines.append(
        "| workload | opportunity score | dominant phase | TPU-GPU throughput delta (items/s) | TPU-GPU p99 delta (ms) | TPU phase shares (host/device/queue) |"
    )
    lines.append("|---|---:|---|---:|---:|---|")
    for row in ranked:
        t = row["tpu"]
        d = row["delta_tpu_minus_gpu"]
        a = row["attribution"]
        lines.append(
            f"| {row['workload']} | {a['opportunity_score']:.3f} | {a['dominant_phase']} | "
            f"{d['throughput_items_per_sec']:+.3f} | {d['latency_p99_ms']:+.3f} | "
            f"{t['phase_host_share']:.3f}/{t['phase_device_share']:.3f}/{t['phase_queue_share']:.3f} |"
        )

    lines.append("")
    lines.append("## Bottleneck Totals")
    lines.append("")
    for key in ["device_compute", "host_orchestration", "scheduling_queue", "mixed"]:
        lines.append(f"- {key}: `{bottleneck_totals[key]:.3f}`")

    lines.append("")
    lines.append("## Top Intervention Candidate")
    lines.append("")
    lines.append(f"- workload: `{report['top_intervention_candidate']['workload']}`")
    lines.append(f"- dominant phase: `{report['top_intervention_candidate']['dominant_phase']}`")
    lines.append(f"- opportunity score: `{report['top_intervention_candidate']['opportunity_score']:.3f}`")
    lines.append(f"- action: {report['top_intervention_candidate']['action']}")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Opportunity score emphasizes TPU throughput deficits; p99 gap is heavily down-weighted when TPU throughput already exceeds GPU."
    )
    lines.append("- Dominant phase is derived from TPU score-phase shares only; GPU internal phases are approximated through host telemetry (utilization/power).")

    args.md_out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
