#!/usr/bin/env python3
"""Generate reliability soak comparison report across concurrency arms."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Arm:
    label: str
    concurrency: int
    summary_path: Path
    gates_path: Path
    summary: dict[str, Any]
    gates: dict[str, Any]


def _load(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _parse_arm(spec: str) -> Arm:
    # label:concurrency:summary:gates
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid arm spec: {spec}")
    label, conc_raw, summary_raw, gates_raw = parts
    return Arm(
        label=label,
        concurrency=int(conc_raw),
        summary_path=Path(summary_raw),
        gates_path=Path(gates_raw),
        summary=_load(Path(summary_raw)),
        gates=_load(Path(gates_raw)),
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _arm_metrics(arm: Arm, ips: int) -> dict[str, Any]:
    by_wl = arm.summary.get("aggregates", {}).get("by_workload", {})
    if not isinstance(by_wl, dict):
        raise ValueError(f"Missing by_workload in {arm.summary_path}")

    per_wl: dict[str, dict[str, float]] = {}
    tputs: list[float] = []
    p50s: list[float] = []
    p99s: list[float] = []
    fails: list[float] = []
    for wl, rows in by_wl.items():
        if not isinstance(rows, list):
            continue
        row = None
        for candidate in rows:
            if isinstance(candidate, dict) and int(candidate.get("items_per_step", 0)) == ips:
                row = candidate
                break
        if row is None:
            continue
        tput = float(row["throughput_mean_items_per_sec"])
        p50 = float(row["latency_p50_median_ms"])
        p99 = float(row["latency_p99_median_ms"])
        fail = float(row["failure_rate"])
        per_wl[str(wl)] = {
            "throughput_items_per_sec": tput,
            "p50_ms": p50,
            "p99_ms": p99,
            "failure_rate": fail,
        }
        tputs.append(tput)
        p50s.append(p50)
        p99s.append(p99)
        fails.append(fail)

    log = arm.summary.get("aggregates", {}).get("log_signals_by_items_per_step", {}).get(str(ips), {})
    if not isinstance(log, dict):
        log = {}
    gate_row = arm.gates.get("by_items_per_step", {}).get(str(ips), {})
    gate_pass = bool(isinstance(gate_row, dict) and gate_row.get("pass", False))
    return {
        "label": arm.label,
        "concurrency": arm.concurrency,
        "items_per_step": ips,
        "gate_pass": gate_pass,
        "mean_throughput_items_per_sec": _mean(tputs),
        "mean_p50_ms": _mean(p50s),
        "mean_p99_ms": _mean(p99s),
        "mean_failure_rate": _mean(fails),
        "timed_xla_compile_total": int(log.get("timed_xla_compile_total", 0)),
        "shape_contract_violation_total": int(log.get("shape_contract_violation_total", 0)),
        "missing_cache_handle_total": int(log.get("missing_cache_handle_total", 0)),
        "per_workload": per_wl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate production soak report.")
    parser.add_argument("--arm", action="append", required=True, help="label:concurrency:summary:gates")
    parser.add_argument("--items-per-step", type=int, default=128)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    metrics = [_arm_metrics(_parse_arm(spec), args.items_per_step) for spec in args.arm]
    if not metrics:
        raise ValueError("No arms provided")

    reliable = [
        m
        for m in metrics
        if m["gate_pass"]
        and m["mean_failure_rate"] == 0
        and m["timed_xla_compile_total"] == 0
        and m["shape_contract_violation_total"] == 0
    ]
    if reliable:
        best_tput = max(reliable, key=lambda x: x["mean_throughput_items_per_sec"])
        best_p99 = min(reliable, key=lambda x: x["mean_p99_ms"])
        if best_tput["label"] == best_p99["label"]:
            recommendation = {
                "label": best_tput["label"],
                "mode": "balanced",
                "rationale": "Single arm is best on both throughput and p99 among reliable arms.",
            }
        else:
            recommendation = {
                "label": best_p99["label"],
                "mode": "latency_first",
                "alternate_label": best_tput["label"],
                "rationale": (
                    "Selected lowest-p99 reliable arm as default; alternate arm offers higher throughput "
                    "with higher tail latency."
                ),
            }
    else:
        recommendation = {
            "label": metrics[0]["label"],
            "mode": "fallback",
            "rationale": "No fully reliable arms; fallback to first arm for manual triage.",
        }

    report = {
        "items_per_step": args.items_per_step,
        "arms": metrics,
        "recommendation": recommendation,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    lines: list[str] = []
    lines.append("# T27 Production Soak Report")
    lines.append("")
    lines.append(f"- Recommendation: `{recommendation['label']}` ({recommendation['mode']})")
    lines.append(f"- Rationale: {recommendation['rationale']}")
    if "alternate_label" in recommendation:
        lines.append(f"- Throughput-oriented alternate: `{recommendation['alternate_label']}`")
    lines.append("")
    lines.append("## Arm Summary")
    lines.append("")
    lines.append(
        "| Label | Concurrency | Gate | Mean Tput | Mean p50 | Mean p99 | Mean Failure | Timed Compile | Shape Violations | Missing Cache Handles |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in sorted(metrics, key=lambda x: int(x["concurrency"])):
        lines.append(
            "| {label} | {conc} | {gate} | {tput:.3f} | {p50:.3f} | {p99:.3f} | {fr:.4f} | {compile} | {shape} | {miss} |".format(
                label=m["label"],
                conc=int(m["concurrency"]),
                gate=str(m["gate_pass"]).lower(),
                tput=float(m["mean_throughput_items_per_sec"]),
                p50=float(m["mean_p50_ms"]),
                p99=float(m["mean_p99_ms"]),
                fr=float(m["mean_failure_rate"]),
                compile=int(m["timed_xla_compile_total"]),
                shape=int(m["shape_contract_violation_total"]),
                miss=int(m["missing_cache_handle_total"]),
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- All reported arms run the same production workload set and `items_per_step=128`; only concurrency differs."
    )
    lines.append("")
    args.md_out.write_text("\n".join(lines))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
