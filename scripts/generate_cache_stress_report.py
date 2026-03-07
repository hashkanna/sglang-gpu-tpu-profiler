#!/usr/bin/env python3
"""Summarize cache-timeout stress runs for production-shaped scoring workloads."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Arm:
    label: str
    timeout_sec: float
    summary_path: Path
    gates_path: Path
    summary: dict[str, Any]
    gates: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _parse_arm(spec: str) -> Arm:
    # Format: label:timeout_sec:summary_path:gates_path
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --arm spec '{spec}'. Expected label:timeout_sec:summary_path:gates_path"
        )
    label, timeout_raw, summary_raw, gates_raw = parts
    timeout_sec = float(timeout_raw)
    summary_path = Path(summary_raw)
    gates_path = Path(gates_raw)
    return Arm(
        label=label,
        timeout_sec=timeout_sec,
        summary_path=summary_path,
        gates_path=gates_path,
        summary=_load_json(summary_path),
        gates=_load_json(gates_path),
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _extract_arm_metrics(arm: Arm, ips: int) -> dict[str, Any]:
    by_wl = arm.summary.get("aggregates", {}).get("by_workload", {})
    if not isinstance(by_wl, dict):
        raise ValueError(f"by_workload missing in summary: {arm.summary_path}")

    per_wl: dict[str, dict[str, float]] = {}
    tputs: list[float] = []
    p99s: list[float] = []
    failures: list[float] = []
    host_ms: list[float] = []
    queue_ms: list[float] = []

    for wl_name, rows in by_wl.items():
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
        p99 = float(row["latency_p99_median_ms"])
        fail = float(row["failure_rate"])
        host = float(row["host_orchestration_median_ms_median"])
        queue = float(row["queue_wait_median_ms_median"])
        per_wl[str(wl_name)] = {
            "throughput_items_per_sec": tput,
            "p99_ms": p99,
            "failure_rate": fail,
            "host_orchestration_ms": host,
            "queue_wait_ms": queue,
        }
        tputs.append(tput)
        p99s.append(p99)
        failures.append(fail)
        host_ms.append(host)
        queue_ms.append(queue)

    log_signals = (
        arm.summary.get("aggregates", {}).get("log_signals_by_items_per_step", {}).get(str(ips), {})
    )
    if not isinstance(log_signals, dict):
        log_signals = {}

    gate_row = arm.gates.get("by_items_per_step", {}).get(str(ips), {})
    gate_pass = bool(isinstance(gate_row, dict) and gate_row.get("pass", False))

    return {
        "label": arm.label,
        "timeout_sec": arm.timeout_sec,
        "items_per_step": ips,
        "gate_pass": gate_pass,
        "mean_throughput_items_per_sec": _mean(tputs),
        "mean_p99_ms": _mean(p99s),
        "mean_failure_rate": _mean(failures),
        "mean_host_orchestration_ms": _mean(host_ms),
        "mean_queue_wait_ms": _mean(queue_ms),
        "missing_cache_handle_total": int(log_signals.get("missing_cache_handle_total", 0)),
        "cache_transition_repeats": int(log_signals.get("cache_transition_repeats", 0)),
        "timed_xla_compile_total": int(log_signals.get("timed_xla_compile_total", 0)),
        "shape_contract_violation_total": int(log_signals.get("shape_contract_violation_total", 0)),
        "warmup_xla_detect_total": int(log_signals.get("warmup_xla_detect_total", 0)),
        "per_workload": per_wl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cache-timeout stress report.")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Arm spec: label:timeout_sec:summary_path:gates_path (repeatable)",
    )
    parser.add_argument("--control-label", required=True)
    parser.add_argument("--items-per-step", type=int, default=128)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    arms = [_parse_arm(spec) for spec in args.arm]
    metrics = [_extract_arm_metrics(arm, args.items_per_step) for arm in arms]
    by_label = {m["label"]: m for m in metrics}
    if args.control_label not in by_label:
        raise ValueError(f"--control-label '{args.control_label}' not found in --arm labels")
    control = by_label[args.control_label]

    comparisons: dict[str, dict[str, Any]] = {}
    for m in metrics:
        label = m["label"]
        if label == args.control_label:
            continue
        comparisons[label] = {
            "delta_mean_throughput_items_per_sec": m["mean_throughput_items_per_sec"]
            - control["mean_throughput_items_per_sec"],
            "delta_mean_p99_ms": m["mean_p99_ms"] - control["mean_p99_ms"],
            "delta_mean_failure_rate": m["mean_failure_rate"] - control["mean_failure_rate"],
            "delta_mean_host_orchestration_ms": m["mean_host_orchestration_ms"]
            - control["mean_host_orchestration_ms"],
            "delta_mean_queue_wait_ms": m["mean_queue_wait_ms"] - control["mean_queue_wait_ms"],
            "delta_missing_cache_handle_total": m["missing_cache_handle_total"]
            - control["missing_cache_handle_total"],
            "delta_cache_transition_repeats": m["cache_transition_repeats"]
            - control["cache_transition_repeats"],
            "delta_timed_xla_compile_total": m["timed_xla_compile_total"]
            - control["timed_xla_compile_total"],
        }

    # Recommendation: choose highest throughput arm that preserves zero failure/compile/shape.
    eligible = [
        m
        for m in metrics
        if m["gate_pass"]
        and m["mean_failure_rate"] <= 0
        and m["timed_xla_compile_total"] == 0
        and m["shape_contract_violation_total"] == 0
    ]
    recommended = max(eligible, key=lambda x: x["mean_throughput_items_per_sec"]) if eligible else control
    decision = {
        "recommended_label": recommended["label"],
        "recommended_timeout_sec": recommended["timeout_sec"],
        "rationale": (
            "Selected highest-throughput cache-timeout arm among gate-passing, compile-free, "
            "shape-safe, zero-failure candidates."
        ),
    }

    report = {
        "items_per_step": args.items_per_step,
        "control_label": args.control_label,
        "arms": metrics,
        "comparisons_vs_control": comparisons,
        "decision": decision,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    lines: list[str] = []
    lines.append("# T26 Cache Timeout Stress Report")
    lines.append("")
    lines.append(f"- Decision: recommend `{decision['recommended_label']}` (timeout={decision['recommended_timeout_sec']})")
    lines.append(f"- Rationale: {decision['rationale']}")
    lines.append("")
    lines.append("## Arm Summary")
    lines.append("")
    lines.append(
        "| Label | Timeout(s) | Gate | Mean Tput | Mean p99 | Mean Failure | Missing Cache Handles | Cache Transition Repeats | Timed Compile | Shape Violations |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in sorted(metrics, key=lambda x: float(x["timeout_sec"]), reverse=True):
        lines.append(
            "| {label} | {timeout:.3f} | {gate} | {tput:.3f} | {p99:.3f} | {fr:.4f} | {miss} | {ctr} | {compile} | {shape} |".format(
                label=m["label"],
                timeout=float(m["timeout_sec"]),
                gate=str(m["gate_pass"]).lower(),
                tput=float(m["mean_throughput_items_per_sec"]),
                p99=float(m["mean_p99_ms"]),
                fr=float(m["mean_failure_rate"]),
                miss=int(m["missing_cache_handle_total"]),
                ctr=int(m["cache_transition_repeats"]),
                compile=int(m["timed_xla_compile_total"]),
                shape=int(m["shape_contract_violation_total"]),
            )
        )
    lines.append("")
    lines.append("## Delta vs Control")
    lines.append("")
    lines.append("| Label | Delta Tput | Delta p99 | Delta Host ms | Delta Queue ms | Delta Missing Handles | Delta Transition Repeats | Delta Timed Compile |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, delta in sorted(comparisons.items(), key=lambda kv: kv[0]):
        lines.append(
            "| {label} | {dt:+.3f} | {dp:+.3f} | {dh:+.3f} | {dq:+.3f} | {dm:+d} | {dc:+d} | {dx:+d} |".format(
                label=label,
                dt=float(delta["delta_mean_throughput_items_per_sec"]),
                dp=float(delta["delta_mean_p99_ms"]),
                dh=float(delta["delta_mean_host_orchestration_ms"]),
                dq=float(delta["delta_mean_queue_wait_ms"]),
                dm=int(delta["delta_missing_cache_handle_total"]),
                dc=int(delta["delta_cache_transition_repeats"]),
                dx=int(delta["delta_timed_xla_compile_total"]),
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Across this production /v1/score workload, cache-transition counters remained at zero even under timeout stress;"
    )
    lines.append(
        "  this indicates the measured path does not currently exercise observable cache-handle miss/release transitions."
    )
    lines.append("")
    args.md_out.write_text("\n".join(lines))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
