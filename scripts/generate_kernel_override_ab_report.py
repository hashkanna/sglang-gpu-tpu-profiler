#!/usr/bin/env python3
"""Compare control vs override TPU matrix outputs for kernel override A/B."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArmData:
    name: str
    summary: dict[str, Any]
    gates: dict[str, Any]
    by_workload: dict[str, dict[int, dict[str, Any]]]
    gate_passing_ips: list[int]
    timed_compile_total: int
    shape_violation_total: int


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _to_arm(name: str, summary_path: Path, gates_path: Path) -> ArmData:
    summary = _load_json(summary_path)
    gates = _load_json(gates_path)
    by_workload_raw = (
        summary.get("aggregates", {}).get("by_workload", {})
        if isinstance(summary.get("aggregates"), dict)
        else {}
    )
    by_workload: dict[str, dict[int, dict[str, Any]]] = {}
    for wl_name, rows in by_workload_raw.items():
        wl_map: dict[int, dict[str, Any]] = {}
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            ips = int(row.get("items_per_step", 0))
            if ips > 0:
                wl_map[ips] = row
        if wl_map:
            by_workload[str(wl_name)] = wl_map

    gate_passing_ips: list[int] = []
    by_ips = gates.get("by_items_per_step", {})
    if isinstance(by_ips, dict):
        for ips_key, row in by_ips.items():
            if isinstance(row, dict) and bool(row.get("pass", False)):
                gate_passing_ips.append(int(ips_key))
    gate_passing_ips = sorted(set(gate_passing_ips))

    timed_compile_total = 0
    shape_violation_total = 0
    for run in summary.get("runs", []):
        if not isinstance(run, dict):
            continue
        timed_compile_total += int(run.get("compile_signals", {}).get("timed_xla_compile_count", 0))
        shape_violation_total += int(
            run.get("shape_signals", {}).get("shape_contract_violation_count", 0)
        )

    return ArmData(
        name=name,
        summary=summary,
        gates=gates,
        by_workload=by_workload,
        gate_passing_ips=gate_passing_ips,
        timed_compile_total=timed_compile_total,
        shape_violation_total=shape_violation_total,
    )


def _pick_best_row(wl_rows: dict[int, dict[str, Any]], allowed_ips: set[int]) -> tuple[int, dict[str, Any]]:
    candidates = [(ips, row) for ips, row in wl_rows.items() if ips in allowed_ips]
    if not candidates:
        # Fall back to all rows if no allowed intersection exists.
        candidates = list(wl_rows.items())
    if not candidates:
        raise ValueError("No rows available for workload")
    # Throughput primary, p99 secondary (lower better).
    candidates.sort(
        key=lambda x: (
            float(x[1].get("throughput_mean_items_per_sec", 0.0)),
            -float(x[1].get("latency_p99_median_ms", 1e18)),
        ),
        reverse=True,
    )
    return candidates[0]


def _collect_fixed_metrics(
    arm: ArmData, ips: int, workloads: list[str]
) -> tuple[float, float, dict[str, dict[str, float]]]:
    per_wl: dict[str, dict[str, float]] = {}
    tputs: list[float] = []
    p99s: list[float] = []
    for wl in workloads:
        row = arm.by_workload[wl][ips]
        tput = float(row["throughput_mean_items_per_sec"])
        p99 = float(row["latency_p99_median_ms"])
        tputs.append(tput)
        p99s.append(p99)
        per_wl[wl] = {"throughput_items_per_sec": tput, "p99_ms": p99}
    return sum(tputs) / len(tputs), sum(p99s) / len(p99s), per_wl


def _collect_best_metrics(
    arm: ArmData, workloads: list[str], allowed_ips: set[int]
) -> tuple[float, float, dict[str, dict[str, float | int]]]:
    per_wl: dict[str, dict[str, float | int]] = {}
    tputs: list[float] = []
    p99s: list[float] = []
    for wl in workloads:
        ips, row = _pick_best_row(arm.by_workload[wl], allowed_ips)
        tput = float(row["throughput_mean_items_per_sec"])
        p99 = float(row["latency_p99_median_ms"])
        tputs.append(tput)
        p99s.append(p99)
        per_wl[wl] = {"items_per_step": ips, "throughput_items_per_sec": tput, "p99_ms": p99}
    return sum(tputs) / len(tputs), sum(p99s) / len(p99s), per_wl


def _find_common_ips(control: ArmData, override: ArmData, workloads: list[str]) -> list[int]:
    common = set(control.gate_passing_ips).intersection(override.gate_passing_ips)
    valid: list[int] = []
    for ips in sorted(common):
        if all(ips in control.by_workload[w] and ips in override.by_workload[w] for w in workloads):
            valid.append(ips)
    return valid


def _compute_decision(
    control_best_mean_tput: float,
    control_best_mean_p99: float,
    override_best_mean_tput: float,
    override_best_mean_p99: float,
    override_compile_total: int,
    override_shape_total: int,
) -> tuple[str, str]:
    # Keep override only if it is clearly better and still hygienic.
    tput_gain = override_best_mean_tput - control_best_mean_tput
    p99_delta = override_best_mean_p99 - control_best_mean_p99
    hygiene_ok = override_compile_total == 0 and override_shape_total == 0
    if hygiene_ok and tput_gain > 0 and p99_delta <= 0:
        return (
            "KEEP_OVERRIDE",
            "Override improves best-of-gate throughput with non-worse p99 and keeps compile/shape hygiene.",
        )
    return (
        "ROLLBACK_OVERRIDE",
        "Override does not improve combined throughput+p99 outcome under compile/shape hygiene constraints.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate kernel override A/B comparison report.")
    parser.add_argument("--control-summary", required=True, type=Path)
    parser.add_argument("--control-gates", required=True, type=Path)
    parser.add_argument("--override-summary", required=True, type=Path)
    parser.add_argument("--override-gates", required=True, type=Path)
    parser.add_argument("--fixed-items-per-step", type=int, default=128)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    control = _to_arm("control", args.control_summary, args.control_gates)
    override = _to_arm("override", args.override_summary, args.override_gates)

    workloads = sorted(set(control.by_workload.keys()).intersection(override.by_workload.keys()))
    if not workloads:
        raise ValueError("No overlapping workloads between control and override")

    common_gate_ips = _find_common_ips(control, override, workloads)
    allowed_ips = set(common_gate_ips) if common_gate_ips else set(
        sorted(set(control.gate_passing_ips).intersection(override.gate_passing_ips))
    )

    fixed_ips = int(args.fixed_items_per_step)
    fixed_available = all(
        fixed_ips in control.by_workload[w] and fixed_ips in override.by_workload[w] for w in workloads
    )
    fixed_control: dict[str, Any] | None = None
    fixed_override: dict[str, Any] | None = None
    if fixed_available:
        c_tput, c_p99, c_per_wl = _collect_fixed_metrics(control, fixed_ips, workloads)
        o_tput, o_p99, o_per_wl = _collect_fixed_metrics(override, fixed_ips, workloads)
        fixed_control = {"mean_throughput_items_per_sec": c_tput, "mean_p99_ms": c_p99, "per_workload": c_per_wl}
        fixed_override = {"mean_throughput_items_per_sec": o_tput, "mean_p99_ms": o_p99, "per_workload": o_per_wl}

    c_best_tput, c_best_p99, c_best_per_wl = _collect_best_metrics(control, workloads, allowed_ips)
    o_best_tput, o_best_p99, o_best_per_wl = _collect_best_metrics(override, workloads, allowed_ips)

    decision, rationale = _compute_decision(
        control_best_mean_tput=c_best_tput,
        control_best_mean_p99=c_best_p99,
        override_best_mean_tput=o_best_tput,
        override_best_mean_p99=o_best_p99,
        override_compile_total=override.timed_compile_total,
        override_shape_total=override.shape_violation_total,
    )

    report = {
        "inputs": {
            "control_summary": str(args.control_summary),
            "control_gates": str(args.control_gates),
            "override_summary": str(args.override_summary),
            "override_gates": str(args.override_gates),
            "fixed_items_per_step": fixed_ips,
        },
        "workloads": workloads,
        "gate_passing_items_per_step": {
            "control": control.gate_passing_ips,
            "override": override.gate_passing_ips,
            "common": common_gate_ips,
        },
        "hygiene": {
            "control": {
                "timed_xla_compile_total": control.timed_compile_total,
                "shape_contract_violation_total": control.shape_violation_total,
            },
            "override": {
                "timed_xla_compile_total": override.timed_compile_total,
                "shape_contract_violation_total": override.shape_violation_total,
            },
        },
        "comparison": {
            "fixed_items_per_step": {
                "enabled": fixed_available,
                "items_per_step": fixed_ips,
                "control": fixed_control,
                "override": fixed_override,
                "delta_override_minus_control": (
                    {
                        "mean_throughput_items_per_sec": float(fixed_override["mean_throughput_items_per_sec"]) - float(fixed_control["mean_throughput_items_per_sec"]),  # type: ignore[index]
                        "mean_p99_ms": float(fixed_override["mean_p99_ms"]) - float(fixed_control["mean_p99_ms"]),  # type: ignore[index]
                    }
                    if fixed_available and fixed_control and fixed_override
                    else None
                ),
            },
            "best_of_gate_per_workload": {
                "allowed_items_per_step": sorted(allowed_ips),
                "control": {
                    "mean_throughput_items_per_sec": c_best_tput,
                    "mean_p99_ms": c_best_p99,
                    "per_workload": c_best_per_wl,
                },
                "override": {
                    "mean_throughput_items_per_sec": o_best_tput,
                    "mean_p99_ms": o_best_p99,
                    "per_workload": o_best_per_wl,
                },
                "delta_override_minus_control": {
                    "mean_throughput_items_per_sec": o_best_tput - c_best_tput,
                    "mean_p99_ms": o_best_p99 - c_best_p99,
                },
            },
        },
        "decision": {"action": decision, "rationale": rationale},
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    lines: list[str] = []
    lines.append("# T24 Kernel Override A/B Report")
    lines.append("")
    lines.append(f"- Decision: **{decision}**")
    lines.append(f"- Rationale: {rationale}")
    lines.append("")
    lines.append("## Gate + Hygiene")
    lines.append("")
    lines.append(
        f"- Gate-passing items_per_step: control={control.gate_passing_ips}, "
        f"override={override.gate_passing_ips}, common={common_gate_ips}"
    )
    lines.append(
        f"- Timed compile total: control={control.timed_compile_total}, "
        f"override={override.timed_compile_total}"
    )
    lines.append(
        f"- Shape-contract violations: control={control.shape_violation_total}, "
        f"override={override.shape_violation_total}"
    )
    lines.append("")
    lines.append("## Fixed Items/Step Comparison")
    lines.append("")
    if fixed_available and fixed_control and fixed_override:
        delta_tput = (
            float(fixed_override["mean_throughput_items_per_sec"])
            - float(fixed_control["mean_throughput_items_per_sec"])
        )
        delta_p99 = float(fixed_override["mean_p99_ms"]) - float(fixed_control["mean_p99_ms"])
        lines.append(f"- items_per_step={fixed_ips}")
        lines.append(
            f"- Mean throughput (items/s): control={fixed_control['mean_throughput_items_per_sec']:.3f}, "
            f"override={fixed_override['mean_throughput_items_per_sec']:.3f}, "
            f"delta={delta_tput:+.3f}"
        )
        lines.append(
            f"- Mean p99 (ms): control={fixed_control['mean_p99_ms']:.3f}, "
            f"override={fixed_override['mean_p99_ms']:.3f}, delta={delta_p99:+.3f}"
        )
    else:
        lines.append(
            f"- Skipped: items_per_step={fixed_ips} not available for every workload in both arms."
        )
    lines.append("")
    lines.append("## Best-of-Gate Per Workload")
    lines.append("")
    lines.append(
        f"- Allowed items_per_step set for best-of-gate selection: {sorted(allowed_ips)}"
    )
    lines.append(
        f"- Mean throughput (items/s): control={c_best_tput:.3f}, override={o_best_tput:.3f}, "
        f"delta={o_best_tput - c_best_tput:+.3f}"
    )
    lines.append(
        f"- Mean p99 (ms): control={c_best_p99:.3f}, override={o_best_p99:.3f}, "
        f"delta={o_best_p99 - c_best_p99:+.3f}"
    )
    lines.append("")
    lines.append("| Workload | Control ips | Control tput | Control p99 | Override ips | Override tput | Override p99 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for wl in workloads:
        c = c_best_per_wl[wl]
        o = o_best_per_wl[wl]
        lines.append(
            "| {wl} | {c_ips} | {c_tput:.3f} | {c_p99:.3f} | {o_ips} | {o_tput:.3f} | {o_p99:.3f} |".format(
                wl=wl,
                c_ips=int(c["items_per_step"]),
                c_tput=float(c["throughput_items_per_sec"]),
                c_p99=float(c["p99_ms"]),
                o_ips=int(o["items_per_step"]),
                o_tput=float(o["throughput_items_per_sec"]),
                o_p99=float(o["p99_ms"]),
            )
        )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Control summary: `{args.control_summary}`")
    lines.append(f"- Control gates: `{args.control_gates}`")
    lines.append(f"- Override summary: `{args.override_summary}`")
    lines.append(f"- Override gates: `{args.override_gates}`")
    lines.append("")
    args.md_out.write_text("\n".join(lines))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
