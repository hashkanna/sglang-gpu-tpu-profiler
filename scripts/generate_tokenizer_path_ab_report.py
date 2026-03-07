#!/usr/bin/env python3
"""Generate control vs tokenizer-batch-path A/B report from matrix summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _row_by_workload(summary: dict[str, Any], ips: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_wl = summary.get("aggregates", {}).get("by_workload", {})
    if not isinstance(by_wl, dict):
        return out
    for wl_name, rows in by_wl.items():
        if not isinstance(rows, list):
            continue
        matched = None
        for row in rows:
            if isinstance(row, dict) and int(row.get("items_per_step", 0)) == ips:
                matched = row
                break
        if matched is not None:
            out[str(wl_name)] = matched
    return out


def _gate_ok(gates: dict[str, Any], ips: int) -> bool:
    by_ips = gates.get("by_items_per_step", {})
    row = by_ips.get(str(ips), {}) if isinstance(by_ips, dict) else {}
    return bool(isinstance(row, dict) and row.get("pass", False))


def _totals(summary: dict[str, Any]) -> tuple[int, int]:
    compile_total = 0
    shape_total = 0
    for run in summary.get("runs", []):
        if not isinstance(run, dict):
            continue
        compile_total += int(run.get("compile_signals", {}).get("timed_xla_compile_count", 0))
        shape_total += int(run.get("shape_signals", {}).get("shape_contract_violation_count", 0))
    return compile_total, shape_total


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tokenizer path A/B report.")
    parser.add_argument("--control-summary", required=True, type=Path)
    parser.add_argument("--control-gates", required=True, type=Path)
    parser.add_argument("--variant-summary", required=True, type=Path)
    parser.add_argument("--variant-gates", required=True, type=Path)
    parser.add_argument("--items-per-step", type=int, default=128)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    control_summary = _load(args.control_summary)
    control_gates = _load(args.control_gates)
    variant_summary = _load(args.variant_summary)
    variant_gates = _load(args.variant_gates)

    ips = int(args.items_per_step)
    control_rows = _row_by_workload(control_summary, ips)
    variant_rows = _row_by_workload(variant_summary, ips)
    workloads = sorted(set(control_rows.keys()).intersection(variant_rows.keys()))
    if not workloads:
        raise ValueError(f"No overlapping workloads found at items_per_step={ips}")

    rows: list[dict[str, Any]] = []
    for wl in workloads:
        c = control_rows[wl]
        v = variant_rows[wl]
        c_tput = float(c["throughput_mean_items_per_sec"])
        v_tput = float(v["throughput_mean_items_per_sec"])
        c_p99 = float(c["latency_p99_median_ms"])
        v_p99 = float(v["latency_p99_median_ms"])
        c_host = float(c["host_orchestration_median_ms_median"])
        v_host = float(v["host_orchestration_median_ms_median"])
        c_queue = float(c["queue_wait_median_ms_median"])
        v_queue = float(v["queue_wait_median_ms_median"])
        rows.append(
            {
                "workload": wl,
                "control": {
                    "throughput_items_per_sec": c_tput,
                    "p99_ms": c_p99,
                    "host_orchestration_ms": c_host,
                    "queue_wait_ms": c_queue,
                },
                "variant": {
                    "throughput_items_per_sec": v_tput,
                    "p99_ms": v_p99,
                    "host_orchestration_ms": v_host,
                    "queue_wait_ms": v_queue,
                },
                "delta_variant_minus_control": {
                    "throughput_items_per_sec": v_tput - c_tput,
                    "p99_ms": v_p99 - c_p99,
                    "host_orchestration_ms": v_host - c_host,
                    "queue_wait_ms": v_queue - c_queue,
                },
            }
        )

    c_compile, c_shape = _totals(control_summary)
    v_compile, v_shape = _totals(variant_summary)

    c_tputs = [r["control"]["throughput_items_per_sec"] for r in rows]
    v_tputs = [r["variant"]["throughput_items_per_sec"] for r in rows]
    c_p99s = [r["control"]["p99_ms"] for r in rows]
    v_p99s = [r["variant"]["p99_ms"] for r in rows]
    c_hosts = [r["control"]["host_orchestration_ms"] for r in rows]
    v_hosts = [r["variant"]["host_orchestration_ms"] for r in rows]
    c_queues = [r["control"]["queue_wait_ms"] for r in rows]
    v_queues = [r["variant"]["queue_wait_ms"] for r in rows]

    agg = {
        "control": {
            "mean_throughput_items_per_sec": _mean(c_tputs),
            "mean_p99_ms": _mean(c_p99s),
            "mean_host_orchestration_ms": _mean(c_hosts),
            "mean_queue_wait_ms": _mean(c_queues),
        },
        "variant": {
            "mean_throughput_items_per_sec": _mean(v_tputs),
            "mean_p99_ms": _mean(v_p99s),
            "mean_host_orchestration_ms": _mean(v_hosts),
            "mean_queue_wait_ms": _mean(v_queues),
        },
    }
    agg_delta = {
        "mean_throughput_items_per_sec": agg["variant"]["mean_throughput_items_per_sec"]
        - agg["control"]["mean_throughput_items_per_sec"],
        "mean_p99_ms": agg["variant"]["mean_p99_ms"] - agg["control"]["mean_p99_ms"],
        "mean_host_orchestration_ms": agg["variant"]["mean_host_orchestration_ms"]
        - agg["control"]["mean_host_orchestration_ms"],
        "mean_queue_wait_ms": agg["variant"]["mean_queue_wait_ms"]
        - agg["control"]["mean_queue_wait_ms"],
    }

    control_gate_pass = _gate_ok(control_gates, ips)
    variant_gate_pass = _gate_ok(variant_gates, ips)

    # Keep only when variant is clearly better overall and hygienic.
    decision = "ROLLBACK_VARIANT"
    rationale = (
        "Tokenizer batch-send path did not improve aggregate throughput+p99 together under "
        "production-like scoring ingress; keep baseline path."
    )
    if (
        control_gate_pass
        and variant_gate_pass
        and v_compile == 0
        and v_shape == 0
        and agg_delta["mean_throughput_items_per_sec"] > 0
        and agg_delta["mean_p99_ms"] <= 0
        and agg_delta["mean_host_orchestration_ms"] <= 0
    ):
        decision = "KEEP_VARIANT"
        rationale = (
            "Tokenizer batch-send path improved throughput with non-worse p99 and lower/equal "
            "host orchestration under compile/shape hygiene constraints."
        )

    report = {
        "inputs": {
            "control_summary": str(args.control_summary),
            "control_gates": str(args.control_gates),
            "variant_summary": str(args.variant_summary),
            "variant_gates": str(args.variant_gates),
            "items_per_step": ips,
        },
        "gate_pass": {"control": control_gate_pass, "variant": variant_gate_pass},
        "hygiene": {
            "control": {
                "timed_xla_compile_total": c_compile,
                "shape_contract_violation_total": c_shape,
            },
            "variant": {
                "timed_xla_compile_total": v_compile,
                "shape_contract_violation_total": v_shape,
            },
        },
        "aggregates": {
            "control": agg["control"],
            "variant": agg["variant"],
            "delta_variant_minus_control": agg_delta,
        },
        "per_workload": rows,
        "decision": {"action": decision, "rationale": rationale},
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    lines: list[str] = []
    lines.append("# T25 Tokenizer Path A/B Report")
    lines.append("")
    lines.append(f"- Decision: **{decision}**")
    lines.append(f"- Rationale: {rationale}")
    lines.append("")
    lines.append("## Gate + Hygiene")
    lines.append("")
    lines.append(f"- items_per_step: {ips}")
    lines.append(f"- Gate pass: control={control_gate_pass}, variant={variant_gate_pass}")
    lines.append(
        f"- Timed compile total: control={c_compile}, variant={v_compile}; "
        f"shape violations: control={c_shape}, variant={v_shape}"
    )
    lines.append("")
    lines.append("## Aggregate Means (Across Workloads)")
    lines.append("")
    lines.append(
        f"- Throughput (items/s): control={agg['control']['mean_throughput_items_per_sec']:.3f}, "
        f"variant={agg['variant']['mean_throughput_items_per_sec']:.3f}, "
        f"delta={agg_delta['mean_throughput_items_per_sec']:+.3f}"
    )
    lines.append(
        f"- p99 (ms): control={agg['control']['mean_p99_ms']:.3f}, "
        f"variant={agg['variant']['mean_p99_ms']:.3f}, delta={agg_delta['mean_p99_ms']:+.3f}"
    )
    lines.append(
        f"- Host orchestration (ms): control={agg['control']['mean_host_orchestration_ms']:.3f}, "
        f"variant={agg['variant']['mean_host_orchestration_ms']:.3f}, "
        f"delta={agg_delta['mean_host_orchestration_ms']:+.3f}"
    )
    lines.append(
        f"- Queue wait (ms): control={agg['control']['mean_queue_wait_ms']:.3f}, "
        f"variant={agg['variant']['mean_queue_wait_ms']:.3f}, "
        f"delta={agg_delta['mean_queue_wait_ms']:+.3f}"
    )
    lines.append("")
    lines.append(
        "| Workload | C Throughput | V Throughput | Delta | C p99 | V p99 | Delta | C Host ms | V Host ms | Delta |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        wl = row["workload"]
        c = row["control"]
        v = row["variant"]
        d = row["delta_variant_minus_control"]
        lines.append(
            "| {wl} | {ct:.3f} | {vt:.3f} | {dt:+.3f} | {cp:.3f} | {vp:.3f} | {dp:+.3f} | {ch:.3f} | {vh:.3f} | {dh:+.3f} |".format(
                wl=wl,
                ct=c["throughput_items_per_sec"],
                vt=v["throughput_items_per_sec"],
                dt=d["throughput_items_per_sec"],
                cp=c["p99_ms"],
                vp=v["p99_ms"],
                dp=d["p99_ms"],
                ch=c["host_orchestration_ms"],
                vh=v["host_orchestration_ms"],
                dh=d["host_orchestration_ms"],
            )
        )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Control summary: `{args.control_summary}`")
    lines.append(f"- Variant summary: `{args.variant_summary}`")
    lines.append(f"- Control gates: `{args.control_gates}`")
    lines.append(f"- Variant gates: `{args.variant_gates}`")
    lines.append("")
    args.md_out.write_text("\n".join(lines))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
