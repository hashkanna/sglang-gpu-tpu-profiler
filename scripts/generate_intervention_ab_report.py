#!/usr/bin/env python3
"""Generate before/after report for a runtime intervention matrix A/B."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return obj


def _workload_rows(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    rows_by_wl = summary.get("aggregates", {}).get("by_workload", {})
    if not isinstance(rows_by_wl, dict):
        return out
    for wl, rows in rows_by_wl.items():
        if not isinstance(rows, list) or not rows:
            continue
        row = rows[0]
        out[str(wl)] = {
            "throughput_items_per_sec": float(row.get("throughput_median_items_per_sec", 0.0)),
            "latency_p50_ms": float(row.get("latency_p50_median_ms", 0.0)),
            "latency_p99_ms": float(row.get("latency_p99_median_ms", 0.0)),
        }
    return out


def _gate_snapshot(gates: dict[str, Any], items_per_step: int) -> dict[str, Any]:
    row = gates.get("by_items_per_step", {}).get(str(items_per_step), {})
    checks = row.get("checks", {}) if isinstance(row, dict) else {}

    def _val(name: str) -> float:
        cell = checks.get(name, {}) if isinstance(checks, dict) else {}
        if not isinstance(cell, dict):
            return 0.0
        value = cell.get("value", 0.0)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def _pass(name: str) -> bool:
        cell = checks.get(name, {}) if isinstance(checks, dict) else {}
        return bool(cell.get("pass", False)) if isinstance(cell, dict) else False

    return {
        "pass": bool(row.get("pass", False)) if isinstance(row, dict) else False,
        "timed_xla_compilation": _val("timed_xla_compilation"),
        "shape_contract_violations": _val("shape_contract"),
        "missing_cache_handle": _val("missing_cache_handle"),
        "score_fastpath_success_pass": _pass("score_fastpath_success"),
        "score_mode_label_only_pass": _pass("score_mode_label_only"),
    }


def _parse_fastpath_lane_counts(matrix_dir: Path) -> dict[str, Any]:
    log_paths = sorted(matrix_dir.glob("runs/*/artifacts/tpu/tpu_server_full.log"))
    lane_counts: dict[int, int] = {}
    max_tokens_by_lane: dict[int, int] = {}
    pattern = re.compile(r"fastpath_items_per_step=(\d+) fastpath_max_total_tokens=(\d+)")
    for log_path in log_paths:
        text = log_path.read_text(errors="ignore")
        for ips_raw, tok_raw in pattern.findall(text):
            ips = int(ips_raw)
            tok = int(tok_raw)
            lane_counts[ips] = lane_counts.get(ips, 0) + 1
            max_tokens_by_lane[ips] = max(max_tokens_by_lane.get(ips, 0), tok)

    return {
        "log_files": [str(p) for p in log_paths],
        "lane_counts": {str(k): v for k, v in sorted(lane_counts.items())},
        "max_tokens_by_lane": {str(k): v for k, v in sorted(max_tokens_by_lane.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate intervention before/after report.")
    parser.add_argument("--before-summary", required=True, type=Path)
    parser.add_argument("--after-summary", required=True, type=Path)
    parser.add_argument("--before-gates", required=True, type=Path)
    parser.add_argument("--after-gates", required=True, type=Path)
    parser.add_argument("--before-matrix-dir", required=True, type=Path)
    parser.add_argument("--after-matrix-dir", required=True, type=Path)
    parser.add_argument("--items-per-step", type=int, default=96)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    before_summary = _load_json(args.before_summary)
    after_summary = _load_json(args.after_summary)
    before_gates = _load_json(args.before_gates)
    after_gates = _load_json(args.after_gates)

    before_rows = _workload_rows(before_summary)
    after_rows = _workload_rows(after_summary)
    workloads = sorted(set(before_rows.keys()) & set(after_rows.keys()))
    if not workloads:
        raise ValueError("No overlapping workloads between before and after summaries")

    per_workload: list[dict[str, Any]] = []
    for wl in workloads:
        b = before_rows[wl]
        a = after_rows[wl]
        per_workload.append(
            {
                "workload": wl,
                "before": b,
                "after": a,
                "delta_after_minus_before": {
                    "throughput_items_per_sec": float(a["throughput_items_per_sec"] - b["throughput_items_per_sec"]),
                    "latency_p50_ms": float(a["latency_p50_ms"] - b["latency_p50_ms"]),
                    "latency_p99_ms": float(a["latency_p99_ms"] - b["latency_p99_ms"]),
                },
            }
        )

    tput_delta_sum = sum(row["delta_after_minus_before"]["throughput_items_per_sec"] for row in per_workload)
    p50_delta_sum = sum(row["delta_after_minus_before"]["latency_p50_ms"] for row in per_workload)
    p99_delta_sum = sum(row["delta_after_minus_before"]["latency_p99_ms"] for row in per_workload)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "before_summary": str(args.before_summary),
            "after_summary": str(args.after_summary),
            "before_gates": str(args.before_gates),
            "after_gates": str(args.after_gates),
            "before_matrix_dir": str(args.before_matrix_dir),
            "after_matrix_dir": str(args.after_matrix_dir),
            "items_per_step": args.items_per_step,
        },
        "per_workload": per_workload,
        "aggregate_delta_after_minus_before": {
            "throughput_items_per_sec_sum": float(tput_delta_sum),
            "latency_p50_ms_sum": float(p50_delta_sum),
            "latency_p99_ms_sum": float(p99_delta_sum),
        },
        "gate_hygiene": {
            "before": _gate_snapshot(before_gates, args.items_per_step),
            "after": _gate_snapshot(after_gates, args.items_per_step),
        },
        "fastpath_lane_activation": {
            "before": _parse_fastpath_lane_counts(args.before_matrix_dir),
            "after": _parse_fastpath_lane_counts(args.after_matrix_dir),
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    lines: list[str] = []
    lines.append("# T39 Intervention A/B Report")
    lines.append("")
    lines.append("## Intervention")
    lines.append("")
    lines.append(
        "Adaptive long-sequence score-from-cache v2 lane: cap fastpath `items_per_step` to long-lane value "
        "when `(query_len + max_item_len)` crosses threshold."
    )
    lines.append("")
    lines.append("## Workload Deltas (after - before)")
    lines.append("")
    lines.append("| workload | throughput delta (items/s) | p50 delta (ms) | p99 delta (ms) |")
    lines.append("|---|---:|---:|---:|")
    for row in report["per_workload"]:
        d = row["delta_after_minus_before"]
        lines.append(
            f"| {row['workload']} | {d['throughput_items_per_sec']:+.3f} | "
            f"{d['latency_p50_ms']:+.3f} | {d['latency_p99_ms']:+.3f} |"
        )

    agg = report["aggregate_delta_after_minus_before"]
    lines.append("")
    lines.append("## Aggregate Delta")
    lines.append("")
    lines.append(f"- throughput_items_per_sec_sum: `{agg['throughput_items_per_sec_sum']:+.3f}`")
    lines.append(f"- latency_p50_ms_sum: `{agg['latency_p50_ms_sum']:+.3f}`")
    lines.append(f"- latency_p99_ms_sum: `{agg['latency_p99_ms_sum']:+.3f}`")

    lines.append("")
    lines.append("## Gate Hygiene")
    lines.append("")
    for side in ["before", "after"]:
        g = report["gate_hygiene"][side]
        lines.append(
            f"- {side}: pass={g['pass']}, timed_xla={g['timed_xla_compilation']}, "
            f"shape_violations={g['shape_contract_violations']}, missing_cache_handle={g['missing_cache_handle']}"
        )

    lines.append("")
    lines.append("## Fastpath Lane Activation")
    lines.append("")
    for side in ["before", "after"]:
        lane = report["fastpath_lane_activation"][side]
        lines.append(f"- {side} lane_counts: `{lane['lane_counts']}`")
        lines.append(f"- {side} max_tokens_by_lane: `{lane['max_tokens_by_lane']}`")

    args.md_out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
