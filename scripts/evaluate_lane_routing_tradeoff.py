#!/usr/bin/env python3
"""Evaluate global-lane vs profile-routed-lane tradeoffs from matrix artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _row_lookup(matrix_summary: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    by_workload = matrix_summary.get("aggregates", {}).get("by_workload", {})
    if not isinstance(by_workload, dict):
        raise ValueError("matrix_summary missing aggregates.by_workload")
    for workload, rows in by_workload.items():
        if not isinstance(rows, list):
            continue
        out[str(workload)] = {int(row["items_per_step"]): dict(row) for row in rows}
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(matrix_summary: dict[str, Any], adaptive_summary: dict[str, Any]) -> dict[str, Any]:
    rows = _row_lookup(matrix_summary)
    global_ips = int(adaptive_summary["global_recommended_items_per_step"])

    per_workload: list[dict[str, Any]] = []
    for rec in adaptive_summary.get("workloads", []):
        workload = str(rec["workload"])
        routed_ips = int(rec["recommended_items_per_step"])
        wl_rows = rows.get(workload)
        if not wl_rows:
            raise ValueError(f"Missing workload rows in matrix summary: {workload}")
        if global_ips not in wl_rows:
            raise ValueError(f"Missing global ips={global_ips} row for workload={workload}")
        if routed_ips not in wl_rows:
            raise ValueError(f"Missing routed ips={routed_ips} row for workload={workload}")

        global_row = wl_rows[global_ips]
        routed_row = wl_rows[routed_ips]

        per_workload.append(
            {
                "workload": workload,
                "num_items": int(rec.get("num_items", 0)),
                "global_items_per_step": global_ips,
                "routed_items_per_step": routed_ips,
                "global_throughput_items_per_sec": float(global_row["throughput_median_items_per_sec"]),
                "routed_throughput_items_per_sec": float(routed_row["throughput_median_items_per_sec"]),
                "global_p50_ms": float(global_row["latency_p50_median_ms"]),
                "routed_p50_ms": float(routed_row["latency_p50_median_ms"]),
                "global_p99_ms": float(global_row["latency_p99_median_ms"]),
                "routed_p99_ms": float(routed_row["latency_p99_median_ms"]),
                "global_failure_rate": float(global_row.get("failure_rate", 0.0)),
                "routed_failure_rate": float(routed_row.get("failure_rate", 0.0)),
            }
        )

    per_workload.sort(key=lambda r: r["workload"])

    global_tput = [r["global_throughput_items_per_sec"] for r in per_workload]
    routed_tput = [r["routed_throughput_items_per_sec"] for r in per_workload]
    global_p50 = [r["global_p50_ms"] for r in per_workload]
    routed_p50 = [r["routed_p50_ms"] for r in per_workload]
    global_p99 = [r["global_p99_ms"] for r in per_workload]
    routed_p99 = [r["routed_p99_ms"] for r in per_workload]

    unique_routed = sorted({int(r["routed_items_per_step"]) for r in per_workload})

    return {
        "global_items_per_step": global_ips,
        "routed_items_per_step_set": unique_routed,
        "lane_count_global": 1,
        "lane_count_routed": len(unique_routed),
        "routing_table_entries": len(per_workload),
        "aggregate": {
            "global_mean_throughput_items_per_sec": _mean(global_tput),
            "routed_mean_throughput_items_per_sec": _mean(routed_tput),
            "global_mean_p50_ms": _mean(global_p50),
            "routed_mean_p50_ms": _mean(routed_p50),
            "global_mean_p99_ms": _mean(global_p99),
            "routed_mean_p99_ms": _mean(routed_p99),
        },
        "workloads": per_workload,
    }


def render_markdown(payload: dict[str, Any], *, matrix_dir: str) -> str:
    agg = payload["aggregate"]
    lines: list[str] = []
    lines.append("# Global-Lane vs Profile-Routed-Lane Evaluation")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- matrix_dir: `{matrix_dir}`")
    lines.append(f"- global_items_per_step: `{payload['global_items_per_step']}`")
    lines.append(
        "- routed_items_per_step_set: `"
        + ",".join(str(x) for x in payload["routed_items_per_step_set"])
        + "`"
    )
    lines.append("")
    lines.append("## Aggregate Comparison (mean across workloads)")
    lines.append("")
    lines.append("| Strategy | mean_throughput_items_per_sec | mean_p50_ms | mean_p99_ms | lane_count |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Global lane | {agg['global_mean_throughput_items_per_sec']:.1f} | "
        f"{agg['global_mean_p50_ms']:.1f} | {agg['global_mean_p99_ms']:.1f} | "
        f"{payload['lane_count_global']} |"
    )
    lines.append(
        f"| Profile-routed lanes | {agg['routed_mean_throughput_items_per_sec']:.1f} | "
        f"{agg['routed_mean_p50_ms']:.1f} | {agg['routed_mean_p99_ms']:.1f} | "
        f"{payload['lane_count_routed']} |"
    )
    lines.append("")
    lines.append("## Per-Workload Comparison")
    lines.append("")
    lines.append(
        "| Workload | Global ips | Routed ips | Global tput | Routed tput | "
        "Global p50 | Routed p50 | Global p99 | Routed p99 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["workloads"]:
        lines.append(
            f"| {row['workload']} | {row['global_items_per_step']} | {row['routed_items_per_step']} | "
            f"{row['global_throughput_items_per_sec']:.1f} | {row['routed_throughput_items_per_sec']:.1f} | "
            f"{row['global_p50_ms']:.1f} | {row['routed_p50_ms']:.1f} | "
            f"{row['global_p99_ms']:.1f} | {row['routed_p99_ms']:.1f} |"
        )
    lines.append("")
    lines.append("## Operational Simplicity")
    lines.append("")
    lines.append(
        f"- Global lane requires a single serving lane (`ips={payload['global_items_per_step']}`), "
        f"minimal routing logic."
    )
    lines.append(
        f"- Profile-routed approach requires `{payload['lane_count_routed']}` lanes and "
        f"`{payload['routing_table_entries']}` workload-to-lane mappings."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--adaptive-summary", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    matrix_summary_path = Path(args.matrix_summary).expanduser().resolve()
    adaptive_summary_path = Path(args.adaptive_summary).expanduser().resolve()
    json_out = Path(args.json_out).expanduser().resolve()
    md_out = Path(args.md_out).expanduser().resolve()

    matrix_summary = _load_json(matrix_summary_path)
    adaptive_summary = _load_json(adaptive_summary_path)
    payload = evaluate(matrix_summary, adaptive_summary)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_out.write_text(render_markdown(payload, matrix_dir=str(matrix_summary_path.parent)) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
