#!/usr/bin/env python3
"""Generate per-profile adaptive lane recommendations from a production matrix run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pr28_baseline import baseline_workloads, load_baseline, matrix_defaults


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Production Adaptive Lane Summary: {summary['name']}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- generated_at_utc: `{summary['generated_at_utc']}`")
    lines.append(f"- matrix_summary: `{summary['matrix_summary_path']}`")
    lines.append(f"- matrix_gates: `{summary['matrix_gates_path']}`")
    lines.append(f"- global_recommended_items_per_step: `{summary['global_recommended_items_per_step']}`")
    lines.append("")
    lines.append("## Gate-Passing Items-Per-Step Candidates")
    lines.append("")
    lines.append(
        "- "
        + ", ".join(str(v) for v in summary["gate_passing_items_per_step"])
    )
    lines.append("")
    lines.append("## Per-Profile Recommendations")
    lines.append("")
    lines.append(
        "| Workload | items_per_step | max_running_requests | extend_batch_size | "
        "throughput_items_per_sec | p50_ms | p99_ms | p99/p50 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["workloads"]:
        lines.append(
            f"| {row['workload']} | {row['recommended_items_per_step']} | "
            f"{row['recommended_max_running_requests']} | {row['recommended_extend_batch_size']} | "
            f"{row['throughput_median_items_per_sec']:.1f} | {row['latency_p50_median_ms']:.1f} | "
            f"{row['latency_p99_median_ms']:.1f} | {row['p99_p50_ratio_median']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate(
    *,
    baseline_path: Path,
    matrix_summary_path: Path,
    matrix_gates_path: Path,
) -> dict[str, Any]:
    baseline = load_baseline(baseline_path)
    matrix_cfg = matrix_defaults(baseline)
    workloads_cfg = {str(w["name"]): dict(w) for w in baseline_workloads(baseline)}

    summary = _load_json(matrix_summary_path)
    gates = _load_json(matrix_gates_path)

    gate_passing = {
        int(ips)
        for ips, row in gates.get("by_items_per_step", {}).items()
        if bool(row.get("pass", False))
    }
    if not gate_passing:
        raise RuntimeError("No gate-passing items_per_step candidates found")

    by_workload = summary.get("aggregates", {}).get("by_workload", {})
    if not isinstance(by_workload, dict) or not by_workload:
        raise RuntimeError("matrix_summary missing aggregates.by_workload")

    default_max_running_requests = int(matrix_cfg.get("max_running_requests", 96))
    default_extend_batch_size = int(matrix_cfg.get("multi_item_extend_batch_size", 64))

    out_rows: list[dict[str, Any]] = []
    for workload_name, rows in by_workload.items():
        if not isinstance(rows, list) or not rows:
            continue

        eligible = []
        for row in rows:
            ips = int(row.get("items_per_step", 0))
            if ips not in gate_passing:
                continue
            if float(row.get("failure_rate", 1.0)) > 0.0:
                continue
            eligible.append(row)
        if not eligible:
            # Fall back to highest score row if no strictly eligible rows remain.
            eligible = list(rows)

        best = max(eligible, key=lambda r: float(r.get("score", 0.0)))
        workload_cfg = workloads_cfg.get(str(workload_name), {})
        num_items = int(workload_cfg.get("num_items", 0))

        out_rows.append(
            {
                "workload": str(workload_name),
                "recommended_items_per_step": int(best["items_per_step"]),
                # Keep lane knobs on proven defaults for this phase while selecting per-profile ips.
                "recommended_max_running_requests": default_max_running_requests,
                "recommended_extend_batch_size": default_extend_batch_size,
                "num_items": num_items,
                "throughput_median_items_per_sec": float(best.get("throughput_median_items_per_sec", 0.0)),
                "latency_p50_median_ms": float(best.get("latency_p50_median_ms", 0.0)),
                "latency_p99_median_ms": float(best.get("latency_p99_median_ms", 0.0)),
                "p99_p50_ratio_median": float(best.get("p99_p50_ratio_median", 0.0) or 0.0),
                "failure_rate": float(best.get("failure_rate", 0.0)),
                "score": float(best.get("score", 0.0)),
            }
        )

    out_rows.sort(key=lambda r: r["workload"])

    global_rec = summary.get("recommended", {})
    payload = {
        "name": matrix_summary_path.parent.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_path": str(baseline_path),
        "matrix_summary_path": str(matrix_summary_path),
        "matrix_gates_path": str(matrix_gates_path),
        "gate_passing_items_per_step": sorted(gate_passing),
        "global_recommended_items_per_step": int(global_rec.get("items_per_step") or 0),
        "global_recommended_reason": str(global_rec.get("reason", "")),
        "workloads": out_rows,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="config/prod_scenario_scoring_baseline.yaml")
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--matrix-gates", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    baseline_path = Path(args.baseline).expanduser().resolve()
    matrix_summary_path = Path(args.matrix_summary).expanduser().resolve()
    matrix_gates_path = Path(args.matrix_gates).expanduser().resolve()
    json_out = Path(args.json_out).expanduser().resolve()
    md_out = Path(args.md_out).expanduser().resolve()

    payload = generate(
        baseline_path=baseline_path,
        matrix_summary_path=matrix_summary_path,
        matrix_gates_path=matrix_gates_path,
    )

    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_out.write_text(_render_markdown(payload) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
