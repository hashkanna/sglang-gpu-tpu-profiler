#!/usr/bin/env python3
"""Generate client co-location A/B report for TPU scoring benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate client co-location A/B report.")
    parser.add_argument("--local-results", required=True, type=Path)
    parser.add_argument("--colocated-results", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    local_data = _load_json(args.local_results)
    colocated_data = _load_json(args.colocated_results)

    local_w = local_data.get("workloads", {})
    colocated_w = colocated_data.get("workloads", {})
    common_workloads = sorted(set(local_w.keys()) & set(colocated_w.keys()))
    if not common_workloads:
        raise ValueError("No overlapping workloads between local and colocated results.")

    per_workload: list[dict[str, Any]] = []
    local_tputs: list[float] = []
    colocated_tputs: list[float] = []
    local_p50s: list[float] = []
    colocated_p50s: list[float] = []
    local_p99s: list[float] = []
    colocated_p99s: list[float] = []

    for wl in common_workloads:
        l = local_w[wl]
        c = colocated_w[wl]
        l_tput = float(l.get("throughput_items_per_sec") or 0.0)
        c_tput = float(c.get("throughput_items_per_sec") or 0.0)
        l_p50 = float(l.get("latency_p50_ms") or 0.0)
        c_p50 = float(c.get("latency_p50_ms") or 0.0)
        l_p99 = float(l.get("latency_p99_ms") or 0.0)
        c_p99 = float(c.get("latency_p99_ms") or 0.0)

        local_tputs.append(l_tput)
        colocated_tputs.append(c_tput)
        local_p50s.append(l_p50)
        colocated_p50s.append(c_p50)
        local_p99s.append(l_p99)
        colocated_p99s.append(c_p99)

        tput_ratio = (c_tput / l_tput) if l_tput > 0 else None
        p50_ratio = (l_p50 / c_p50) if c_p50 > 0 else None
        p99_ratio = (l_p99 / c_p99) if c_p99 > 0 else None

        per_workload.append(
            {
                "workload": wl,
                "shape": {
                    "query_tokens": int(c.get("query_tokens", 0)),
                    "num_items": int(c.get("num_items", 0)),
                    "item_tokens": int(c.get("item_tokens", 0)),
                },
                "local_client": {
                    "throughput_items_per_sec": l_tput,
                    "latency_p50_ms": l_p50,
                    "latency_p99_ms": l_p99,
                    "num_failures": int(l.get("num_failures", 0)),
                },
                "colocated_client": {
                    "throughput_items_per_sec": c_tput,
                    "latency_p50_ms": c_p50,
                    "latency_p99_ms": c_p99,
                    "num_failures": int(c.get("num_failures", 0)),
                },
                "delta_colocated_minus_local": {
                    "throughput_items_per_sec": c_tput - l_tput,
                    "latency_p50_ms": c_p50 - l_p50,
                    "latency_p99_ms": c_p99 - l_p99,
                },
                "ratio": {
                    "throughput_x": tput_ratio,
                    "p50_reduction_x": p50_ratio,
                    "p99_reduction_x": p99_ratio,
                },
            }
        )

    summary = {
        "mean_local_throughput_items_per_sec": _mean(local_tputs),
        "mean_colocated_throughput_items_per_sec": _mean(colocated_tputs),
        "mean_local_latency_p50_ms": _mean(local_p50s),
        "mean_colocated_latency_p50_ms": _mean(colocated_p50s),
        "mean_local_latency_p99_ms": _mean(local_p99s),
        "mean_colocated_latency_p99_ms": _mean(colocated_p99s),
    }
    summary["throughput_multiplier_colocated_vs_local"] = (
        summary["mean_colocated_throughput_items_per_sec"] / summary["mean_local_throughput_items_per_sec"]
        if summary["mean_local_throughput_items_per_sec"] > 0
        else None
    )
    summary["p50_reduction_multiplier_local_vs_colocated"] = (
        summary["mean_local_latency_p50_ms"] / summary["mean_colocated_latency_p50_ms"]
        if summary["mean_colocated_latency_p50_ms"] > 0
        else None
    )
    summary["p99_reduction_multiplier_local_vs_colocated"] = (
        summary["mean_local_latency_p99_ms"] / summary["mean_colocated_latency_p99_ms"]
        if summary["mean_colocated_latency_p99_ms"] > 0
        else None
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "local_results": str(args.local_results),
            "colocated_results": str(args.colocated_results),
        },
        "benchmark_contract": {
            "model": colocated_data.get("model"),
            "endpoint": colocated_data.get("url"),
            "warmup_requests": int(colocated_data.get("benchmark", {}).get("warmup_requests", 0)),
            "timed_requests": int(colocated_data.get("benchmark", {}).get("timed_requests", 0)),
            "concurrency": int(colocated_data.get("benchmark", {}).get("concurrency", 0)),
            "timeout_sec": int(colocated_data.get("benchmark", {}).get("timeout_sec", 0)),
        },
        "summary": summary,
        "per_workload": per_workload,
        "decision": {
            "policy": "USE_COLOCATED_CLIENT_FOR_TPU_BENCHMARKS",
            "rationale": (
                "Client-location overhead materially inflates local-workstation latency; "
                "co-located client better reflects serving-path performance."
            ),
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    lines: list[str] = []
    lines.append("# TPU Client Co-location A/B Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Mean throughput: local={summary['mean_local_throughput_items_per_sec']:.3f}, "
        f"co-located={summary['mean_colocated_throughput_items_per_sec']:.3f}, "
        f"multiplier={summary['throughput_multiplier_colocated_vs_local']:.3f}x"
    )
    lines.append(
        f"- Mean p50 latency: local={summary['mean_local_latency_p50_ms']:.3f} ms, "
        f"co-located={summary['mean_colocated_latency_p50_ms']:.3f} ms, "
        f"reduction={summary['p50_reduction_multiplier_local_vs_colocated']:.3f}x"
    )
    lines.append(
        f"- Mean p99 latency: local={summary['mean_local_latency_p99_ms']:.3f} ms, "
        f"co-located={summary['mean_colocated_latency_p99_ms']:.3f} ms, "
        f"reduction={summary['p99_reduction_multiplier_local_vs_colocated']:.3f}x"
    )
    lines.append("")
    lines.append("## Per-Workload Comparison")
    lines.append("")
    lines.append(
        "| Workload | Shape (q,n,it) | Local tput | Co-located tput | tput x | Local p50 | Co-located p50 | p50 reduction x | Local p99 | Co-located p99 | p99 reduction x |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in per_workload:
        shape = row["shape"]
        local = row["local_client"]
        colocated = row["colocated_client"]
        ratio = row["ratio"]
        lines.append(
            f"| {row['workload']} | ({shape['query_tokens']},{shape['num_items']},{shape['item_tokens']}) | "
            f"{local['throughput_items_per_sec']:.3f} | {colocated['throughput_items_per_sec']:.3f} | "
            f"{(ratio['throughput_x'] or 0.0):.3f} | {local['latency_p50_ms']:.3f} | {colocated['latency_p50_ms']:.3f} | "
            f"{(ratio['p50_reduction_x'] or 0.0):.3f} | {local['latency_p99_ms']:.3f} | {colocated['latency_p99_ms']:.3f} | "
            f"{(ratio['p99_reduction_x'] or 0.0):.3f} |"
        )

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- Policy: `USE_COLOCATED_CLIENT_FOR_TPU_BENCHMARKS`")
    lines.append(
        "- Reason: local workstation client path introduces substantial extra latency and lowers measured throughput."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Local-client results: `{args.local_results}`")
    lines.append(f"- Co-located-client results: `{args.colocated_results}`")

    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
