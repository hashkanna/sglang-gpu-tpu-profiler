#!/usr/bin/env python3
"""Summarize low/mean uplift candidates against a baseline lane."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _metric_row(summary: dict[str, Any], workload: str, ips: int) -> dict[str, Any]:
    by_wl = summary.get("aggregates", {}).get("by_workload", {})
    rows = by_wl.get(workload, [])
    if not isinstance(rows, list):
        raise ValueError(f"Missing by_workload rows for {workload}")
    for row in rows:
        if int(row.get("items_per_step", -1)) == int(ips):
            return row
    raise ValueError(f"items_per_step={ips} not found for workload={workload}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate low/mean uplift recommendation report.")
    parser.add_argument("--matrix-summary", required=True, type=Path)
    parser.add_argument("--baseline-ips", type=int, default=128)
    parser.add_argument("--candidate-ips", required=True, help="Comma-separated list, e.g. 64,96,160")
    parser.add_argument(
        "--low-mean-workloads",
        default="track_low,track_mean,home_low,home_mean",
    )
    parser.add_argument(
        "--high-workloads",
        default="track_high,home_high",
    )
    parser.add_argument("--variant-summary", default=None, type=Path)
    parser.add_argument("--variant-label", default="variant96_mrr128")
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    summary = _load_json(args.matrix_summary)
    low_mean_workloads = _parse_csv(args.low_mean_workloads)
    high_workloads = _parse_csv(args.high_workloads)
    candidates = _parse_int_csv(args.candidate_ips)

    decisions: list[dict[str, Any]] = []
    for ips in candidates:
        low_mean_rows: list[dict[str, Any]] = []
        high_rows: list[dict[str, Any]] = []
        low_mean_score = 0
        throughput_gain_sum = 0.0
        p99_gain_sum = 0.0

        for wl in low_mean_workloads:
            base = _metric_row(summary, wl, args.baseline_ips)
            cand = _metric_row(summary, wl, ips)
            d_tput = float(cand["throughput_median_items_per_sec"]) - float(
                base["throughput_median_items_per_sec"]
            )
            d_p99 = float(cand["latency_p99_median_ms"]) - float(base["latency_p99_median_ms"])
            improved = (d_tput > 0.0) or (d_p99 < 0.0)
            if improved:
                low_mean_score += 1
            throughput_gain_sum += d_tput
            p99_gain_sum += -d_p99
            low_mean_rows.append(
                {
                    "workload": wl,
                    "baseline": {
                        "throughput_items_per_sec": float(base["throughput_median_items_per_sec"]),
                        "latency_p99_ms": float(base["latency_p99_median_ms"]),
                    },
                    "candidate": {
                        "throughput_items_per_sec": float(cand["throughput_median_items_per_sec"]),
                        "latency_p99_ms": float(cand["latency_p99_median_ms"]),
                    },
                    "delta_candidate_minus_baseline": {
                        "throughput_items_per_sec": d_tput,
                        "latency_p99_ms": d_p99,
                    },
                    "improved": improved,
                }
            )

        high_no_regress = True
        for wl in high_workloads:
            base = _metric_row(summary, wl, args.baseline_ips)
            cand = _metric_row(summary, wl, ips)
            d_tput = float(cand["throughput_median_items_per_sec"]) - float(
                base["throughput_median_items_per_sec"]
            )
            d_p99 = float(cand["latency_p99_median_ms"]) - float(base["latency_p99_median_ms"])
            no_reg = (d_tput >= 0.0) and (d_p99 <= 0.0)
            high_no_regress = high_no_regress and no_reg
            high_rows.append(
                {
                    "workload": wl,
                    "delta_candidate_minus_baseline": {
                        "throughput_items_per_sec": d_tput,
                        "latency_p99_ms": d_p99,
                    },
                    "no_regression": no_reg,
                }
            )

        decisions.append(
            {
                "items_per_step": ips,
                "high_no_regression": high_no_regress,
                "low_mean_improved_workloads": low_mean_score,
                "low_mean_workload_count": len(low_mean_workloads),
                "low_mean_throughput_delta_sum": throughput_gain_sum,
                "low_mean_p99_improvement_sum_ms": p99_gain_sum,
                "low_mean_details": low_mean_rows,
                "high_details": high_rows,
            }
        )

    eligible = [d for d in decisions if d["high_no_regression"]]
    if eligible:
        recommended = sorted(
            eligible,
            key=lambda d: (
                d["low_mean_improved_workloads"],
                d["low_mean_throughput_delta_sum"],
                d["low_mean_p99_improvement_sum_ms"],
            ),
            reverse=True,
        )[0]
    else:
        recommended = sorted(
            decisions,
            key=lambda d: (
                d["low_mean_improved_workloads"],
                d["low_mean_throughput_delta_sum"],
                d["low_mean_p99_improvement_sum_ms"],
            ),
            reverse=True,
        )[0]

    variant_section: dict[str, Any] | None = None
    if args.variant_summary is not None:
        variant = _load_json(args.variant_summary)
        base_96 = {wl: _metric_row(summary, wl, 96) for wl in low_mean_workloads + high_workloads}
        var_96 = {wl: _metric_row(variant, wl, 96) for wl in low_mean_workloads + high_workloads}
        comp = {}
        for wl in low_mean_workloads + high_workloads:
            d_tput = float(var_96[wl]["throughput_median_items_per_sec"]) - float(
                base_96[wl]["throughput_median_items_per_sec"]
            )
            d_p99 = float(var_96[wl]["latency_p99_median_ms"]) - float(
                base_96[wl]["latency_p99_median_ms"]
            )
            comp[wl] = {
                "delta_variant_minus_main96": {
                    "throughput_items_per_sec": d_tput,
                    "latency_p99_ms": d_p99,
                }
            }
        variant_section = {
            "label": args.variant_label,
            "summary_path": str(args.variant_summary),
            "comparison_vs_main_sweep_96": comp,
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "matrix_summary": str(args.matrix_summary),
            "baseline_items_per_step": int(args.baseline_ips),
            "candidate_items_per_step": candidates,
            "low_mean_workloads": low_mean_workloads,
            "high_workloads": high_workloads,
        },
        "decisions": decisions,
        "recommended": {
            "items_per_step": int(recommended["items_per_step"]),
            "high_no_regression": bool(recommended["high_no_regression"]),
            "low_mean_improved_workloads": int(recommended["low_mean_improved_workloads"]),
            "low_mean_workload_count": int(recommended["low_mean_workload_count"]),
            "low_mean_throughput_delta_sum": float(recommended["low_mean_throughput_delta_sum"]),
            "low_mean_p99_improvement_sum_ms": float(recommended["low_mean_p99_improvement_sum_ms"]),
            "rationale": (
                "Best low/mean uplift score under high-workload non-regression constraint."
                if recommended["high_no_regression"]
                else "No candidate met strict high-workload non-regression; selected best low/mean score."
            ),
        },
        "variant_check": variant_section,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    lines: list[str] = []
    lines.append("# Low/Mean Uplift Sweep Report")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- recommended_items_per_step: `{report['recommended']['items_per_step']}`")
    lines.append(f"- high_no_regression: `{str(report['recommended']['high_no_regression']).lower()}`")
    lines.append(
        f"- low_mean_improved_workloads: `{report['recommended']['low_mean_improved_workloads']}/{report['recommended']['low_mean_workload_count']}`"
    )
    lines.append(
        f"- low_mean_throughput_delta_sum: `{report['recommended']['low_mean_throughput_delta_sum']:+.3f} items/s`"
    )
    lines.append(
        f"- low_mean_p99_improvement_sum_ms: `{report['recommended']['low_mean_p99_improvement_sum_ms']:+.3f} ms`"
    )
    lines.append(f"- rationale: {report['recommended']['rationale']}")
    lines.append("")
    lines.append("## Candidate Scores")
    lines.append("")
    lines.append(
        "| items_per_step | high_no_regression | low_mean_improved | low_mean_tput_delta_sum | low_mean_p99_improvement_sum_ms |"
    )
    lines.append("|---:|---|---:|---:|---:|")
    for row in sorted(report["decisions"], key=lambda x: int(x["items_per_step"])):
        lines.append(
            f"| {row['items_per_step']} | {str(row['high_no_regression']).lower()} | "
            f"{row['low_mean_improved_workloads']}/{row['low_mean_workload_count']} | "
            f"{row['low_mean_throughput_delta_sum']:+.3f} | {row['low_mean_p99_improvement_sum_ms']:+.3f} |"
        )

    if report.get("variant_check"):
        lines.append("")
        lines.append("## Variant Check")
        lines.append("")
        vc = report["variant_check"]
        lines.append(f"- label: `{vc['label']}`")
        lines.append(f"- summary: `{vc['summary_path']}`")
        lines.append("- Values are `variant96_mrr128 - main_sweep_ips96`.")
        lines.append("")
        lines.append("| Workload | delta_throughput_items_per_sec | delta_p99_ms |")
        lines.append("|---|---:|---:|")
        for wl, detail in vc["comparison_vs_main_sweep_96"].items():
            d = detail["delta_variant_minus_main96"]
            lines.append(f"| {wl} | {d['throughput_items_per_sec']:+.3f} | {d['latency_p99_ms']:+.3f} |")

    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- matrix_summary: `{args.matrix_summary}`")
    lines.append(f"- baseline_items_per_step: `{args.baseline_ips}`")
    lines.append(f"- candidate_items_per_step: `{','.join(str(x) for x in candidates)}`")
    args.md_out.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
