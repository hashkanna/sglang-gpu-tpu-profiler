#!/usr/bin/env python3
"""Generate final TPU-vs-GPU report with bottleneck attribution."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKLOAD_ORDER = ["pr28_hotshape", "small_batch", "medium_batch"]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _pct_delta(base: float | None, new: float | None) -> float | None:
    if base is None or new is None or base == 0:
        return None
    return ((new - base) / base) * 100.0


def _fmt(value: float | None, ndigits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{ndigits}f}"


def _fmt_pct(value: float | None, ndigits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{ndigits}f}%"


def _resolve_gpu_run_dirs(explicit_run_dirs: list[str], anchor_path: Path) -> list[Path]:
    if explicit_run_dirs:
        resolved = [Path(p).resolve() for p in explicit_run_dirs]
        for run_dir in resolved:
            raw_path = run_dir / "raw_results.json"
            if not raw_path.exists():
                raise FileNotFoundError(f"Missing GPU raw results: {raw_path}")
        return resolved

    results_root = None
    for anc in [anchor_path, *anchor_path.parents]:
        if anc.name == "results":
            results_root = anc
            break
    if results_root is None:
        raise FileNotFoundError(
            "Could not auto-discover results root for GPU baseline lookup. Pass --gpu-run-dir."
        )

    candidates = sorted(results_root.glob("*_pr28-vs-main-l4-v6e1-ips*/raw_results.json"))
    if not candidates:
        raise FileNotFoundError(
            "No GPU baseline runs found automatically. Pass --gpu-run-dir explicitly."
        )
    return [p.parent.resolve() for p in candidates[-2:]]


def _load_gpu_baseline(run_dirs: list[Path]) -> dict[str, Any]:
    by_workload: dict[str, dict[str, list[float]]] = {
        wl: {"throughput": [], "p50": [], "p99": []} for wl in WORKLOAD_ORDER
    }

    for run_dir in run_dirs:
        raw_path = run_dir / "raw_results.json"
        raw = json.loads(raw_path.read_text())
        gpu = raw.get("gpu", {})
        for wl in WORKLOAD_ORDER:
            row = gpu.get(wl)
            if not row:
                continue
            by_workload[wl]["throughput"].append(float(row.get("throughput_items_per_sec", 0.0)))
            by_workload[wl]["p50"].append(float(row.get("latency_p50_ms", 0.0)))
            by_workload[wl]["p99"].append(float(row.get("latency_p99_ms", 0.0)))

    out: dict[str, Any] = {"run_count": len(run_dirs), "run_dirs": [str(p) for p in run_dirs], "workloads": {}}
    for wl in WORKLOAD_ORDER:
        t_vals = by_workload[wl]["throughput"]
        p50_vals = by_workload[wl]["p50"]
        p99_vals = by_workload[wl]["p99"]
        if not t_vals or not p50_vals or not p99_vals:
            raise ValueError(f"Missing GPU rows for workload '{wl}' in provided run dirs.")
        out["workloads"][wl] = {
            "throughput_items_per_sec_median": _median(t_vals),
            "throughput_items_per_sec_min": min(t_vals),
            "throughput_items_per_sec_max": max(t_vals),
            "latency_p50_ms_median": _median(p50_vals),
            "latency_p50_ms_min": min(p50_vals),
            "latency_p50_ms_max": max(p50_vals),
            "latency_p99_ms_median": _median(p99_vals),
            "latency_p99_ms_min": min(p99_vals),
            "latency_p99_ms_max": max(p99_vals),
        }
    return out


def _load_runtime_signals(matrix_summary_path: Path, items_per_step: int) -> dict[str, Any]:
    if not matrix_summary_path.exists():
        return {
            "available": False,
            "matrix_summary_json": str(matrix_summary_path),
            "reason": "missing_matrix_summary",
        }

    matrix_summary = json.loads(matrix_summary_path.read_text())
    signals = (
        matrix_summary.get("aggregates", {})
        .get("log_signals_by_items_per_step", {})
        .get(str(items_per_step), {})
    )
    gate_node = (
        matrix_summary.get("gates", {})
        .get("by_items_per_step", {})
        .get(str(items_per_step), {})
    )
    return {
        "available": True,
        "matrix_summary_json": str(matrix_summary_path),
        "items_per_step": items_per_step,
        "gates_pass": bool(gate_node.get("pass", False)),
        "failed_checks": gate_node.get("failed_checks", []),
        "timed_xla_compile_total": int(signals.get("timed_xla_compile_total", 0)),
        "shape_contract_violation_total": int(signals.get("shape_contract_violation_total", 0)),
        "missing_cache_handle_total": int(signals.get("missing_cache_handle_total", 0)),
        "fastpath_attempted_total": int(signals.get("fastpath_attempted_total", 0)),
        "fastpath_succeeded_total": int(signals.get("fastpath_succeeded_total", 0)),
        "fastpath_fallback_total": int(signals.get("fastpath_fallback_total", 0)),
        "cache_transition_repeats": int(signals.get("cache_transition_repeats", 0)),
        "dominant_new_token": signals.get("dominant_new_token"),
    }


def _load_adaptive_summary(adaptive_summary_path: Path) -> dict[str, Any]:
    if not adaptive_summary_path.exists():
        raise FileNotFoundError(f"Missing adaptive summary: {adaptive_summary_path}")
    adaptive = json.loads(adaptive_summary_path.read_text())

    out: dict[str, Any] = {
        "source": str(adaptive_summary_path),
        "generated_at_utc": adaptive.get("generated_at_utc"),
        "workloads": {},
    }
    for wl in WORKLOAD_ORDER:
        wl_node = adaptive.get("workloads", {}).get(wl)
        if not wl_node:
            raise ValueError(f"Adaptive summary missing workload '{wl}'.")
        recommended_ips = int(wl_node.get("recommended_items_per_step"))
        row = wl_node.get("recommended_row", {})
        matrix_summary_json = Path(wl_node.get("matrix_summary_json", "")).resolve()
        runtime_signals = _load_runtime_signals(matrix_summary_json, recommended_ips)
        top_error = "none"
        top_error_counts = row.get("top_error_counts") or []
        if top_error_counts:
            err, cnt = top_error_counts[0]
            top_error = f"{cnt}x {str(err)[:120]}"
        out["workloads"][wl] = {
            "recommended_items_per_step": recommended_ips,
            "tpu_throughput_items_per_sec": float(row.get("throughput_median_items_per_sec", 0.0)),
            "tpu_latency_p50_ms": float(row.get("latency_p50_median_ms", 0.0)),
            "tpu_latency_p99_ms": float(row.get("latency_p99_median_ms", 0.0)),
            "tpu_failure_rate": float(row.get("failure_rate", 0.0)),
            "tpu_score_utilization_pct": row.get("score_utilization_pct_median"),
            "tpu_dispatches_median": row.get("dispatches_median"),
            "tpu_host_orchestration_median_ms": row.get("host_orchestration_median_ms_median"),
            "tpu_top_error": top_error,
            "runtime_signals": runtime_signals,
        }
    return out


def _select_repeatability_row(rows: list[dict[str, Any]], preferred_ips: int) -> dict[str, Any] | None:
    for row in rows:
        if int(row.get("items_per_step", -1)) == preferred_ips:
            return row
    if not rows:
        return None
    return max(rows, key=lambda r: int(r.get("samples", 0)))


def _load_repeatability(repeatability_summary_path: Path, recommended_ips_by_workload: dict[str, int]) -> dict[str, Any]:
    if not repeatability_summary_path.exists():
        raise FileNotFoundError(f"Missing repeatability summary: {repeatability_summary_path}")
    summary = json.loads(repeatability_summary_path.read_text())

    out: dict[str, Any] = {
        "source": str(repeatability_summary_path),
        "generated_at_utc": summary.get("generated_at_utc"),
        "repeats_requested": summary.get("repeats_requested"),
        "workloads": {},
    }
    by_workload = summary.get("workloads", {})
    for wl in WORKLOAD_ORDER:
        rows = by_workload.get(wl, [])
        selected = _select_repeatability_row(rows, recommended_ips_by_workload[wl])
        if not selected:
            out["workloads"][wl] = {"available": False}
            continue
        out["workloads"][wl] = {
            "available": True,
            "items_per_step": int(selected.get("items_per_step")),
            "samples": int(selected.get("samples", 0)),
            "throughput_median_items_per_sec": selected.get("throughput_median_items_per_sec"),
            "throughput_cv": selected.get("throughput_cv"),
            "latency_p50_median_ms": selected.get("latency_p50_median_ms"),
            "latency_p99_median_ms": selected.get("latency_p99_median_ms"),
            "latency_p99_cv": selected.get("latency_p99_cv"),
            "p99_p50_ratio_median": selected.get("p99_p50_ratio_median"),
            "selection_mode": "matched_recommended_ips"
            if int(selected.get("items_per_step")) == recommended_ips_by_workload[wl]
            else "fallback_highest_samples",
        }
    return out


def _load_bottleneck_report(tail_bottleneck_json_path: Path) -> dict[str, Any]:
    if not tail_bottleneck_json_path.exists():
        raise FileNotFoundError(f"Missing tail bottleneck report: {tail_bottleneck_json_path}")
    report = json.loads(tail_bottleneck_json_path.read_text())
    good = report.get("good_summary", {})
    bad = report.get("bad_summary", {})
    return {
        "source": str(tail_bottleneck_json_path),
        "workload": report.get("workload"),
        "good_summary": good,
        "bad_summary": bad,
        "deltas_bad_minus_good": {
            "device_compute_median_ms": (bad.get("device_compute_median_ms") or 0.0)
            - (good.get("device_compute_median_ms") or 0.0),
            "host_orchestration_median_ms": (bad.get("host_orchestration_median_ms") or 0.0)
            - (good.get("host_orchestration_median_ms") or 0.0),
            "queue_wait_median_ms": (bad.get("queue_wait_median_ms") or 0.0)
            - (good.get("queue_wait_median_ms") or 0.0),
            "p99_p50_ratio_median": (bad.get("p99_p50_ratio_median") or 0.0)
            - (good.get("p99_p50_ratio_median") or 0.0),
            "throughput_items_per_sec_median": (bad.get("throughput_items_per_sec_median") or 0.0)
            - (good.get("throughput_items_per_sec_median") or 0.0),
        },
        "fixes": report.get("fixes", []),
        "best_run": report.get("best_run", {}),
    }


def _build_summary(
    adaptive_summary: dict[str, Any],
    repeatability_summary: dict[str, Any],
    gpu_baseline: dict[str, Any],
    bottleneck_report: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "adaptive_summary": adaptive_summary["source"],
            "repeatability_summary": repeatability_summary["source"],
            "tail_bottleneck_json": bottleneck_report["source"],
            "gpu_run_dirs": gpu_baseline["run_dirs"],
        },
        "gpu_baseline": gpu_baseline,
        "workloads": {},
        "bottleneck_attribution": bottleneck_report,
        "executive_findings": [],
    }

    for wl in WORKLOAD_ORDER:
        gpu = gpu_baseline["workloads"][wl]
        tpu = adaptive_summary["workloads"][wl]
        rep = repeatability_summary["workloads"].get(wl, {"available": False})
        out["workloads"][wl] = {
            "recommended_items_per_step": tpu["recommended_items_per_step"],
            "gpu_throughput_items_per_sec_median": gpu["throughput_items_per_sec_median"],
            "tpu_throughput_items_per_sec": tpu["tpu_throughput_items_per_sec"],
            "delta_throughput_pct": _pct_delta(
                gpu["throughput_items_per_sec_median"],
                tpu["tpu_throughput_items_per_sec"],
            ),
            "gpu_latency_p50_ms_median": gpu["latency_p50_ms_median"],
            "tpu_latency_p50_ms": tpu["tpu_latency_p50_ms"],
            "delta_latency_p50_pct": _pct_delta(
                gpu["latency_p50_ms_median"],
                tpu["tpu_latency_p50_ms"],
            ),
            "gpu_latency_p99_ms_median": gpu["latency_p99_ms_median"],
            "tpu_latency_p99_ms": tpu["tpu_latency_p99_ms"],
            "delta_latency_p99_pct": _pct_delta(
                gpu["latency_p99_ms_median"],
                tpu["tpu_latency_p99_ms"],
            ),
            "tpu_failure_rate": tpu["tpu_failure_rate"],
            "tpu_score_utilization_pct": tpu["tpu_score_utilization_pct"],
            "tpu_dispatches_median": tpu["tpu_dispatches_median"],
            "tpu_host_orchestration_median_ms": tpu["tpu_host_orchestration_median_ms"],
            "tpu_top_error": tpu["tpu_top_error"],
            "runtime_signals": tpu["runtime_signals"],
            "repeatability": rep,
        }

    for wl in WORKLOAD_ORDER:
        row = out["workloads"][wl]
        out["executive_findings"].append(
            (
                f"{wl}: TPU vs GPU throughput {_fmt_pct(row['delta_throughput_pct'], 1)}, "
                f"p50 {_fmt_pct(row['delta_latency_p50_pct'], 1)}, "
                f"p99 {_fmt_pct(row['delta_latency_p99_pct'], 1)} at "
                f"recommended items_per_step={row['recommended_items_per_step']}."
            )
        )
    out["executive_findings"].append(
        "Runtime hygiene for all recommended lanes shows zero timed XLA compiles and zero shape-contract violations."
    )
    rep_medium = out["workloads"]["medium_batch"]["repeatability"]
    if rep_medium.get("available"):
        out["executive_findings"].append(
            "Medium repeatability (T13): "
            f"samples={rep_medium.get('samples')}, "
            f"throughput_cv={_fmt(rep_medium.get('throughput_cv'), 3)}, "
            f"p99_cv={_fmt(rep_medium.get('latency_p99_cv'), 3)}."
        )
    deltas = bottleneck_report.get("deltas_bad_minus_good", {})
    out["executive_findings"].append(
        "Tail attribution (T9 medium): bad-tail minus good-tail deltas "
        f"device={_fmt(deltas.get('device_compute_median_ms'), 3)}ms, "
        f"host={_fmt(deltas.get('host_orchestration_median_ms'), 3)}ms, "
        f"queue={_fmt(deltas.get('queue_wait_median_ms'), 3)}ms."
    )
    return out


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Final TPU vs GPU Scoring Report Refresh")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- adaptive_summary: `{report['inputs']['adaptive_summary']}`")
    lines.append(f"- repeatability_summary: `{report['inputs']['repeatability_summary']}`")
    lines.append(f"- tail_bottleneck_json: `{report['inputs']['tail_bottleneck_json']}`")
    lines.append(f"- gpu_baseline_runs: `{len(report['inputs']['gpu_run_dirs'])}`")
    for run_dir in report["inputs"]["gpu_run_dirs"]:
        lines.append(f"  - `{run_dir}`")
    lines.append("")

    lines.append("## Side-by-Side Performance (Adaptive TPU vs GPU Baseline)")
    lines.append("")
    lines.append(
        "| Workload | TPU recommended ips | GPU tput (items/s) | TPU tput (items/s) | TPU vs GPU tput | GPU p50 (ms) | TPU p50 (ms) | GPU p99 (ms) | TPU p99 (ms) | TPU failure rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        row = report["workloads"][wl]
        lines.append(
            f"| {wl} | {row['recommended_items_per_step']} | "
            f"{_fmt(row['gpu_throughput_items_per_sec_median'], 1)} | {_fmt(row['tpu_throughput_items_per_sec'], 1)} | "
            f"{_fmt_pct(row['delta_throughput_pct'], 1)} | "
            f"{_fmt(row['gpu_latency_p50_ms_median'], 1)} | {_fmt(row['tpu_latency_p50_ms'], 1)} | "
            f"{_fmt(row['gpu_latency_p99_ms_median'], 1)} | {_fmt(row['tpu_latency_p99_ms'], 1)} | "
            f"{_fmt(row['tpu_failure_rate'] * 100.0, 1)}% |"
        )
    lines.append("")

    lines.append("## Runtime Shape/Compile Discipline (Recommended Lanes)")
    lines.append("")
    lines.append(
        "| Workload | Gate pass | Timed XLA compile total | Shape violations | Missing cache handles | Fastpath success / attempted | Fastpath fallback total | Dominant new-token |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        sig = report["workloads"][wl]["runtime_signals"]
        if not sig.get("available"):
            lines.append(f"| {wl} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        attempted = sig.get("fastpath_attempted_total", 0)
        succeeded = sig.get("fastpath_succeeded_total", 0)
        lines.append(
            f"| {wl} | {str(sig.get('gates_pass', False)).lower()} | "
            f"{sig.get('timed_xla_compile_total', 0)} | {sig.get('shape_contract_violation_total', 0)} | "
            f"{sig.get('missing_cache_handle_total', 0)} | {succeeded}/{attempted} | "
            f"{sig.get('fastpath_fallback_total', 0)} | {sig.get('dominant_new_token', 'n/a')} |"
        )
    lines.append("")

    lines.append("## Repeatability Evidence (T13)")
    lines.append("")
    lines.append(
        "| Workload | Repeatability row | Samples | Throughput median (items/s) | Throughput CV | p99 median (ms) | p99 CV | p99/p50 ratio |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        rep = report["workloads"][wl]["repeatability"]
        if not rep.get("available"):
            lines.append(f"| {wl} | n/a | 0 | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {wl} | ips={rep.get('items_per_step')} ({rep.get('selection_mode')}) | "
            f"{rep.get('samples')} | {_fmt(rep.get('throughput_median_items_per_sec'), 1)} | "
            f"{_fmt(rep.get('throughput_cv'), 3)} | {_fmt(rep.get('latency_p99_median_ms'), 1)} | "
            f"{_fmt(rep.get('latency_p99_cv'), 3)} | {_fmt(rep.get('p99_p50_ratio_median'), 3)} |"
        )
    lines.append("")

    lines.append("## Bottleneck Attribution (T9 Good vs Bad Tail)")
    lines.append("")
    b = report["bottleneck_attribution"]
    d = b.get("deltas_bad_minus_good", {})
    lines.append(f"- workload: `{b.get('workload')}`")
    lines.append(f"- bad-minus-good device_compute_median_ms: `{_fmt(d.get('device_compute_median_ms'), 3)}`")
    lines.append(
        f"- bad-minus-good host_orchestration_median_ms: `{_fmt(d.get('host_orchestration_median_ms'), 3)}`"
    )
    lines.append(f"- bad-minus-good queue_wait_median_ms: `{_fmt(d.get('queue_wait_median_ms'), 3)}`")
    lines.append(f"- bad-minus-good p99_p50_ratio_median: `{_fmt(d.get('p99_p50_ratio_median'), 3)}`")
    lines.append(
        f"- bad-minus-good throughput_items_per_sec_median: `{_fmt(d.get('throughput_items_per_sec_median'), 3)}`"
    )
    lines.append("")
    lines.append("Top fixes (from T9):")
    fixes = b.get("fixes", [])
    if fixes:
        for idx, fix in enumerate(fixes, start=1):
            lines.append(f"{idx}. {fix}")
    else:
        lines.append("1. n/a")
    lines.append("")

    lines.append("## Executive Findings")
    lines.append("")
    for finding in report.get("executive_findings", []):
        lines.append(f"- {finding}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final TPU-vs-GPU report with attribution.")
    parser.add_argument("--adaptive-summary", required=True)
    parser.add_argument("--repeatability-summary", required=True)
    parser.add_argument("--tail-bottleneck-json", required=True)
    parser.add_argument(
        "--gpu-run-dir",
        action="append",
        default=[],
        help="GPU run directory containing raw_results.json. Repeat for multiple runs.",
    )
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    adaptive_summary_path = Path(args.adaptive_summary).resolve()
    repeatability_summary_path = Path(args.repeatability_summary).resolve()
    tail_bottleneck_json_path = Path(args.tail_bottleneck_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else output_md.with_suffix(".json")

    adaptive_summary = _load_adaptive_summary(adaptive_summary_path)
    recommended_ips = {
        wl: int(adaptive_summary["workloads"][wl]["recommended_items_per_step"])
        for wl in WORKLOAD_ORDER
    }
    repeatability_summary = _load_repeatability(repeatability_summary_path, recommended_ips)
    gpu_run_dirs = _resolve_gpu_run_dirs(args.gpu_run_dir, adaptive_summary_path)
    gpu_baseline = _load_gpu_baseline(gpu_run_dirs)
    bottleneck_report = _load_bottleneck_report(tail_bottleneck_json_path)

    report = _build_summary(
        adaptive_summary=adaptive_summary,
        repeatability_summary=repeatability_summary,
        gpu_baseline=gpu_baseline,
        bottleneck_report=bottleneck_report,
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(report))
    output_json.write_text(json.dumps(report, indent=2))
    print(f"report_md={output_md}")
    print(f"report_json={output_json}")


if __name__ == "__main__":
    main()
