#!/usr/bin/env python3
"""Generate final production-shaped TPU vs GPU report with actual numbers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final production-shaped TPU vs GPU report.")
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--tpu-summary", required=True, type=Path)
    parser.add_argument("--tpu-gates", required=True, type=Path)
    parser.add_argument("--gpu-results", required=True, type=Path)
    parser.add_argument("--items-per-step", type=int, default=128)
    parser.add_argument(
        "--tpu-client-location-label",
        default="tpu_vm_local",
        help="Label describing TPU benchmark client location.",
    )
    parser.add_argument(
        "--gpu-client-location-label",
        default="gpu_host_local",
        help="Label describing GPU benchmark client location.",
    )
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    baseline = yaml.safe_load(args.baseline_config.read_text())
    tpu_summary = _load_json(args.tpu_summary)
    tpu_gates = _load_json(args.tpu_gates)
    gpu_results = _load_json(args.gpu_results)

    ips = int(args.items_per_step)
    tpu_by_wl_raw = tpu_summary.get("aggregates", {}).get("by_workload", {})
    if not isinstance(tpu_by_wl_raw, dict):
        raise ValueError("TPU summary missing aggregates.by_workload")
    gpu_wl_raw = gpu_results.get("workloads", {})
    if not isinstance(gpu_wl_raw, dict):
        raise ValueError("GPU results missing workloads")

    workloads = sorted(set(tpu_by_wl_raw.keys()).intersection(gpu_wl_raw.keys()))
    rows: list[dict[str, Any]] = []
    for wl in workloads:
        tpu_rows = tpu_by_wl_raw[wl]
        if not isinstance(tpu_rows, list):
            continue
        tpu_row = None
        for r in tpu_rows:
            if isinstance(r, dict) and int(r.get("items_per_step", 0)) == ips:
                tpu_row = r
                break
        if tpu_row is None:
            continue
        gpu_row = gpu_wl_raw[wl]
        tpu_tput = float(tpu_row["throughput_mean_items_per_sec"])
        tpu_p50 = float(tpu_row["latency_p50_median_ms"])
        tpu_p99 = float(tpu_row["latency_p99_median_ms"])
        gpu_tput = float(gpu_row["throughput_items_per_sec_mean"])
        gpu_p50 = float(gpu_row["latency_p50_ms_mean"])
        gpu_p99 = float(gpu_row["latency_p99_ms_mean"])
        rows.append(
            {
                "workload": wl,
                "shape": {
                    "query_tokens": int(gpu_row["query_tokens"]),
                    "num_items": int(gpu_row["num_items"]),
                    "item_tokens": int(gpu_row["item_tokens"]),
                },
                "tpu": {
                    "throughput_items_per_sec": tpu_tput,
                    "latency_p50_ms": tpu_p50,
                    "latency_p99_ms": tpu_p99,
                    "num_failures_total": int(tpu_row.get("num_failures_total", 0)),
                },
                "gpu": {
                    "throughput_items_per_sec": gpu_tput,
                    "latency_p50_ms": gpu_p50,
                    "latency_p99_ms": gpu_p99,
                    "num_failures_total": int(gpu_row.get("num_failures_total", 0)),
                },
                "delta_tpu_minus_gpu": {
                    "throughput_items_per_sec": tpu_tput - gpu_tput,
                    "latency_p50_ms": tpu_p50 - gpu_p50,
                    "latency_p99_ms": tpu_p99 - gpu_p99,
                },
            }
        )

    tpu_tputs = [r["tpu"]["throughput_items_per_sec"] for r in rows]
    gpu_tputs = [r["gpu"]["throughput_items_per_sec"] for r in rows]
    tpu_p50s = [r["tpu"]["latency_p50_ms"] for r in rows]
    gpu_p50s = [r["gpu"]["latency_p50_ms"] for r in rows]
    tpu_p99s = [r["tpu"]["latency_p99_ms"] for r in rows]
    gpu_p99s = [r["gpu"]["latency_p99_ms"] for r in rows]

    tpu_log = tpu_summary.get("aggregates", {}).get("log_signals_by_items_per_step", {}).get(str(ips), {})
    if not isinstance(tpu_log, dict):
        tpu_log = {}
    tpu_gate = tpu_gates.get("by_items_per_step", {}).get(str(ips), {})
    if not isinstance(tpu_gate, dict):
        tpu_gate = {}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "model": baseline.get("experiment", {}).get("model"),
            "workload_count": len(workloads),
            "benchmark": {
                "warmup_requests": int(gpu_results.get("benchmark", {}).get("warmup_requests", 0)),
                "timed_requests": int(gpu_results.get("benchmark", {}).get("timed_requests", 0)),
                "concurrency": int(gpu_results.get("benchmark", {}).get("concurrency", 0)),
                "timeout_sec": int(gpu_results.get("benchmark", {}).get("timeout_sec", 0)),
            },
        },
        "sources": {
            "baseline_config": str(args.baseline_config),
            "tpu_summary": str(args.tpu_summary),
            "tpu_gates": str(args.tpu_gates),
            "gpu_results": str(args.gpu_results),
        },
        "tpu_runtime": {
            "items_per_step": ips,
            "benchmark_client_location": str(args.tpu_client_location_label),
            "gate_pass": bool(tpu_gate.get("pass", False)),
            "timed_xla_compile_total": int(tpu_log.get("timed_xla_compile_total", 0)),
            "shape_contract_violation_total": int(tpu_log.get("shape_contract_violation_total", 0)),
            "missing_cache_handle_total": int(tpu_log.get("missing_cache_handle_total", 0)),
            "cache_transition_repeats": int(tpu_log.get("cache_transition_repeats", 0)),
            "server_env": baseline.get("tpu_server", {}).get("env", {}),
            "server_static_args": baseline.get("tpu_server", {}).get("static_args", []),
        },
        "gpu_runtime": {
            "hardware": gpu_results.get("hardware"),
            "benchmark_client_location": str(args.gpu_client_location_label),
            "server": gpu_results.get("server", {}),
        },
        "per_workload": rows,
        "aggregate_means": {
            "tpu": {
                "throughput_items_per_sec": _mean(tpu_tputs),
                "latency_p50_ms": _mean(tpu_p50s),
                "latency_p99_ms": _mean(tpu_p99s),
            },
            "gpu": {
                "throughput_items_per_sec": _mean(gpu_tputs),
                "latency_p50_ms": _mean(gpu_p50s),
                "latency_p99_ms": _mean(gpu_p99s),
            },
            "delta_tpu_minus_gpu": {
                "throughput_items_per_sec": _mean(tpu_tputs) - _mean(gpu_tputs),
                "latency_p50_ms": _mean(tpu_p50s) - _mean(gpu_p50s),
                "latency_p99_ms": _mean(tpu_p99s) - _mean(gpu_p99s),
            },
        },
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))

    md: list[str] = []
    md.append("# Final Production-Shaped TPU vs GPU Report")
    md.append("")
    md.append("## Contract")
    md.append("")
    md.append(f"- Model: `{report['contract']['model']}`")
    b = report["contract"]["benchmark"]
    md.append(
        f"- Benchmark: warmup={b['warmup_requests']}, timed={b['timed_requests']}, "
        f"concurrency={b['concurrency']}, timeout_sec={b['timeout_sec']}"
    )
    md.append(f"- Workloads: {report['contract']['workload_count']}")
    md.append("")
    md.append("## TPU Runtime Hygiene")
    md.append("")
    tpu_rt = report["tpu_runtime"]
    md.append(f"- benchmark_client_location: {tpu_rt['benchmark_client_location']}")
    md.append(f"- items_per_step: {tpu_rt['items_per_step']}")
    md.append(f"- gate_pass: {str(tpu_rt['gate_pass']).lower()}")
    md.append(f"- timed_xla_compile_total: {tpu_rt['timed_xla_compile_total']}")
    md.append(f"- shape_contract_violation_total: {tpu_rt['shape_contract_violation_total']}")
    md.append(f"- missing_cache_handle_total: {tpu_rt['missing_cache_handle_total']}")
    md.append("")
    md.append("## Per-Workload Actual Numbers")
    md.append("")
    md.append(
        "| Workload | Shape (q,n,it) | TPU Throughput | GPU Throughput | TPU p50 | GPU p50 | TPU p99 | GPU p99 | TPU Failures | GPU Failures |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        shape = row["shape"]
        md.append(
            "| {wl} | ({q},{n},{it}) | {tt:.3f} | {gt:.3f} | {tp50:.3f} | {gp50:.3f} | {tp99:.3f} | {gp99:.3f} | {tf} | {gf} |".format(
                wl=row["workload"],
                q=shape["query_tokens"],
                n=shape["num_items"],
                it=shape["item_tokens"],
                tt=row["tpu"]["throughput_items_per_sec"],
                gt=row["gpu"]["throughput_items_per_sec"],
                tp50=row["tpu"]["latency_p50_ms"],
                gp50=row["gpu"]["latency_p50_ms"],
                tp99=row["tpu"]["latency_p99_ms"],
                gp99=row["gpu"]["latency_p99_ms"],
                tf=row["tpu"]["num_failures_total"],
                gf=row["gpu"]["num_failures_total"],
            )
        )
    md.append("")
    md.append("## Aggregate Means")
    md.append("")
    agg = report["aggregate_means"]
    md.append(
        f"- TPU throughput={agg['tpu']['throughput_items_per_sec']:.3f}, "
        f"p50={agg['tpu']['latency_p50_ms']:.3f}, p99={agg['tpu']['latency_p99_ms']:.3f}"
    )
    md.append(
        f"- GPU throughput={agg['gpu']['throughput_items_per_sec']:.3f}, "
        f"p50={agg['gpu']['latency_p50_ms']:.3f}, p99={agg['gpu']['latency_p99_ms']:.3f}"
    )
    md.append(
        f"- Delta (TPU-GPU): throughput={agg['delta_tpu_minus_gpu']['throughput_items_per_sec']:+.3f}, "
        f"p50={agg['delta_tpu_minus_gpu']['latency_p50_ms']:+.3f}, "
        f"p99={agg['delta_tpu_minus_gpu']['latency_p99_ms']:+.3f}"
    )
    md.append("")
    md.append("## Reproducibility")
    md.append("")
    md.append(
        f"- Client locations: TPU=`{report['tpu_runtime']['benchmark_client_location']}`, "
        f"GPU=`{report['gpu_runtime']['benchmark_client_location']}`"
    )
    md.append(f"- Baseline config: `{args.baseline_config}`")
    md.append(f"- TPU summary source: `{args.tpu_summary}`")
    md.append(f"- GPU source: `{args.gpu_results}`")
    md.append("")
    args.md_out.write_text("\n".join(md))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
