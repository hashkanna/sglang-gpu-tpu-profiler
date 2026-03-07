#!/usr/bin/env python3
"""Run repeated TPU scoring matrix sweeps and emit repeatability variance summary."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pr28_baseline import (
    DEFAULT_BASELINE_PATH,
    baseline_workloads,
    benchmark_defaults,
    load_baseline,
    matrix_defaults,
    tpu_defaults,
)


RUN_DIR_RE = re.compile(r"ips(\d+)-r(\d+)")


def run_cmd(cmd: list[str], *, cwd: Path, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(x) for x in cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=capture_output,
        check=True,
    )


def parse_matrix_group(stdout: str) -> Path:
    m = re.search(r"Matrix group directory:\s*(.+)", stdout)
    if not m:
        raise RuntimeError("Failed to parse Matrix group directory from matrix runner output.")
    return Path(m.group(1).strip()).resolve()


def _cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_v = statistics.mean(values)
    if mean_v == 0:
        return 0.0
    return statistics.pstdev(values) / mean_v


def _iqr_outlier_count(values: list[float]) -> int:
    if len(values) < 4:
        return 0
    q = statistics.quantiles(values, n=4, method="inclusive")
    q1 = float(q[0])
    q3 = float(q[2])
    iqr = q3 - q1
    if iqr <= 0:
        return 0
    lo = q1 - (1.5 * iqr)
    hi = q3 + (1.5 * iqr)
    return sum(1 for v in values if v < lo or v > hi)


def _aggregate_repeatability_rows(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    workloads: dict[str, dict[int, dict[str, list[float]]]] = {}
    for run in runs:
        ips = int(run.get("items_per_step", 0))
        for wl_name, wl in (run.get("workloads", {}) or {}).items():
            wl_map = workloads.setdefault(str(wl_name), {})
            series = wl_map.setdefault(
                ips,
                {
                    "throughput_items_per_sec": [],
                    "latency_p50_ms": [],
                    "latency_p99_ms": [],
                    "p99_p50_ratio": [],
                },
            )
            throughput = float(wl.get("throughput_items_per_sec", 0.0))
            p50 = float(wl.get("latency_p50_ms", 0.0))
            p99 = float(wl.get("latency_p99_ms", 0.0))
            series["throughput_items_per_sec"].append(throughput)
            series["latency_p50_ms"].append(p50)
            series["latency_p99_ms"].append(p99)
            ratio = wl.get("p99_p50_ratio")
            if ratio is None and p50 > 0:
                ratio = p99 / p50
            if ratio is not None:
                series["p99_p50_ratio"].append(float(ratio))

    out: dict[str, list[dict[str, Any]]] = {}
    for wl_name, by_ips in workloads.items():
        rows: list[dict[str, Any]] = []
        for ips, s in sorted(by_ips.items()):
            tput = s["throughput_items_per_sec"]
            p50 = s["latency_p50_ms"]
            p99 = s["latency_p99_ms"]
            ratio = s["p99_p50_ratio"]
            rows.append(
                {
                    "items_per_step": ips,
                    "samples": len(tput),
                    "throughput_median_items_per_sec": float(statistics.median(tput)) if tput else 0.0,
                    "throughput_cv": _cv(tput),
                    "throughput_outlier_count": _iqr_outlier_count(tput),
                    "latency_p50_median_ms": float(statistics.median(p50)) if p50 else 0.0,
                    "latency_p99_median_ms": float(statistics.median(p99)) if p99 else 0.0,
                    "latency_p99_cv": _cv(p99),
                    "latency_p99_outlier_count": _iqr_outlier_count(p99),
                    "p99_p50_ratio_median": float(statistics.median(ratio)) if ratio else None,
                }
            )
        out[wl_name] = rows
    return out


def summarize_repeatability_from_matrix_summary(matrix_summary_path: Path) -> dict[str, Any]:
    payload = json.loads(matrix_summary_path.read_text())
    runs = payload.get("runs", [])
    out = _aggregate_repeatability_rows(runs)
    return {
        "matrix_summary_json": str(matrix_summary_path),
        "matrix_group_dir": str(matrix_summary_path.parent),
        "repeats_requested": int(payload.get("repeats_requested", 0)),
        "timed_requests_per_repeat": int(payload.get("timed_requests_per_repeat", 0)),
        "workloads": out,
    }


def summarize_repeatability_from_group_dir(matrix_group_dir: Path) -> dict[str, Any]:
    runs_dir = matrix_group_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Missing runs directory: {runs_dir}")

    runs: list[dict[str, Any]] = []
    repeat_indices: list[int] = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        raw_path = run_dir / "raw_results.json"
        if not raw_path.exists():
            continue
        m = RUN_DIR_RE.search(run_dir.name)
        ips = int(m.group(1)) if m else 0
        repeat_idx = int(m.group(2)) if m else 0
        if repeat_idx > 0:
            repeat_indices.append(repeat_idx)
        raw = json.loads(raw_path.read_text())
        runs.append(
            {
                "items_per_step": ips,
                "repeat_idx": repeat_idx,
                "workloads": raw.get("tpu", {}) or {},
            }
        )

    if not runs:
        raise RuntimeError(f"No run rows found under: {runs_dir}")

    out = _aggregate_repeatability_rows(runs)
    repeats_requested = max(repeat_indices) if repeat_indices else len(runs)
    return {
        "matrix_summary_json": None,
        "matrix_group_dir": str(matrix_group_dir),
        "repeats_requested": int(repeats_requested),
        "timed_requests_per_repeat": None,
        "workloads": out,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# TPU Repeatability Summary")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(
        f"- matrix_summary_json: `{report['matrix_summary_json'] or 'n/a (derived from raw run files)'}`"
    )
    lines.append(f"- repeats_requested: `{report['repeats_requested']}`")
    lines.append(
        f"- timed_requests_per_repeat: `{report['timed_requests_per_repeat'] if report['timed_requests_per_repeat'] is not None else 'n/a'}`"
    )
    lines.append("")

    for wl_name in sorted(report["workloads"].keys()):
        lines.append(f"## Workload: {wl_name}")
        lines.append("")
        lines.append(
            "| items_per_step | samples | tput_med (items/s) | tput_cv | tput_outliers | "
            "p50_med (ms) | p99_med (ms) | p99_cv | p99_outliers | p99/p50_med |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in report["workloads"][wl_name]:
            p99_p50 = row["p99_p50_ratio_median"]
            p99_p50_txt = f"{p99_p50:.4f}" if p99_p50 is not None else "n/a"
            tput_cv = row["throughput_cv"]
            p99_cv = row["latency_p99_cv"]
            lines.append(
                f"| {row['items_per_step']} | {row['samples']} | "
                f"{row['throughput_median_items_per_sec']:.2f} | "
                f"{'n/a' if tput_cv is None else f'{tput_cv:.4f}'} | "
                f"{row['throughput_outlier_count']} | "
                f"{row['latency_p50_median_ms']:.2f} | {row['latency_p99_median_ms']:.2f} | "
                f"{'n/a' if p99_cv is None else f'{p99_cv:.4f}'} | "
                f"{row['latency_p99_outlier_count']} | {p99_p50_txt} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    baseline = load_baseline(DEFAULT_BASELINE_PATH)
    benchmark = benchmark_defaults(baseline)
    matrix = matrix_defaults(baseline)
    tpu = tpu_defaults(baseline)
    workload_filter_default = ",".join(w["name"] for w in baseline_workloads(baseline))

    parser = argparse.ArgumentParser(description="Run repeated TPU matrix and summarize repeatability.")
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--tpu-name")
    parser.add_argument("--tpu-zone")
    parser.add_argument("--ssh-mode", choices=["direct", "gcloud"], default="direct")
    parser.add_argument("--tpu-host", default=None)
    parser.add_argument("--ssh-user", default="kanna")
    parser.add_argument("--ssh-key", default=str(Path.home() / ".ssh/google_compute_engine"))
    parser.add_argument("--tpu-repo-path", default=tpu["repo_path"])
    parser.add_argument("--tpu-url", default=tpu["url"])
    parser.add_argument("--tpu-connection-mode", choices=["auto", "direct", "tunnel"], default="auto")
    parser.add_argument("--model", default=str(baseline["experiment"]["model"]))
    parser.add_argument("--workload-filter", default=workload_filter_default)
    parser.add_argument("--items-per-step", default="96")
    parser.add_argument(
        "--align-items-per-step-with-workloads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject workload item count as extra items_per_step candidate when single-workload "
            "mode is used."
        ),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=benchmark["warmup_requests"])
    parser.add_argument("--timed-requests", type=int, default=benchmark["timed_requests"])
    parser.add_argument("--concurrency", type=int, default=benchmark["concurrency"])
    parser.add_argument("--timeout-sec", type=int, default=benchmark["timeout_sec"])
    parser.add_argument("--max-running-requests", type=int, default=matrix["max_running_requests"])
    parser.add_argument(
        "--multi-item-extend-batch-size",
        type=int,
        default=matrix["multi_item_extend_batch_size"],
    )
    parser.add_argument(
        "--precompile-token-paddings",
        default=",".join(str(v) for v in matrix["precompile_token_paddings"]),
    )
    parser.add_argument(
        "--precompile-bs-paddings",
        default=",".join(str(v) for v in matrix["precompile_bs_paddings"]),
    )
    parser.add_argument(
        "--allow-score-full-vocab-fallback",
        action=argparse.BooleanOptionalAction,
        default=bool(matrix.get("allow_score_full_vocab_fallback", False)),
    )
    parser.add_argument(
        "--require-cache-transition-exercise",
        action=argparse.BooleanOptionalAction,
        default=bool(matrix.get("require_cache_transition_exercise", False)),
    )
    parser.add_argument(
        "--analyze-existing-group",
        default=None,
        help=(
            "Skip execution and summarize an existing matrix group directory "
            "(expects runs/*/raw_results.json)."
        ),
    )
    parser.add_argument("--output-base", default="results/tpu_tuning/repeatability")
    parser.add_argument("--profiler-dir", default=".")
    args = parser.parse_args()

    profiler_dir = Path(args.profiler_dir).resolve()
    matrix_script = profiler_dir / "scripts" / "run_tpu_scoring_matrix.py"
    if not matrix_script.exists():
        raise FileNotFoundError(f"Missing matrix runner: {matrix_script}")

    matrix_group_dir: Path
    if args.analyze_existing_group:
        matrix_group_dir = Path(args.analyze_existing_group).expanduser().resolve()
        if not matrix_group_dir.exists():
            raise FileNotFoundError(f"--analyze-existing-group not found: {matrix_group_dir}")
        report = summarize_repeatability_from_group_dir(matrix_group_dir)
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        out_json = matrix_group_dir / "repeatability_summary.json"
        out_md = matrix_group_dir / "repeatability_report.md"
        out_json.write_text(json.dumps(report, indent=2))
        out_md.write_text(render_markdown(report))
        print(f"repeatability_summary_json={out_json}")
        print(f"repeatability_report_md={out_md}")
        return

    if not args.tpu_name or not args.tpu_zone:
        raise ValueError("--tpu-name and --tpu-zone are required unless --analyze-existing-group is used.")

    cmd = [
        "python3",
        str(matrix_script),
        "--baseline-config",
        args.baseline_config,
        "--tpu-name",
        str(args.tpu_name),
        "--tpu-zone",
        str(args.tpu_zone),
        "--ssh-mode",
        args.ssh_mode,
        "--ssh-user",
        args.ssh_user,
        "--ssh-key",
        args.ssh_key,
        "--tpu-repo-path",
        args.tpu_repo_path,
        "--tpu-url",
        args.tpu_url,
        "--tpu-connection-mode",
        args.tpu_connection_mode,
        "--model",
        args.model,
        "--workload-filter",
        args.workload_filter,
        "--items-per-step",
        args.items_per_step,
        "--align-items-per-step-with-workloads"
        if args.align_items_per_step_with_workloads
        else "--no-align-items-per-step-with-workloads",
        "--max-running-requests",
        str(args.max_running_requests),
        "--multi-item-extend-batch-size",
        str(args.multi_item_extend_batch_size),
        "--precompile-token-paddings",
        args.precompile_token_paddings,
        "--precompile-bs-paddings",
        args.precompile_bs_paddings,
        "--repeats",
        str(args.repeats),
        "--warmup-requests",
        str(args.warmup_requests),
        "--timed-requests",
        str(args.timed_requests),
        "--concurrency",
        str(args.concurrency),
        "--timeout-sec",
        str(args.timeout_sec),
        "--stabilize-with-workload-warmup",
        "--reuse-server-per-candidate",
        "--output-base",
        args.output_base,
        "--profiler-dir",
        str(profiler_dir),
    ]
    cmd.append(
        "--allow-score-full-vocab-fallback"
        if args.allow_score_full_vocab_fallback
        else "--no-allow-score-full-vocab-fallback"
    )
    cmd.append(
        "--require-cache-transition-exercise"
        if args.require_cache_transition_exercise
        else "--no-require-cache-transition-exercise"
    )
    if args.tpu_host:
        cmd.extend(["--tpu-host", args.tpu_host])

    proc = run_cmd(cmd, cwd=profiler_dir, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    matrix_group_dir = parse_matrix_group(proc.stdout)
    matrix_summary_path = matrix_group_dir / "matrix_summary.json"
    if matrix_summary_path.exists():
        report = summarize_repeatability_from_matrix_summary(matrix_summary_path)
    else:
        report = summarize_repeatability_from_group_dir(matrix_group_dir)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    out_json = matrix_group_dir / "repeatability_summary.json"
    out_md = matrix_group_dir / "repeatability_report.md"
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text(render_markdown(report))
    print(f"repeatability_summary_json={out_json}")
    print(f"repeatability_report_md={out_md}")


if __name__ == "__main__":
    main()
