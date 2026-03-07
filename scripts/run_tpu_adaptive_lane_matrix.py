#!/usr/bin/env python3
"""Run per-workload TPU items_per_step sweeps and emit an adaptive-lane recommendation."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pr28_baseline import (
    DEFAULT_BASELINE_PATH,
    benchmark_defaults,
    load_baseline,
    matrix_defaults,
    tpu_defaults,
)


WORKLOAD_CANDIDATE_FLAGS = {
    "pr28_hotshape": "hotshape_items_per_step",
    "small_batch": "small_items_per_step",
    "medium_batch": "medium_items_per_step",
}
WORKLOAD_ORDER = ["pr28_hotshape", "small_batch", "medium_batch"]


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
        raise RuntimeError("Failed to parse Matrix group directory from child output.")
    return Path(m.group(1).strip()).resolve()


def resolve_baseline_path(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE_PATH))
    args, _ = parser.parse_known_args(argv)
    return Path(args.baseline_config).expanduser().resolve()


def load_recommended_row(summary_path: Path, workload: str) -> tuple[dict[str, Any], int]:
    summary = json.loads(summary_path.read_text())
    rec_ips = int(summary["recommended"]["items_per_step"])
    rows = summary["aggregates"]["by_workload"][workload]
    rec_row = None
    for row in rows:
        if int(row["items_per_step"]) == rec_ips:
            rec_row = row
            break
    if rec_row is None and rows:
        rec_row = rows[0]
    if rec_row is None:
        raise RuntimeError(f"No ranking rows found for workload={workload} in {summary_path}")
    return rec_row, rec_ips


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# TPU Adaptive Lane Recommendation: {report['name']}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- output_dir: `{report['output_dir']}`")
    lines.append(f"- repeats: `{report['config']['repeats']}`")
    lines.append(f"- timed_requests: `{report['config']['timed_requests']}`")
    lines.append("")
    lines.append("## Recommended items_per_step by Workload")
    lines.append("")
    lines.append("| Workload | Recommended ips | tput_med (items/s) | p50_med (ms) | p99_med (ms) | failure_rate | util_med (%) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        row = report["workloads"][wl]["recommended_row"]
        lines.append(
            f"| {wl} | {report['workloads'][wl]['recommended_items_per_step']} | "
            f"{float(row['throughput_median_items_per_sec']):.1f} | "
            f"{float(row['latency_p50_median_ms']):.1f} | "
            f"{float(row['latency_p99_median_ms']):.1f} | "
            f"{float(row['failure_rate']) * 100.0:.1f}% | "
            f"{float(row['score_utilization_pct_median']):.1f} |"
        )
    lines.append("")
    lines.append("## Matrix Artifacts")
    lines.append("")
    for wl in WORKLOAD_ORDER:
        lines.append(f"- {wl}: `{report['workloads'][wl]['matrix_group_dir']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    baseline_path = resolve_baseline_path(sys.argv[1:])
    baseline = load_baseline(baseline_path)
    benchmark = benchmark_defaults(baseline)
    matrix = matrix_defaults(baseline)
    shape = dict(matrix.get("shape_contract", {}))
    tpu = tpu_defaults(baseline)
    default_ips_csv = ",".join(str(v) for v in matrix["items_per_step_candidates"])
    shape_query_default = ",".join(str(v) for v in shape.get("query_token_buckets", [120, 500, 2000]))
    shape_item_default = ",".join(str(v) for v in shape.get("item_token_buckets", [20]))
    shape_num_items_default = ",".join(str(v) for v in shape.get("num_items_buckets", [10, 100, 500]))

    parser = argparse.ArgumentParser(
        description="Per-workload adaptive-lane matrix runner for TPU scoring."
    )
    parser.add_argument("--baseline-config", default=str(baseline_path))
    parser.add_argument("--tpu-name", required=True)
    parser.add_argument("--tpu-zone", required=True)
    parser.add_argument("--ssh-mode", choices=["direct", "gcloud"], default="direct")
    parser.add_argument("--tpu-host", default=None)
    parser.add_argument("--ssh-user", default="kanna")
    parser.add_argument("--ssh-key", default=str(Path.home() / ".ssh/google_compute_engine"))
    parser.add_argument("--tpu-repo-path", default=tpu["repo_path"])
    parser.add_argument("--tpu-url", default=tpu["url"])
    parser.add_argument(
        "--tpu-connection-mode",
        choices=["auto", "direct", "tunnel"],
        default="auto",
    )
    parser.add_argument(
        "--tpu-tunnel-autostart",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tpu-tunnel-local-port", type=int, default=None)
    parser.add_argument("--tpu-tunnel-remote-port", type=int, default=None)
    parser.add_argument("--model", default=str(baseline["experiment"]["model"]))
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=matrix["max_running_requests"],
    )
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
    parser.add_argument("--hotshape-items-per-step", default=default_ips_csv)
    parser.add_argument("--small-items-per-step", default=default_ips_csv)
    parser.add_argument("--medium-items-per-step", default=default_ips_csv)
    parser.add_argument(
        "--align-items-per-step-with-workloads",
        action=argparse.BooleanOptionalAction,
        default=bool(matrix.get("align_items_per_step_with_workloads", True)),
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=benchmark["warmup_requests"])
    parser.add_argument("--timed-requests", type=int, default=benchmark["timed_requests"])
    parser.add_argument("--concurrency", type=int, default=benchmark["concurrency"])
    parser.add_argument("--timeout-sec", type=int, default=benchmark["timeout_sec"])
    parser.add_argument(
        "--request-retry-attempts",
        type=int,
        default=int(benchmark.get("request_retry_attempts", 3)),
    )
    parser.add_argument(
        "--request-retry-backoff-sec",
        type=float,
        default=float(benchmark.get("request_retry_backoff_sec", 0.25)),
    )
    parser.add_argument(
        "--reuse-server-per-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse TPU server across repeats for each candidate (default: true).",
    )
    parser.add_argument(
        "--allow-timed-xla-compilation",
        action="store_true",
        help="Allow matrix candidates with timed first-request outlier signals.",
    )
    parser.add_argument(
        "--shape-contract-enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(shape.get("enabled", True)),
    )
    parser.add_argument(
        "--shape-contract-use-token-ids",
        action=argparse.BooleanOptionalAction,
        default=bool(shape.get("use_token_ids", True)),
    )
    parser.add_argument(
        "--shape-contract-strict",
        action=argparse.BooleanOptionalAction,
        default=bool(shape.get("strict", True)),
    )
    parser.add_argument("--shape-query-token-buckets", default=shape_query_default)
    parser.add_argument("--shape-item-token-buckets", default=shape_item_default)
    parser.add_argument("--shape-num-items-buckets", default=shape_num_items_default)
    parser.add_argument("--shape-pad-token-id", type=int, default=int(shape.get("pad_token_id", 0)))
    parser.add_argument(
        "--shape-query-fill-token-id",
        type=int,
        default=int(shape.get("query_fill_token_id", 42)),
    )
    parser.add_argument(
        "--shape-item-fill-token-id",
        type=int,
        default=int(shape.get("item_fill_token_id", 84)),
    )
    parser.add_argument(
        "--allow-shape-contract-violations",
        action="store_true",
        help="Allow matrix candidates with shape-contract violations.",
    )
    parser.add_argument(
        "--allow-score-full-vocab-fallback",
        action=argparse.BooleanOptionalAction,
        default=bool(matrix.get("allow_score_full_vocab_fallback", False)),
        help="Allow score-path candidates that fallback away from label-only fastpath.",
    )
    parser.add_argument(
        "--require-cache-transition-exercise",
        action=argparse.BooleanOptionalAction,
        default=bool(matrix.get("require_cache_transition_exercise", False)),
        help="Require cache-transition events (missing cache handle) during candidate runs.",
    )
    parser.add_argument(
        "--auto-bump-lane-capacity-with-workload",
        action=argparse.BooleanOptionalAction,
        default=bool(matrix.get("auto_bump_lane_capacity_with_workload", False)),
    )
    parser.add_argument(
        "--lane-capacity-bump-cap",
        type=int,
        default=int(matrix.get("lane_capacity_bump_cap", 128)),
    )
    parser.add_argument(
        "--stabilization-sleep-sec",
        type=float,
        default=matrix["stabilization_sleep_sec"],
    )
    parser.add_argument(
        "--workload-warmup-attempts",
        type=int,
        default=matrix["workload_warmup_attempts"],
    )
    parser.add_argument(
        "--workload-warmup-backoff-sec",
        type=float,
        default=matrix["workload_warmup_backoff_sec"],
    )
    parser.add_argument(
        "--warmup-all-shape-buckets",
        action=argparse.BooleanOptionalAction,
        default=bool(matrix.get("warmup_all_shape_buckets", True)),
    )
    parser.add_argument(
        "--warmup-max-shape-bucket-requests",
        type=int,
        default=int(matrix.get("warmup_max_shape_bucket_requests", 64)),
    )
    parser.add_argument("--precheck-attempts", type=int, default=4)
    parser.add_argument("--precheck-backoff-sec", type=float, default=2.0)
    parser.add_argument("--precheck-health-timeout-sec", type=int, default=90)
    parser.add_argument("--precheck-score-timeout-sec", type=int, default=180)
    parser.add_argument(
        "--multi-item-prefill-extend-cache-timeout-sec",
        type=float,
        default=None,
        help="Optional override for cache-timeout stress testing of hit/miss transitions.",
    )
    parser.add_argument(
        "--output-base",
        default="results/tpu_tuning/adaptive_lane",
        help="Base output directory (relative to profiler dir).",
    )
    parser.add_argument("--profiler-dir", default=".")
    args = parser.parse_args()

    profiler_dir = Path(args.profiler_dir).resolve()
    group_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_adaptive-lane")
    output_dir = (profiler_dir / args.output_base / group_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script = profiler_dir / "scripts" / "run_tpu_scoring_matrix.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing matrix script: {script}")

    workload_results: dict[str, Any] = {}
    for wl in WORKLOAD_ORDER:
        ips_flag = WORKLOAD_CANDIDATE_FLAGS[wl]
        ips_values = getattr(args, ips_flag)
        wl_output_base = output_dir / f"matrix_{wl}"
        rel_out_base = str(wl_output_base.relative_to(profiler_dir))

        cmd = [
            "python3",
            str(script),
            "--baseline-config",
            args.baseline_config,
            "--tpu-name",
            args.tpu_name,
            "--tpu-zone",
            args.tpu_zone,
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
            "--shape-query-token-buckets",
            args.shape_query_token_buckets,
            "--shape-item-token-buckets",
            args.shape_item_token_buckets,
            "--shape-num-items-buckets",
            args.shape_num_items_buckets,
            "--shape-pad-token-id",
            str(args.shape_pad_token_id),
            "--shape-query-fill-token-id",
            str(args.shape_query_fill_token_id),
            "--shape-item-fill-token-id",
            str(args.shape_item_fill_token_id),
            "--lane-capacity-bump-cap",
            str(args.lane_capacity_bump_cap),
            "--workload-filter",
            wl,
            "--items-per-step",
            ips_values,
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
            "--request-retry-attempts",
            str(args.request_retry_attempts),
            "--request-retry-backoff-sec",
            str(args.request_retry_backoff_sec),
            "--stabilize-with-workload-warmup",
            "--workload-warmup-attempts",
            str(args.workload_warmup_attempts),
            "--workload-warmup-backoff-sec",
            str(args.workload_warmup_backoff_sec),
            "--warmup-max-shape-bucket-requests",
            str(args.warmup_max_shape_bucket_requests),
            "--stabilization-sleep-sec",
            str(args.stabilization_sleep_sec),
            "--precheck-attempts",
            str(args.precheck_attempts),
            "--precheck-backoff-sec",
            str(args.precheck_backoff_sec),
            "--precheck-health-timeout-sec",
            str(args.precheck_health_timeout_sec),
            "--precheck-score-timeout-sec",
            str(args.precheck_score_timeout_sec),
            "--output-base",
            rel_out_base,
            "--profiler-dir",
            str(profiler_dir),
        ]
        if args.multi_item_prefill_extend_cache_timeout_sec is not None:
            cmd.extend(
                [
                    "--multi-item-prefill-extend-cache-timeout-sec",
                    str(args.multi_item_prefill_extend_cache_timeout_sec),
                ]
            )
        if args.reuse_server_per_candidate:
            cmd.append("--reuse-server-per-candidate")
        if args.allow_timed_xla_compilation:
            cmd.append("--allow-timed-xla-compilation")
        if args.allow_shape_contract_violations:
            cmd.append("--allow-shape-contract-violations")
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
        cmd.append(
            "--shape-contract-enabled"
            if args.shape_contract_enabled
            else "--no-shape-contract-enabled"
        )
        cmd.append(
            "--shape-contract-use-token-ids"
            if args.shape_contract_use_token_ids
            else "--no-shape-contract-use-token-ids"
        )
        cmd.append(
            "--shape-contract-strict"
            if args.shape_contract_strict
            else "--no-shape-contract-strict"
        )
        cmd.append(
            "--auto-bump-lane-capacity-with-workload"
            if args.auto_bump_lane_capacity_with_workload
            else "--no-auto-bump-lane-capacity-with-workload"
        )
        cmd.append(
            "--warmup-all-shape-buckets"
            if args.warmup_all_shape_buckets
            else "--no-warmup-all-shape-buckets"
        )
        if args.tpu_host:
            cmd.extend(["--tpu-host", args.tpu_host])
        if args.tpu_tunnel_local_port is not None:
            cmd.extend(["--tpu-tunnel-local-port", str(args.tpu_tunnel_local_port)])
        if args.tpu_tunnel_remote_port is not None:
            cmd.extend(["--tpu-tunnel-remote-port", str(args.tpu_tunnel_remote_port)])
        if not args.tpu_tunnel_autostart:
            cmd.append("--no-tpu-tunnel-autostart")

        proc = run_cmd(cmd, cwd=profiler_dir, capture_output=True)
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)

        matrix_group_dir = parse_matrix_group(proc.stdout)
        summary_path = matrix_group_dir / "matrix_summary.json"
        row, rec_ips = load_recommended_row(summary_path, wl)
        workload_results[wl] = {
            "matrix_group_dir": str(matrix_group_dir),
            "matrix_summary_json": str(summary_path),
            "recommended_items_per_step": rec_ips,
            "recommended_row": row,
        }

    report = {
        "name": group_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "config": {
            "repeats": args.repeats,
            "timed_requests": args.timed_requests,
            "max_running_requests": args.max_running_requests,
            "multi_item_extend_batch_size": args.multi_item_extend_batch_size,
        },
        "workloads": workload_results,
    }

    summary_json = output_dir / "adaptive_lane_summary.json"
    summary_md = output_dir / "adaptive_lane_report.md"
    summary_json.write_text(json.dumps(report, indent=2))
    summary_md.write_text(render_markdown(report))

    print(f"adaptive_summary_json={summary_json}")
    print(f"adaptive_report_md={summary_md}")


if __name__ == "__main__":
    main()
