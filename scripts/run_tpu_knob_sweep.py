#!/usr/bin/env python3
"""Sweep TPU scheduling knobs at fixed items_per_step and rank reliability/tail behavior."""

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
    baseline_workloads,
    benchmark_defaults,
    load_baseline,
    matrix_defaults,
    tpu_defaults,
)


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
        raise RuntimeError("Failed to parse matrix group directory from output.")
    return Path(m.group(1).strip()).resolve()


def resolve_baseline_path(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE_PATH))
    args, _ = parser.parse_known_args(argv)
    return Path(args.baseline_config).expanduser().resolve()


def _score_combo(combo: dict[str, Any]) -> float:
    # Higher is better. Penalize failures and bad tails heavily.
    score = 0.0
    for wl in WORKLOAD_ORDER:
        row = combo["workloads"].get(wl, {})
        tput = float(row.get("throughput_median_items_per_sec") or 0.0)
        p99 = float(row.get("latency_p99_median_ms") or 0.0)
        fail = float(row.get("failure_rate") or 0.0)
        util = float(row.get("score_utilization_pct_median") or 0.0)
        tput_term = min(1.0, tput / 500.0)
        p99_term = 0.0 if p99 <= 0 else min(1.0, 1000.0 / p99)
        util_term = min(1.0, util / 30.0)
        score += (0.5 * tput_term) + (0.3 * p99_term) + (0.2 * util_term) - (1.5 * fail)
    return score


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# TPU Knob Sweep: {report['name']}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- fixed_items_per_step: `{report['config']['items_per_step']}`")
    lines.append(f"- repeats: `{report['config']['repeats']}`")
    lines.append(f"- timed_requests: `{report['config']['timed_requests']}`")
    lines.append("")
    lines.append("## Combo Ranking")
    lines.append("")
    lines.append("| Rank | max_running_requests | extend_batch_size | score | hotshape fail | small fail | medium fail |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for i, combo in enumerate(report["ranked_combos"], start=1):
        lines.append(
            f"| {i} | {combo['max_running_requests']} | {combo['extend_batch_size']} | "
            f"{combo['score']:.4f} | "
            f"{float(combo['workloads']['pr28_hotshape']['failure_rate']) * 100.0:.1f}% | "
            f"{float(combo['workloads']['small_batch']['failure_rate']) * 100.0:.1f}% | "
            f"{float(combo['workloads']['medium_batch']['failure_rate']) * 100.0:.1f}% |"
        )
    lines.append("")

    best = report["ranked_combos"][0] if report["ranked_combos"] else None
    if best:
        lines.append("## Recommended Knobs")
        lines.append("")
        lines.append(f"- max_running_requests: `{best['max_running_requests']}`")
        lines.append(f"- multi_item_extend_batch_size: `{best['extend_batch_size']}`")
        lines.append(f"- composite_score: `{best['score']:.4f}`")
        lines.append(f"- matrix_group_dir: `{best['matrix_group_dir']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    baseline_path = resolve_baseline_path(sys.argv[1:])
    baseline = load_baseline(baseline_path)
    benchmark = benchmark_defaults(baseline)
    matrix = matrix_defaults(baseline)
    shape = dict(matrix.get("shape_contract", {}))
    tpu = tpu_defaults(baseline)
    workload_filter_default = ",".join(w["name"] for w in baseline_workloads(baseline))
    default_items = (
        96 if 96 in matrix["items_per_step_candidates"] else matrix["items_per_step_candidates"][0]
    )
    shape_query_default = ",".join(str(v) for v in shape.get("query_token_buckets", [120, 500, 2000]))
    shape_item_default = ",".join(str(v) for v in shape.get("item_token_buckets", [20]))
    shape_num_items_default = ",".join(str(v) for v in shape.get("num_items_buckets", [10, 100, 500]))

    parser = argparse.ArgumentParser(description="Sweep TPU knob combos at fixed items_per_step.")
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
    parser.add_argument("--items-per-step", type=int, default=default_items)
    parser.add_argument(
        "--align-items-per-step-with-workloads",
        action=argparse.BooleanOptionalAction,
        default=bool(matrix.get("align_items_per_step_with_workloads", True)),
    )
    parser.add_argument("--max-running-requests-values", default="64,96,128")
    parser.add_argument("--extend-batch-values", default="64,96")
    parser.add_argument(
        "--precompile-token-paddings",
        default=",".join(str(v) for v in matrix["precompile_token_paddings"]),
    )
    parser.add_argument(
        "--precompile-bs-paddings",
        default=",".join(str(v) for v in matrix["precompile_bs_paddings"]),
    )
    parser.add_argument("--repeats", type=int, default=3)
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
    parser.add_argument("--workload-filter", default=workload_filter_default)
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
        default="results/tpu_tuning/knob_sweep",
        help="Base output directory (relative to profiler dir).",
    )
    parser.add_argument("--profiler-dir", default=".")
    args = parser.parse_args()

    profiler_dir = Path(args.profiler_dir).resolve()
    script = profiler_dir / "scripts" / "run_tpu_scoring_matrix.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing matrix runner: {script}")

    max_running_values = [int(x.strip()) for x in args.max_running_requests_values.split(",") if x.strip()]
    extend_values = [int(x.strip()) for x in args.extend_batch_values.split(",") if x.strip()]
    if not max_running_values or not extend_values:
        raise ValueError("max-running-requests-values and extend-batch-values must be non-empty.")

    group_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_knob-sweep")
    output_dir = (profiler_dir / args.output_base / group_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combos: list[dict[str, Any]] = []
    for mrr in max_running_values:
        for ebs in extend_values:
            combo_base = output_dir / f"mrr{mrr}_ebs{ebs}"
            rel_out_base = str(combo_base.relative_to(profiler_dir))
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
                args.workload_filter,
                "--items-per-step",
                str(args.items_per_step),
                "--align-items-per-step-with-workloads"
                if args.align_items_per_step_with_workloads
                else "--no-align-items-per-step-with-workloads",
                "--max-running-requests",
                str(mrr),
                "--multi-item-extend-batch-size",
                str(ebs),
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
            summary = json.loads((matrix_group_dir / "matrix_summary.json").read_text())
            workloads_rows = summary["aggregates"]["by_workload"]
            combo = {
                "max_running_requests": mrr,
                "extend_batch_size": ebs,
                "matrix_group_dir": str(matrix_group_dir),
                "workloads": {},
            }
            for wl in WORKLOAD_ORDER:
                rows = workloads_rows.get(wl, [])
                combo["workloads"][wl] = rows[0] if rows else {}
            combo["score"] = _score_combo(combo)
            combos.append(combo)

    combos.sort(key=lambda c: c["score"], reverse=True)

    report = {
        "name": group_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "config": {
            "items_per_step": args.items_per_step,
            "repeats": args.repeats,
            "timed_requests": args.timed_requests,
            "max_running_requests_values": max_running_values,
            "extend_batch_values": extend_values,
        },
        "ranked_combos": combos,
    }

    summary_json = output_dir / "knob_sweep_summary.json"
    summary_md = output_dir / "knob_sweep_report.md"
    summary_json.write_text(json.dumps(report, indent=2))
    summary_md.write_text(render_markdown(report))

    print(f"knob_sweep_summary_json={summary_json}")
    print(f"knob_sweep_report_md={summary_md}")


if __name__ == "__main__":
    main()
