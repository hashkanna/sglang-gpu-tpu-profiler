"""CLI entry point for the profiler framework."""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from profiler.config import load_config
from profiler.runner import ScoreRunner
from profiler.analyzer import analyze_results
from profiler.reporter import print_console_report, generate_dashboard_json


def _create_run_dir(base: Path, config_name: str) -> Path:
    """Create a timestamped run directory."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{ts}_{config_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _serialize_metrics(all_results: dict) -> dict:
    """Convert WorkloadMetrics objects to JSON-serializable dicts."""
    out = {}
    for bkey, workloads in all_results.items():
        out[bkey] = {}
        for wname, m in workloads.items():
            out[bkey][wname] = {
                "workload_name": m.workload_name,
                "backend_key": m.backend_key,
                "num_requests": m.num_requests,
                "num_successes": m.num_successes,
                "num_failures": m.num_failures,
                "total_items": m.total_items,
                "total_time_sec": m.total_time_sec,
                "throughput_items_per_sec": m.throughput_items_per_sec,
                "latency_p50_ms": m.latency_p50_ms,
                "latency_p90_ms": m.latency_p90_ms,
                "latency_p95_ms": m.latency_p95_ms,
                "latency_p99_ms": m.latency_p99_ms,
                "latency_mean_ms": m.latency_mean_ms,
                "latency_stdev_ms": m.latency_stdev_ms,
                "latency_min_ms": m.latency_min_ms,
                "latency_max_ms": m.latency_max_ms,
                "raw_latencies_ms": m.raw_latencies_ms,
                "failed_requests": m.failed_requests,
                "error_counts": m.error_counts,
            }
    return out


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmarks against one or both backends."""
    config = load_config(args.config)
    base_dir = Path(args.output)

    backend_keys = [args.backend] if args.backend else None

    print(f"Experiment: {config.experiment.name}")
    print(f"Model: {config.experiment.model}")
    print(f"Workloads: {len(config.workloads)}")
    if backend_keys:
        print(f"Backend: {backend_keys[0]}")
    else:
        print(f"Backends: {', '.join(config.backend_keys)}")

    runner = ScoreRunner(config)
    start = time.monotonic()
    all_results = asyncio.run(runner.run_all(backend_keys))
    elapsed = time.monotonic() - start

    # Save raw results
    run_dir = _create_run_dir(base_dir, config.experiment.name)
    raw_path = run_dir / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(_serialize_metrics(all_results), f, indent=2)
    print(f"\nRaw results saved to: {raw_path}")

    metadata_path = run_dir / "run_metadata.json"
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "name": config.experiment.name,
            "model": config.experiment.model,
        },
        "benchmark": {
            "warmup_requests": config.benchmark.warmup_requests,
            "timed_requests": config.benchmark.timed_requests,
            "concurrency": config.benchmark.concurrency,
            "timeout_sec": config.benchmark.timeout_sec,
        },
        "run_diagnostics": runner.run_diagnostics,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Run metadata saved to: {metadata_path}")

    # If both backends ran, auto-analyze and report
    if len(all_results) >= 2:
        comparison = analyze_results(all_results, config)
        print_console_report(comparison, config)
        dashboard_path = run_dir / "dashboard_data.json"
        generate_dashboard_json(comparison, all_results, config, dashboard_path)
        print(f"Dashboard data saved to: {dashboard_path}")
    else:
        # Single backend — just print summary
        for bkey, workloads in all_results.items():
            backend = config.get_backend(bkey)
            print(f"\n--- {backend.name} Summary ---")
            for wname, m in workloads.items():
                print(f"  {wname}: {m.throughput_items_per_sec:,.1f} items/s, "
                      f"p50={m.latency_p50_ms:,.1f}ms, p99={m.latency_p99_ms:,.1f}ms")

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"Results directory: {run_dir}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze previously saved results."""
    run_dir = Path(args.run_dir)
    raw_path = run_dir / "raw_results.json"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    with open(raw_path) as f:
        raw = json.load(f)

    # Reconstruct WorkloadMetrics from raw JSON
    from profiler.metrics import WorkloadMetrics
    all_results = {}
    for bkey, workloads in raw.items():
        all_results[bkey] = {}
        for wname, mdata in workloads.items():
            all_results[bkey][wname] = WorkloadMetrics(**mdata)

    comparison = analyze_results(all_results, config)
    print_console_report(comparison, config)


def cmd_report(args: argparse.Namespace) -> None:
    """Generate reports from previously saved results."""
    run_dir = Path(args.run_dir)
    raw_path = run_dir / "raw_results.json"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    with open(raw_path) as f:
        raw = json.load(f)

    from profiler.metrics import WorkloadMetrics
    all_results = {}
    for bkey, workloads in raw.items():
        all_results[bkey] = {}
        for wname, mdata in workloads.items():
            all_results[bkey][wname] = WorkloadMetrics(**mdata)

    comparison = analyze_results(all_results, config)

    fmt = args.format
    if fmt in ("console", "all"):
        print_console_report(comparison, config)
    if fmt in ("json", "dashboard", "all"):
        dashboard_path = run_dir / "dashboard_data.json"
        generate_dashboard_json(comparison, all_results, config, dashboard_path)
        print(f"Dashboard data saved to: {dashboard_path}")
    if fmt in ("html", "all"):
        from profiler.reporter import generate_dashboard_html
        html_path = run_dir / "dashboard.html"
        dashboard_path = run_dir / "dashboard_data.json"
        if not dashboard_path.exists():
            generate_dashboard_json(comparison, all_results, config, dashboard_path)
        generate_dashboard_html(dashboard_path, html_path)
        print(f"Dashboard HTML saved to: {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="profiler",
        description="sglang GPU vs TPU profiler comparison framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run benchmarks")
    p_run.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p_run.add_argument("--backend", "-b", help="Run only this backend (e.g. gpu, tpu)")
    p_run.add_argument("--output", "-o", default="results", help="Output base directory")
    p_run.set_defaults(func=cmd_run)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze saved results")
    p_analyze.add_argument("run_dir", help="Path to run directory")
    p_analyze.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p_analyze.set_defaults(func=cmd_analyze)

    # report
    p_report = subparsers.add_parser("report", help="Generate reports")
    p_report.add_argument("run_dir", help="Path to run directory")
    p_report.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p_report.add_argument(
        "--format", "-f",
        choices=["console", "json", "dashboard", "html", "all"],
        default="all",
        help="Output format",
    )
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
