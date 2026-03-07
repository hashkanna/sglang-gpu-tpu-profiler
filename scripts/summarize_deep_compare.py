#!/usr/bin/env python3
"""Generate a concise deep-dive summary from profiler artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


TPU_METRICS_RE = re.compile(
    r"ScorePathMetrics path=(\S+) items=(\d+) dispatches=(\d+) "
    r"lifecycle_sent=(\d+) lifecycle_recv=(\d+) queue_wait_s=([0-9.]+) "
    r"device_compute_s=([0-9.]+) host_orchestration_s=([0-9.]+) "
    r"fastpath_attempted=(\S+) fastpath_succeeded=(\S+) "
    r"fastpath_fallback_reason=(\S+)"
)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _fmt(v: float | None, places: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{places}f}"


def parse_tpu_metrics(log_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in TPU_METRICS_RE.finditer(log_text):
        rows.append(
            {
                "path": m.group(1),
                "items": int(m.group(2)),
                "dispatches": int(m.group(3)),
                "queue_wait_s": float(m.group(6)),
                "device_compute_s": float(m.group(7)),
                "host_orchestration_s": float(m.group(8)),
                "fastpath_attempted": m.group(9) == "True",
                "fastpath_succeeded": m.group(10) == "True",
                "fallback_reason": m.group(11),
            }
        )
    return rows


def build_summary(run_dir: Path) -> str:
    raw_path = run_dir / "raw_results.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    raw = json.loads(raw_path.read_text())

    gpu_log_path = run_dir / "artifacts" / "gpu" / "gpu_server.log"
    tpu_log_path = run_dir / "artifacts" / "tpu" / "tpu_server.log"

    gpu_log = gpu_log_path.read_text(errors="ignore") if gpu_log_path.exists() else ""
    tpu_log = tpu_log_path.read_text(errors="ignore") if tpu_log_path.exists() else ""

    tpu_rows = parse_tpu_metrics(tpu_log)
    tpu_hot = [r for r in tpu_rows if r["items"] == 500]
    tpu_hot_steady = [r for r in tpu_hot if r["device_compute_s"] < 5.0]

    gpu_tok_s = [float(x) for x in re.findall(r"input throughput \(token/s\): ([0-9.]+)", gpu_log)]

    missing_cache_handle = len(re.findall(r"Missing scoring cache handle", tpu_log))
    cache_evictions = len(re.findall(r"Evicted \d+ expired scoring cache handles", tpu_log))

    lines: list[str] = []
    lines.append(f"# Deep Summary: {run_dir.name}")
    lines.append("")
    lines.append("## Workload Metrics")
    lines.append("")
    lines.append("| Backend | Workload | Throughput (items/s) | p50 (ms) | p99 (ms) | Failures |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for backend in ("gpu", "tpu"):
        for workload, m in raw.get(backend, {}).items():
            lines.append(
                f"| {backend} | {workload} | {m['throughput_items_per_sec']:.1f} | "
                f"{m['latency_p50_ms']:.1f} | {m['latency_p99_ms']:.1f} | {m['num_failures']} |"
            )

    lines.append("")
    lines.append("## TPU Path Timing (ScorePathMetrics)")
    lines.append("")
    lines.append(f"- samples_total: `{len(tpu_rows)}`")
    lines.append(f"- samples_hotshape_items500: `{len(tpu_hot)}`")
    lines.append(f"- fastpath_success_rate_items500: `{_fmt(_median([1.0 if r['fastpath_succeeded'] else 0.0 for r in tpu_hot]), 3)}`")
    lines.append(
        f"- device_compute_s_items500: median=`{_fmt(_median([r['device_compute_s'] for r in tpu_hot]))}` "
        f"steady_median=`{_fmt(_median([r['device_compute_s'] for r in tpu_hot_steady]))}`"
    )
    lines.append(
        f"- host_orchestration_s_items500: median=`{_fmt(_median([r['host_orchestration_s'] for r in tpu_hot]))}` "
        f"steady_median=`{_fmt(_median([r['host_orchestration_s'] for r in tpu_hot_steady]))}`"
    )
    lines.append(f"- queue_wait_s_items500: median=`{_fmt(_median([r['queue_wait_s'] for r in tpu_hot]))}`")
    lines.append(f"- missing_scoring_cache_handle_count: `{missing_cache_handle}`")
    lines.append(f"- scoring_cache_eviction_lines: `{cache_evictions}`")

    lines.append("")
    lines.append("## GPU Runtime Signals")
    lines.append("")
    lines.append(f"- gpu_prefill_throughput_samples: `{len(gpu_tok_s)}`")
    lines.append(f"- gpu_prefill_tok_s_median: `{_fmt(_median(gpu_tok_s), 2)}`")
    if gpu_tok_s:
        s = sorted(gpu_tok_s)
        p10 = s[max(0, int(0.1 * len(s)) - 1)]
        p90 = s[min(len(s) - 1, int(0.9 * len(s)))]
        lines.append(f"- gpu_prefill_tok_s_p10: `{p10:.2f}`")
        lines.append(f"- gpu_prefill_tok_s_p90: `{p90:.2f}`")

    lines.append("")
    lines.append("## Artifact Paths")
    lines.append("")
    lines.append(f"- raw_results: `{raw_path}`")
    lines.append(f"- gpu_log: `{gpu_log_path}`")
    lines.append(f"- tpu_log: `{tpu_log_path}`")
    lines.append(f"- tpu_trace: `{run_dir / 'artifacts' / 'tpu' / 'profile' / 'tpu.trace.json.gz'}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize deep compare artifacts for one run dir.")
    parser.add_argument("run_dir", help="Profiler run directory path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path (default: <run_dir>/deep_summary.md)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    md = build_summary(run_dir)

    output = Path(args.output).resolve() if args.output else run_dir / "deep_summary.md"
    output.write_text(md)
    print(output)


if __name__ == "__main__":
    main()

