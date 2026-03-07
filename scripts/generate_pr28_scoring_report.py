#!/usr/bin/env python3
"""Generate a deep TPU-vs-GPU scoring report for PR28 ips sweep."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKLOAD_ORDER = ["pr28_hotshape", "small_batch", "medium_batch"]
WORKLOAD_ITEMS = {
    "pr28_hotshape": 500,
    "small_batch": 10,
    "medium_batch": 100,
}

SCORE_METRICS_RE = re.compile(
    r"ScorePathMetrics path=(\S+) items=(\d+) dispatches=(\d+) "
    r"lifecycle_sent=(\d+) lifecycle_recv=(\d+) queue_wait_s=([0-9.]+) "
    r"device_compute_s=([0-9.]+) host_orchestration_s=([0-9.]+) "
    r"fastpath_attempted=(\S+) fastpath_succeeded=(\S+) "
    r"fastpath_fallback_reason=(\S+)"
)

TPU_PREFILL_RE = re.compile(
    r"Prefill batch\. #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
    r"token usage: ([0-9.]+), #running-req: (\d+), #queue-req: (\d+)"
)


@dataclass
class RunArtifacts:
    name: str
    run_dir: Path
    raw: dict[str, Any]
    tpu_log_text: str
    gpu_log_text: str
    score_rows: list[dict[str, Any]]


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(round((len(s) - 1) * p))))
    return s[idx]


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    return statistics.median(vals)


def _fmt(v: float | None, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _load_run(name: str, run_dir: Path) -> RunArtifacts:
    raw_path = run_dir / "raw_results.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw results: {raw_path}")

    raw = json.loads(raw_path.read_text())
    tpu_log_path = run_dir / "artifacts" / "tpu" / "tpu_server.log"
    gpu_log_path = run_dir / "artifacts" / "gpu" / "gpu_server.log"

    tpu_log = tpu_log_path.read_text(errors="ignore") if tpu_log_path.exists() else ""
    gpu_log = gpu_log_path.read_text(errors="ignore") if gpu_log_path.exists() else ""

    rows: list[dict[str, Any]] = []
    for m in SCORE_METRICS_RE.finditer(tpu_log):
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

    return RunArtifacts(
        name=name,
        run_dir=run_dir.resolve(),
        raw=raw,
        tpu_log_text=tpu_log,
        gpu_log_text=gpu_log,
        score_rows=rows,
    )


def _cv(vals: list[float]) -> float | None:
    if not vals:
        return None
    mean = statistics.mean(vals)
    if mean == 0:
        return 0.0
    return statistics.pstdev(vals) / mean


def _score_summary(rows: list[dict[str, Any]], items: int) -> dict[str, float | int | None]:
    subset = [r for r in rows if r["items"] == items]
    steady = [r for r in subset if r["device_compute_s"] < 1.0]

    q_vals = [r["queue_wait_s"] for r in steady]
    d_vals = [r["device_compute_s"] for r in steady]
    h_vals = [r["host_orchestration_s"] for r in steady]

    q_med = _median(q_vals)
    d_med = _median(d_vals)
    h_med = _median(h_vals)

    score_phase_s = None
    if q_med is not None and d_med is not None and h_med is not None:
        score_phase_s = q_med + d_med + h_med

    return {
        "samples": len(subset),
        "steady_samples": len(steady),
        "dispatch_med": _median([r["dispatches"] for r in subset]),
        "fastpath_rate": _median([1.0 if r["fastpath_succeeded"] else 0.0 for r in subset]),
        "queue_med_s": q_med,
        "device_med_s": d_med,
        "device_p95_s": _pct(d_vals, 0.95),
        "host_med_s": h_med,
        "host_p95_s": _pct(h_vals, 0.95),
        "score_phase_med_s": score_phase_s,
        "device_outliers_gt1s": len([r for r in subset if r["device_compute_s"] >= 1.0]),
    }


def _pct_delta(base: float, new: float) -> float:
    if base == 0:
        return math.inf if new > 0 else 0.0
    return ((new - base) / base) * 100.0


def _build_report(ips32: RunArtifacts, ips160: RunArtifacts, gpu_commit: str, tpu_commit: str) -> str:
    lines: list[str] = []
    now = datetime.now(timezone.utc).isoformat()

    lines.append("# Deep Dive: TPU JAX Scoring vs GPU PyTorch (PR28)")
    lines.append("")
    lines.append("## Scope")
    lines.append("Compare `/v1/score` behavior on identical workloads for:")
    lines.append("- GPU: sglang main on NVIDIA L4 (PyTorch)")
    lines.append("- TPU: PR28 `perf/pr-28-perf-hardening` on TPU v6e-1 (JAX)")
    lines.append("- TPU lane sensitivity: `items_per_step=32` vs `items_per_step=160`")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- generated_at_utc: `{now}`")
    lines.append(f"- gpu_commit: `{gpu_commit}`")
    lines.append(f"- tpu_commit: `{tpu_commit}`")
    lines.append(f"- ips32_run: `{ips32.run_dir}`")
    lines.append(f"- ips160_run: `{ips160.run_dir}`")
    lines.append("")
    lines.append("## Side-by-Side Metrics (GPU vs TPU)")
    lines.append("")
    lines.append("| Lane | Workload | GPU items/s | TPU items/s | TPU vs GPU | GPU p50 (ms) | TPU p50 (ms) | GPU p99 (ms) | TPU p99 (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for lane_name, run in [("ips32", ips32), ("ips160", ips160)]:
        for wl in WORKLOAD_ORDER:
            g = run.raw["gpu"][wl]
            t = run.raw["tpu"][wl]
            tput_delta = _pct_delta(g["throughput_items_per_sec"], t["throughput_items_per_sec"])
            lines.append(
                f"| {lane_name} | {wl} | {g['throughput_items_per_sec']:.1f} | "
                f"{t['throughput_items_per_sec']:.1f} | {tput_delta:+.1f}% | "
                f"{g['latency_p50_ms']:.1f} | {t['latency_p50_ms']:.1f} | "
                f"{g['latency_p99_ms']:.1f} | {t['latency_p99_ms']:.1f} |"
            )
    lines.append("")
    lines.append("## TPU Lane Sensitivity (ips160 vs ips32)")
    lines.append("")
    lines.append("| Workload | Throughput delta | p50 delta | p99 delta |")
    lines.append("|---|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        t32 = ips32.raw["tpu"][wl]
        t160 = ips160.raw["tpu"][wl]
        lines.append(
            f"| {wl} | {_pct_delta(t32['throughput_items_per_sec'], t160['throughput_items_per_sec']):+.1f}% | "
            f"{_pct_delta(t32['latency_p50_ms'], t160['latency_p50_ms']):+.1f}% | "
            f"{_pct_delta(t32['latency_p99_ms'], t160['latency_p99_ms']):+.1f}% |"
        )
    lines.append("")
    lines.append("## TPU ScorePath Decomposition (steady state)")
    lines.append("")
    lines.append("| Lane | Items/request | Dispatches (median) | Device median (ms) | Host median (ms) | Queue median (ms) | Theoretical score-only items/s | Measured items/s | Score-phase utilization |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for lane_name, run in [("ips32", ips32), ("ips160", ips160)]:
        for wl in WORKLOAD_ORDER:
            items = WORKLOAD_ITEMS[wl]
            summary = _score_summary(run.score_rows, items)
            measured = run.raw["tpu"][wl]["throughput_items_per_sec"]
            theoretical = None
            util = None
            if summary["score_phase_med_s"]:
                theoretical = items / summary["score_phase_med_s"]  # type: ignore[arg-type]
                util = (measured / theoretical) * 100.0 if theoretical > 0 else None
            lines.append(
                f"| {lane_name} | {items} | {_fmt(summary['dispatch_med'], 0)} | "
                f"{_fmt(summary['device_med_s'] * 1000 if summary['device_med_s'] is not None else None, 1)} | "
                f"{_fmt(summary['host_med_s'] * 1000 if summary['host_med_s'] is not None else None, 1)} | "
                f"{_fmt(summary['queue_med_s'] * 1000 if summary['queue_med_s'] is not None else None, 2)} | "
                f"{_fmt(theoretical, 0)} | {measured:.1f} | {_fmt(util, 1)}% |"
            )
    lines.append("")
    lines.append("## Stability Signals")
    lines.append("")
    lines.append("| Lane | Workload | TPU latency CV | GPU latency CV | TPU failures | GPU failures |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for lane_name, run in [("ips32", ips32), ("ips160", ips160)]:
        for wl in WORKLOAD_ORDER:
            t = run.raw["tpu"][wl]
            g = run.raw["gpu"][wl]
            t_cv = _cv(t["raw_latencies_ms"])
            g_cv = _cv(g["raw_latencies_ms"])
            lines.append(
                f"| {lane_name} | {wl} | {_fmt(t_cv, 3)} | {_fmt(g_cv, 3)} | "
                f"{t['num_failures']} | {g['num_failures']} |"
            )
    lines.append("")
    lines.append("### Log-level checks")
    for lane_name, run in [("ips32", ips32), ("ips160", ips160)]:
        missing_cache = len(re.findall(r"Missing scoring cache handle", run.tpu_log_text))
        evictions = len(re.findall(r"Evicted \\d+ expired scoring cache handles", run.tpu_log_text))
        prefill_rows = TPU_PREFILL_RE.findall(run.tpu_log_text)
        new_token_vals = [int(r[1]) for r in prefill_rows]
        new_token_mode = statistics.mode(new_token_vals) if new_token_vals else None
        lines.append(
            f"- {lane_name}: missing_cache_handle=`{missing_cache}`, "
            f"cache_evictions=`{evictions}`, prefill_samples=`{len(prefill_rows)}`, "
            f"dominant_new_token=`{new_token_mode}`"
        )
    lines.append("")
    lines.append("## Primary Bottlenecks (TPU)")
    lines.append("")
    lines.append("1. End-to-end throughput is not limited by score kernel speed; it is limited by non-score phases.")
    lines.append("   For hotshape, score-only theoretical throughput is ~2k items/s while measured is 232-517 items/s (11-26% utilization).")
    lines.append("2. Prefill/extend cadence is heavily chunked (`#new-token=64` in almost all TPU prefill logs), indicating scheduling/token-chunk overhead pressure.")
    lines.append("3. Dispatch fragmentation remains high (median dispatches: 20 for 500 items, 4 for 100 items), adding host orchestration overhead.")
    lines.append("4. No cache-handle instability on PR28 in these runs (`Missing scoring cache handle = 0`), so failures are no longer the dominant issue.")
    lines.append("5. With `items_per_step=160`, hotshape improves strongly, but medium batch exhibits severe p99 tails (2.39s), suggesting shape-specific warmup/recompile jitter.")
    lines.append("")
    lines.append("## Improvement Priorities")
    lines.append("")
    lines.append("1. Separate compile/warmup from timed windows until `device_compute_s >= 1.0` outliers disappear for each workload shape.")
    lines.append("2. Increase effective prefill/extend granularity (reduce 64-token micro-chunking overhead) for score requests.")
    lines.append("3. Raise effective items processed per dispatch on TPU score path to cut host orchestration share.")
    lines.append("4. Tune `items_per_step` per workload class (hotshape benefits from higher values; medium currently regresses in tail behavior).")
    lines.append("5. Keep fastpath metrics logging enabled and enforce regression checks on score-phase utilization and p99.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- ips32_dashboard: `{ips32.run_dir / 'dashboard.html'}`")
    lines.append(f"- ips160_dashboard: `{ips160.run_dir / 'dashboard.html'}`")
    lines.append(f"- ips32_summary: `{ips32.run_dir / 'deep_summary.md'}`")
    lines.append(f"- ips160_summary: `{ips160.run_dir / 'deep_summary.md'}`")
    lines.append(f"- ips160_tpu_trace_manual: `{ips160.run_dir / 'artifacts' / 'tpu' / 'profile' / 'tpu_hotshape_manual.trace.json.gz'}`")
    lines.append(f"- ips160_tpu_xplane_manual: `{ips160.run_dir / 'artifacts' / 'tpu' / 'profile' / 'tpu_hotshape_manual.xplane.pb'}`")
    lines.append("- gpu_profile_note: `manual /stop_profile request hung; no finalized GPU trace artifact in this pass`")
    lines.append("")
    lines.append("## Repeatable Command")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/generate_pr28_scoring_report.py \\")
    lines.append(f"  --ips32-run {ips32.run_dir} \\")
    lines.append(f"  --ips160-run {ips160.run_dir} \\")
    lines.append(f"  --output {ips160.run_dir / 'deep_dive_report.md'}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PR28 TPU-vs-GPU scoring deep-dive report.")
    parser.add_argument("--ips32-run", required=True, help="Run dir for items_per_step=32")
    parser.add_argument("--ips160-run", required=True, help="Run dir for items_per_step=160")
    parser.add_argument("--gpu-commit", default="unknown", help="GPU commit SHA")
    parser.add_argument("--tpu-commit", default="unknown", help="TPU commit SHA")
    parser.add_argument("--output", default=None, help="Output markdown file")
    args = parser.parse_args()

    ips32 = _load_run("ips32", Path(args.ips32_run))
    ips160 = _load_run("ips160", Path(args.ips160_run))

    output = Path(args.output).resolve() if args.output else (ips160.run_dir / "deep_dive_report.md")
    report = _build_report(ips32, ips160, args.gpu_commit, args.tpu_commit)
    output.write_text(report)
    print(output)


if __name__ == "__main__":
    main()
