#!/usr/bin/env python3
"""Generate a repeatable TPU soak vs GPU baseline deep-dive report."""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKLOAD_ORDER = ["pr28_hotshape", "small_batch", "medium_batch"]

SCORE_METRICS_RE = re.compile(
    r"ScorePathMetrics path=(\S+) items=(\d+) dispatches=(\d+) .*? "
    r"queue_wait_s=([0-9.]+) device_compute_s=([0-9.]+) host_orchestration_s=([0-9.]+)"
)
PREFILL_RE = re.compile(
    r"Prefill batch\. #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
    r"token usage: ([0-9.]+), #running-req: (\d+), #queue-req: (\d+)"
)


@dataclass
class WorkloadStats:
    repeats_total: int
    repeats_success: int
    repeats_failed: int
    success_rate: float
    throughput_success_median: float | None
    throughput_success_mean: float | None
    throughput_all_mean: float | None
    p50_success_median_ms: float | None
    p99_success_median_ms: float | None
    p99_p50_success_ratio: float | None
    throughput_success_cv: float | None
    p99_success_cv: float | None
    top_error_counts: list[tuple[str, int]]


def _median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def _mean(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def _cv(vals: list[float]) -> float | None:
    if not vals:
        return None
    m = statistics.mean(vals)
    if m == 0:
        return 0.0
    return statistics.pstdev(vals) / m


def _pct_delta(base: float | None, new: float | None) -> float | None:
    if base is None or new is None:
        return None
    if base == 0:
        return None
    return ((new - base) / base) * 100.0


def _fmt(v: float | None, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_pct(v: float | None, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}%"


def _resolve_gpu_run_paths(
    matrix_group_dir: Path,
    raw_paths: list[str],
) -> list[Path]:
    if raw_paths:
        return [Path(p).resolve() for p in raw_paths]

    # Discover ".../results" from ancestors so this works for nested tpu_tuning paths.
    results_root = None
    for anc in [matrix_group_dir, *matrix_group_dir.parents]:
        if anc.name == "results":
            results_root = anc
            break
    if results_root is None:
        raise FileNotFoundError(
            "Could not locate results root for auto GPU baseline discovery. Pass --gpu-run explicitly."
        )

    candidates = sorted(results_root.glob("*_pr28-vs-main-l4-v6e1-ips*/raw_results.json"))
    if not candidates:
        raise FileNotFoundError(
            "No GPU baseline runs found automatically. Pass --gpu-run explicitly."
        )
    # Keep latest two runs to reduce one-off noise if available.
    return [p.parent.resolve() for p in candidates[-2:]]


def _load_gpu_baseline(gpu_run_dirs: list[Path]) -> dict[str, dict[str, float | int | None]]:
    by_workload: dict[str, dict[str, list[float]]] = {
        wl: {"throughput": [], "p50": [], "p99": []} for wl in WORKLOAD_ORDER
    }
    run_count = 0
    for run_dir in gpu_run_dirs:
        raw_path = run_dir / "raw_results.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing GPU raw_results: {raw_path}")
        raw = json.loads(raw_path.read_text())
        gpu = raw.get("gpu", {})
        run_count += 1
        for wl in WORKLOAD_ORDER:
            m = gpu.get(wl, {})
            by_workload[wl]["throughput"].append(float(m.get("throughput_items_per_sec", 0.0)))
            by_workload[wl]["p50"].append(float(m.get("latency_p50_ms", 0.0)))
            by_workload[wl]["p99"].append(float(m.get("latency_p99_ms", 0.0)))

    out: dict[str, dict[str, float | int | None]] = {}
    for wl in WORKLOAD_ORDER:
        t_med = _median(by_workload[wl]["throughput"])
        p50_med = _median(by_workload[wl]["p50"])
        p99_med = _median(by_workload[wl]["p99"])
        out[wl] = {
            "runs": run_count,
            "throughput_median_items_per_sec": t_med,
            "latency_p50_median_ms": p50_med,
            "latency_p99_median_ms": p99_med,
            "p99_p50_ratio_median": (p99_med / p50_med) if (p50_med and p50_med > 0) else None,
        }
    return out


def _load_matrix_runs(matrix_summary: dict[str, Any]) -> list[dict[str, Any]]:
    runs = matrix_summary.get("runs", [])
    if not runs:
        raise ValueError("matrix_summary.json does not contain runs.")
    return runs


def _infer_items_by_workload(runs: list[dict[str, Any]]) -> dict[str, int]:
    items_map: dict[str, int] = {}
    for run in runs:
        run_dir = Path(run["run_dir"]).resolve()
        log_path = run_dir / "artifacts" / "tpu" / "tpu_server.log"
        if not log_path.exists():
            continue
        text = log_path.read_text(errors="ignore")
        for m in SCORE_METRICS_RE.finditer(text):
            items = int(m.group(2))
            # Resolve by nearest known workload cardinality if unique.
            if items in (500, 10, 100):
                if items == 500:
                    items_map["pr28_hotshape"] = items
                elif items == 10:
                    items_map["small_batch"] = items
                elif items == 100:
                    items_map["medium_batch"] = items
        if len(items_map) == len(WORKLOAD_ORDER):
            break
    # Fallback defaults.
    items_map.setdefault("pr28_hotshape", 500)
    items_map.setdefault("small_batch", 10)
    items_map.setdefault("medium_batch", 100)
    return items_map


def _aggregate_tpu_stats(
    runs: list[dict[str, Any]],
    items_by_workload: dict[str, int],
) -> tuple[dict[str, WorkloadStats], dict[str, dict[str, float | int | None]], dict[str, Any]]:
    wl_vals: dict[str, dict[str, list[float] | int]] = {
        wl: {
            "throughput_all": [],
            "throughput_success": [],
            "p50_success": [],
            "p99_success": [],
            "repeats_total": 0,
            "repeats_success": 0,
            "error_counts": {},
        }
        for wl in WORKLOAD_ORDER
    }
    phase_rows: dict[str, list[dict[str, float]]] = {wl: [] for wl in WORKLOAD_ORDER}
    missing_cache_total = 0
    prefill_vals: list[int] = []
    score_samples_total = 0

    for run in runs:
        run_dir = Path(run["run_dir"]).resolve()
        log_path = run_dir / "artifacts" / "tpu" / "tpu_server.log"
        log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
        missing_cache_total += len(re.findall(r"Missing scoring cache handle", log_text))
        prefill_vals.extend([int(m.group(2)) for m in PREFILL_RE.finditer(log_text)])

        parsed_rows: list[dict[str, float]] = []
        for m in SCORE_METRICS_RE.finditer(log_text):
            parsed_rows.append(
                {
                    "items": float(int(m.group(2))),
                    "dispatches": float(int(m.group(3))),
                    "queue_wait_s": float(m.group(4)),
                    "device_compute_s": float(m.group(5)),
                    "host_orch_s": float(m.group(6)),
                }
            )
        score_samples_total += len(parsed_rows)

        workloads = run.get("workloads", {})
        for wl in WORKLOAD_ORDER:
            m = workloads.get(wl, {})
            tput = float(m.get("throughput_items_per_sec", 0.0))
            p50 = float(m.get("latency_p50_ms", 0.0))
            p99 = float(m.get("latency_p99_ms", 0.0))
            fails = int(m.get("num_failures", 0))
            raw_error_counts = m.get("error_counts", {}) or {}
            agg_error_counts = wl_vals[wl]["error_counts"]
            assert isinstance(agg_error_counts, dict)
            for err, cnt in raw_error_counts.items():
                key = str(err)
                agg_error_counts[key] = int(agg_error_counts.get(key, 0)) + int(cnt)

            wl_vals[wl]["repeats_total"] = int(wl_vals[wl]["repeats_total"]) + 1
            cast_throughput_all = wl_vals[wl]["throughput_all"]
            assert isinstance(cast_throughput_all, list)
            cast_throughput_all.append(tput)

            is_success = fails == 0 and tput > 0 and p50 > 0 and p99 > 0
            if is_success:
                wl_vals[wl]["repeats_success"] = int(wl_vals[wl]["repeats_success"]) + 1
                cast_tput_success = wl_vals[wl]["throughput_success"]
                cast_p50_success = wl_vals[wl]["p50_success"]
                cast_p99_success = wl_vals[wl]["p99_success"]
                assert isinstance(cast_tput_success, list)
                assert isinstance(cast_p50_success, list)
                assert isinstance(cast_p99_success, list)
                cast_tput_success.append(tput)
                cast_p50_success.append(p50)
                cast_p99_success.append(p99)

                items = float(items_by_workload[wl])
                subset = [r for r in parsed_rows if int(r["items"]) == int(items)]
                phase_rows[wl].extend(subset)

    wl_stats: dict[str, WorkloadStats] = {}
    phase_summary: dict[str, dict[str, float | int | None]] = {}
    for wl in WORKLOAD_ORDER:
        total = int(wl_vals[wl]["repeats_total"])
        success = int(wl_vals[wl]["repeats_success"])
        failed = total - success

        throughput_all = [float(v) for v in wl_vals[wl]["throughput_all"]]  # type: ignore[list-item]
        throughput_success = [float(v) for v in wl_vals[wl]["throughput_success"]]  # type: ignore[list-item]
        p50_success = [float(v) for v in wl_vals[wl]["p50_success"]]  # type: ignore[list-item]
        p99_success = [float(v) for v in wl_vals[wl]["p99_success"]]  # type: ignore[list-item]
        error_counts = {
            str(k): int(v)
            for k, v in (wl_vals[wl]["error_counts"]).items()  # type: ignore[union-attr]
        }
        top_error_counts = sorted(
            error_counts.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]

        t_success_med = _median(throughput_success)
        p50_success_med = _median(p50_success)
        p99_success_med = _median(p99_success)

        wl_stats[wl] = WorkloadStats(
            repeats_total=total,
            repeats_success=success,
            repeats_failed=failed,
            success_rate=(success / total) if total > 0 else 0.0,
            throughput_success_median=t_success_med,
            throughput_success_mean=_mean(throughput_success),
            throughput_all_mean=_mean(throughput_all),
            p50_success_median_ms=p50_success_med,
            p99_success_median_ms=p99_success_med,
            p99_p50_success_ratio=(p99_success_med / p50_success_med)
            if (p50_success_med and p50_success_med > 0)
            else None,
            throughput_success_cv=_cv(throughput_success),
            p99_success_cv=_cv(p99_success),
            top_error_counts=top_error_counts,
        )

        rows = phase_rows[wl]
        steady = [r for r in rows if float(r["device_compute_s"]) < 1.0]
        queue_med = _median([float(r["queue_wait_s"]) for r in steady])
        device_med = _median([float(r["device_compute_s"]) for r in steady])
        host_med = _median([float(r["host_orch_s"]) for r in steady])
        dispatch_med = _median([float(r["dispatches"]) for r in rows])
        score_phase_s = (
            (queue_med + device_med + host_med)
            if queue_med is not None and device_med is not None and host_med is not None
            else None
        )
        theoretical = (
            (items_by_workload[wl] / score_phase_s) if (score_phase_s is not None and score_phase_s > 0) else None
        )
        util = (
            (t_success_med / theoretical) * 100.0
            if (theoretical is not None and theoretical > 0 and t_success_med is not None)
            else None
        )
        phase_summary[wl] = {
            "samples": len(rows),
            "steady_samples": len(steady),
            "dispatches_median": dispatch_med,
            "queue_wait_median_ms": queue_med * 1000.0 if queue_med is not None else None,
            "device_compute_median_ms": device_med * 1000.0 if device_med is not None else None,
            "host_orchestration_median_ms": host_med * 1000.0 if host_med is not None else None,
            "score_phase_median_s": score_phase_s,
            "theoretical_items_per_sec": theoretical,
            "utilization_pct": util,
        }

    dominant_new_token = None
    if prefill_vals:
        counts: dict[int, int] = {}
        for v in prefill_vals:
            counts[v] = counts.get(v, 0) + 1
        dominant_new_token = max(sorted(counts.keys()), key=lambda k: counts[k])

    signals = {
        "missing_cache_handle_total": missing_cache_total,
        "prefill_samples_total": len(prefill_vals),
        "dominant_new_token": dominant_new_token,
        "score_samples_total": score_samples_total,
    }
    return wl_stats, phase_summary, signals


def _build_summary(
    matrix_group_dir: Path,
    matrix_summary: dict[str, Any],
    gpu_baseline: dict[str, dict[str, float | int | None]],
    tpu_stats: dict[str, WorkloadStats],
    tpu_phase: dict[str, dict[str, float | int | None]],
    tpu_signals: dict[str, Any],
    gpu_run_dirs: list[Path],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "matrix_group_dir": str(matrix_group_dir),
        "matrix_recommended_items_per_step": matrix_summary.get("recommended", {}).get("items_per_step"),
        "gpu_run_dirs": [str(p) for p in gpu_run_dirs],
        "gpu_baseline": gpu_baseline,
        "tpu": {
            "workloads": {},
            "score_phase": tpu_phase,
            "signals": tpu_signals,
        },
    }

    for wl in WORKLOAD_ORDER:
        s = tpu_stats[wl]
        g = gpu_baseline[wl]
        top_error_counts = getattr(s, "top_error_counts", [])
        summary["tpu"]["workloads"][wl] = {
            "repeats_total": s.repeats_total,
            "repeats_success": s.repeats_success,
            "repeats_failed": s.repeats_failed,
            "success_rate": s.success_rate,
            "throughput_success_median_items_per_sec": s.throughput_success_median,
            "throughput_success_mean_items_per_sec": s.throughput_success_mean,
            "throughput_all_mean_items_per_sec": s.throughput_all_mean,
            "latency_p50_success_median_ms": s.p50_success_median_ms,
            "latency_p99_success_median_ms": s.p99_success_median_ms,
            "p99_p50_success_ratio": s.p99_p50_success_ratio,
            "throughput_success_cv": s.throughput_success_cv,
            "latency_p99_success_cv": s.p99_success_cv,
            "delta_vs_gpu_success_throughput_pct": _pct_delta(
                float(g["throughput_median_items_per_sec"]) if g["throughput_median_items_per_sec"] is not None else None,
                s.throughput_success_median,
            ),
            "delta_vs_gpu_success_p50_pct": _pct_delta(
                float(g["latency_p50_median_ms"]) if g["latency_p50_median_ms"] is not None else None,
                s.p50_success_median_ms,
            ),
            "delta_vs_gpu_success_p99_pct": _pct_delta(
                float(g["latency_p99_median_ms"]) if g["latency_p99_median_ms"] is not None else None,
                s.p99_success_median_ms,
            ),
            "top_error_counts": top_error_counts,
        }
    return summary


def _render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Deep Dive: TPU Soak vs GPU Baseline")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- generated_at_utc: `{summary['generated_at_utc']}`")
    lines.append(f"- matrix_group_dir: `{summary['matrix_group_dir']}`")
    lines.append(
        f"- matrix_recommended_items_per_step: `{summary['matrix_recommended_items_per_step']}`"
    )
    lines.append(f"- gpu_baseline_runs: `{len(summary['gpu_run_dirs'])}`")
    for p in summary["gpu_run_dirs"]:
        lines.append(f"  - `{p}`")
    lines.append("")

    lines.append("## Side-by-Side Key Metrics (GPU vs TPU)")
    lines.append("")
    lines.append(
        "| Workload | GPU tput (items/s) | TPU tput (success median) | TPU tput (all-repeat mean) | TPU vs GPU (success) | GPU p50 (ms) | TPU p50 (ms) | GPU p99 (ms) | TPU p99 (ms) | TPU success rate | Top failure reason |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for wl in WORKLOAD_ORDER:
        g = summary["gpu_baseline"][wl]
        t = summary["tpu"]["workloads"][wl]
        top_err = "none"
        if t.get("top_error_counts"):
            err, cnt = t["top_error_counts"][0]
            top_err = f"{cnt}x {str(err)[:100]}"
        lines.append(
            f"| {wl} | {_fmt(g['throughput_median_items_per_sec'], 1)} | "
            f"{_fmt(t['throughput_success_median_items_per_sec'], 1)} | "
            f"{_fmt(t['throughput_all_mean_items_per_sec'], 1)} | "
            f"{_fmt_pct(t['delta_vs_gpu_success_throughput_pct'], 1)} | "
            f"{_fmt(g['latency_p50_median_ms'], 1)} | {_fmt(t['latency_p50_success_median_ms'], 1)} | "
            f"{_fmt(g['latency_p99_median_ms'], 1)} | {_fmt(t['latency_p99_success_median_ms'], 1)} | "
            f"{_fmt(t['success_rate'] * 100.0, 1)}% | {top_err} |"
        )
    lines.append("")

    lines.append("## TPU Stability (Successful Repeats)")
    lines.append("")
    lines.append(
        "| Workload | Repeats (success/total) | Throughput CV | p99 CV | p99/p50 ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        t = summary["tpu"]["workloads"][wl]
        lines.append(
            f"| {wl} | {t['repeats_success']}/{t['repeats_total']} | "
            f"{_fmt(t['throughput_success_cv'], 3)} | {_fmt(t['latency_p99_success_cv'], 3)} | "
            f"{_fmt(t['p99_p50_success_ratio'], 2)} |"
        )
    lines.append("")

    lines.append("## TPU ScorePath Decomposition")
    lines.append("")
    lines.append(
        "| Workload | Samples | Dispatches med | Queue med (ms) | Device med (ms) | Host med (ms) | Theoretical score-only (items/s) | Realized tput (items/s) | Score utilization |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for wl in WORKLOAD_ORDER:
        p = summary["tpu"]["score_phase"][wl]
        t = summary["tpu"]["workloads"][wl]
        lines.append(
            f"| {wl} | {p['steady_samples']}/{p['samples']} | {_fmt(p['dispatches_median'], 1)} | "
            f"{_fmt(p['queue_wait_median_ms'], 3)} | {_fmt(p['device_compute_median_ms'], 2)} | "
            f"{_fmt(p['host_orchestration_median_ms'], 2)} | {_fmt(p['theoretical_items_per_sec'], 1)} | "
            f"{_fmt(t['throughput_success_median_items_per_sec'], 1)} | {_fmt(p['utilization_pct'], 1)}% |"
        )
    lines.append("")

    lines.append("## TPU Log Signals")
    lines.append("")
    sig = summary["tpu"]["signals"]
    lines.append(f"- missing_scoring_cache_handle_total: `{sig['missing_cache_handle_total']}`")
    lines.append(f"- score_samples_total: `{sig['score_samples_total']}`")
    lines.append(f"- prefill_samples_total: `{sig['prefill_samples_total']}`")
    lines.append(f"- dominant_new_token: `{sig['dominant_new_token']}`")
    lines.append("")

    lines.append("## Bottleneck Readout")
    lines.append("")
    for wl in WORKLOAD_ORDER:
        p = summary["tpu"]["score_phase"][wl]
        t = summary["tpu"]["workloads"][wl]
        util = p["utilization_pct"]
        success_rate = float(t["success_rate"])
        lines.append(
            f"- {wl}: utilization={_fmt(util, 1)}%, success_rate={_fmt(success_rate * 100.0, 1)}%, "
            f"dispatches_med={_fmt(p['dispatches_median'], 1)}, "
            f"host_med={_fmt(p['host_orchestration_median_ms'], 2)}ms, "
            f"device_med={_fmt(p['device_compute_median_ms'], 2)}ms."
        )
    lines.append("")
    lines.append(
        "- Interpretation: score kernel compute is not the dominant limit when utilization is low; "
        "effective throughput is constrained by scheduling/chunking/host orchestration and repeat reliability."
    )
    lines.append("")
    return "\n".join(lines)


def _table_row(cols: list[str]) -> str:
    esc = [html.escape(c) for c in cols]
    return "<tr>" + "".join(f"<td>{c}</td>" for c in esc) + "</tr>"


def _render_static_html(summary: dict[str, Any]) -> str:
    rows_key = []
    rows_stability = []
    rows_phase = []
    for wl in WORKLOAD_ORDER:
        g = summary["gpu_baseline"][wl]
        t = summary["tpu"]["workloads"][wl]
        p = summary["tpu"]["score_phase"][wl]
        top_err = "none"
        if t.get("top_error_counts"):
            err, cnt = t["top_error_counts"][0]
            top_err = f"{cnt}x {str(err)[:90]}"
        rows_key.append(
            _table_row(
                [
                    wl,
                    _fmt(g["throughput_median_items_per_sec"], 1),
                    _fmt(t["throughput_success_median_items_per_sec"], 1),
                    _fmt(t["throughput_all_mean_items_per_sec"], 1),
                    _fmt_pct(t["delta_vs_gpu_success_throughput_pct"], 1),
                    _fmt(g["latency_p50_median_ms"], 1),
                    _fmt(t["latency_p50_success_median_ms"], 1),
                    _fmt(g["latency_p99_median_ms"], 1),
                    _fmt(t["latency_p99_success_median_ms"], 1),
                    _fmt(float(t["success_rate"]) * 100.0, 1) + "%",
                    top_err,
                ]
            )
        )
        rows_stability.append(
            _table_row(
                [
                    wl,
                    f"{t['repeats_success']}/{t['repeats_total']}",
                    _fmt(t["throughput_success_cv"], 3),
                    _fmt(t["latency_p99_success_cv"], 3),
                    _fmt(t["p99_p50_success_ratio"], 2),
                ]
            )
        )
        rows_phase.append(
            _table_row(
                [
                    wl,
                    f"{p['steady_samples']}/{p['samples']}",
                    _fmt(p["dispatches_median"], 1),
                    _fmt(p["queue_wait_median_ms"], 3),
                    _fmt(p["device_compute_median_ms"], 2),
                    _fmt(p["host_orchestration_median_ms"], 2),
                    _fmt(p["theoretical_items_per_sec"], 1),
                    _fmt(t["throughput_success_median_items_per_sec"], 1),
                    _fmt(p["utilization_pct"], 1) + "%",
                ]
            )
        )

    sig = summary["tpu"]["signals"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TPU Soak vs GPU Baseline</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121a30;
      --text: #e7ecff;
      --muted: #9cb0df;
      --line: #2a3b69;
      --accent: #4fd1c5;
      --warn: #f6ad55;
    }}
    body {{
      margin: 0;
      background: radial-gradient(1200px 700px at 10% -10%, #1f305f, var(--bg));
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 18px; margin-top: 24px; }}
    p, li {{ color: var(--muted); }}
    .panel {{
      background: linear-gradient(180deg, #162247, var(--panel));
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      margin-top: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 8px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--accent); font-weight: 600; }}
    .meta code {{
      color: var(--text);
      background: #0e1630;
      border: 1px solid var(--line);
      padding: 1px 6px;
      border-radius: 6px;
    }}
    .sig {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 8px;
    }}
    .kpi {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #0f1733;
    }}
    .kpi .k {{ color: var(--muted); font-size: 12px; }}
    .kpi .v {{ color: var(--text); font-size: 18px; margin-top: 4px; }}
    .note {{ color: var(--warn); }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>TPU Soak vs GPU Baseline</h1>
    <div class="panel meta">
      <p>Generated: <code>{html.escape(summary["generated_at_utc"])}</code></p>
      <p>Matrix group: <code>{html.escape(summary["matrix_group_dir"])}</code></p>
      <p>Recommended items_per_step from matrix: <code>{html.escape(str(summary["matrix_recommended_items_per_step"]))}</code></p>
      <p>GPU baseline runs: <code>{len(summary["gpu_run_dirs"])}</code></p>
    </div>

    <h2>Side-by-Side Key Metrics</h2>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>Workload</th><th>GPU tput</th><th>TPU tput (success med)</th><th>TPU tput (all mean)</th>
            <th>TPU vs GPU</th><th>GPU p50</th><th>TPU p50</th><th>GPU p99</th><th>TPU p99</th><th>TPU success</th><th>Top failure</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_key)}
        </tbody>
      </table>
    </div>

    <h2>TPU Stability (Successful Repeats)</h2>
    <div class="panel">
      <table>
        <thead>
          <tr><th>Workload</th><th>Success/Total</th><th>Throughput CV</th><th>p99 CV</th><th>p99/p50</th></tr>
        </thead>
        <tbody>
          {''.join(rows_stability)}
        </tbody>
      </table>
    </div>

    <h2>TPU ScorePath Decomposition</h2>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>Workload</th><th>Steady/Samples</th><th>Dispatches med</th><th>Queue med (ms)</th>
            <th>Device med (ms)</th><th>Host med (ms)</th><th>Theoretical (items/s)</th>
            <th>Realized (items/s)</th><th>Utilization</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_phase)}
        </tbody>
      </table>
    </div>

    <h2>TPU Log Signals</h2>
    <div class="panel sig">
      <div class="kpi"><div class="k">Missing scoring cache handle</div><div class="v">{sig['missing_cache_handle_total']}</div></div>
      <div class="kpi"><div class="k">ScorePath samples</div><div class="v">{sig['score_samples_total']}</div></div>
      <div class="kpi"><div class="k">Prefill samples</div><div class="v">{sig['prefill_samples_total']}</div></div>
      <div class="kpi"><div class="k">Dominant #new-token</div><div class="v">{sig['dominant_new_token']}</div></div>
    </div>

    <p class="note">Interpretation: low score-phase utilization indicates throughput is mostly constrained by orchestration/chunking/reliability, not raw score kernel speed.</p>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deep TPU soak vs GPU baseline report."
    )
    parser.add_argument(
        "--matrix-group-dir",
        required=True,
        help="Path to matrix group directory containing matrix_summary.json and runs/",
    )
    parser.add_argument(
        "--gpu-run",
        action="append",
        default=[],
        help="Path to prior run directory containing raw_results.json (repeatable flag).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <matrix-group-dir>/soak_vs_gpu_report)",
    )
    args = parser.parse_args()

    matrix_group_dir = Path(args.matrix_group_dir).resolve()
    matrix_summary_path = matrix_group_dir / "matrix_summary.json"
    if not matrix_summary_path.exists():
        raise FileNotFoundError(f"Missing: {matrix_summary_path}")

    matrix_summary = json.loads(matrix_summary_path.read_text())
    runs = _load_matrix_runs(matrix_summary)
    items_by_workload = _infer_items_by_workload(runs)
    gpu_run_dirs = _resolve_gpu_run_paths(matrix_group_dir, args.gpu_run)
    gpu_baseline = _load_gpu_baseline(gpu_run_dirs)
    tpu_stats, tpu_phase, tpu_signals = _aggregate_tpu_stats(runs, items_by_workload)
    summary = _build_summary(
        matrix_group_dir=matrix_group_dir,
        matrix_summary=matrix_summary,
        gpu_baseline=gpu_baseline,
        tpu_stats=tpu_stats,
        tpu_phase=tpu_phase,
        tpu_signals=tpu_signals,
        gpu_run_dirs=gpu_run_dirs,
    )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (matrix_group_dir / "soak_vs_gpu_report").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "summary.json"
    markdown_path = output_dir / "deep_dive_soak_vs_gpu.md"
    html_path = output_dir / "dashboard_static.html"

    summary_json.write_text(json.dumps(summary, indent=2))
    markdown_path.write_text(_render_markdown(summary))
    html_path.write_text(_render_static_html(summary))

    print(f"summary_json={summary_json}")
    print(f"markdown={markdown_path}")
    print(f"html={html_path}")


if __name__ == "__main__":
    main()
