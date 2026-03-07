"""Console table output and dashboard JSON/HTML generation."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from profiler.analyzer import ComparisonResult, WorkloadComparison, MetricDelta
from profiler.config import ProfilerConfig
from profiler.metrics import WorkloadMetrics


# ──────────────────────────────────────────────────────────────────────
# Console report
# ──────────────────────────────────────────────────────────────────────

def print_console_report(comparison: ComparisonResult, config: ProfilerConfig) -> None:
    """Print a formatted comparison table to stdout."""
    keys = config.backend_keys
    gpu_cfg = config.get_backend(keys[0])
    tpu_cfg = config.get_backend(keys[1])

    print(f"\n{'='*72}")
    print(f"  COMPARISON: {gpu_cfg.name} ({gpu_cfg.hardware}) vs {tpu_cfg.name} ({tpu_cfg.hardware})")
    print(f"  Model: {config.experiment.model}")
    print(f"{'='*72}")

    for wc in comparison.workload_comparisons:
        _print_workload_table(wc, gpu_cfg.hardware, tpu_cfg.hardware)

    anomalies = comparison.all_anomalies
    if anomalies:
        print(f"\n{'─'*72}")
        print("  ANOMALIES")
        print(f"{'─'*72}")
        for a in anomalies:
            sev = a.severity.upper().ljust(6)
            print(f"  [{sev}] {a.backend}: {a.finding}")
            print(f"           → {a.suggestion}")


def _print_workload_table(wc: WorkloadComparison, gpu_hw: str, tpu_hw: str) -> None:
    """Print one workload comparison table."""
    print(f"\n{'─'*72}")
    print(f"  Workload: {wc.workload_name} | {gpu_hw} vs {tpu_hw}")
    print(f"{'─'*72}")

    header = f"  {'Metric':<28} | {'GPU':>14} | {'TPU':>14} | {'Delta':>12}"
    print(header)
    print(f"  {'-'*28}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")

    for d in wc.deltas:
        label = _metric_label(d.metric_name)
        gpu_str = _format_value(d.metric_name, d.gpu_value)
        tpu_str = _format_value(d.metric_name, d.tpu_value)
        delta_str = _format_delta(d)
        print(f"  {label:<28} | {gpu_str:>14} | {tpu_str:>14} | {delta_str:>12}")


def _metric_label(name: str) -> str:
    labels = {
        "throughput_items_per_sec": "Throughput (items/s)",
        "latency_p50_ms": "Latency p50 (ms)",
        "latency_p90_ms": "Latency p90 (ms)",
        "latency_p95_ms": "Latency p95 (ms)",
        "latency_p99_ms": "Latency p99 (ms)",
        "latency_mean_ms": "Latency mean (ms)",
    }
    return labels.get(name, name)


def _format_value(metric_name: str, value: float) -> str:
    if "throughput" in metric_name:
        return f"{value:,.1f}"
    return f"{value:,.1f}"


def _format_delta(d: MetricDelta) -> str:
    sign = "+" if d.delta_pct > 0 else ""
    winner = d.winner.upper() if d.winner != "tie" else "tie"
    return f"{sign}{d.delta_pct:.1f}% {winner}"


# ──────────────────────────────────────────────────────────────────────
# Dashboard JSON — matches profiler_report.jsx data shapes
# ──────────────────────────────────────────────────────────────────────

def generate_dashboard_json(
    comparison: ComparisonResult,
    all_results: dict[str, dict[str, WorkloadMetrics]],
    config: ProfilerConfig,
    output_path: Path,
) -> None:
    """Generate dashboard_data.json matching the profiler_report.jsx schema."""
    keys = config.backend_keys
    gpu_cfg = config.get_backend(keys[0])
    tpu_cfg = config.get_backend(keys[1])

    data = {
        "meta": {
            "experiment": config.experiment.name,
            "model": config.experiment.model,
            "gpu_hardware": gpu_cfg.hardware,
            "tpu_hardware": tpu_cfg.hardware,
            "gpu_name": gpu_cfg.name,
            "tpu_name": tpu_cfg.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "headlineMetrics": _build_headline_metrics(comparison, config),
        "throughputByBatch": _build_throughput_by_batch(comparison),
        "latencyBreakdown": _build_latency_breakdown(comparison),
        "anomalies": _build_anomalies(comparison),
        "sweepResults": _build_sweep_results(all_results, config),
        "radarData": _build_radar_data(comparison, config),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _build_headline_metrics(
    comparison: ComparisonResult,
    config: ProfilerConfig,
) -> list[dict]:
    """Build headlineMetrics array matching JSX shape."""
    metrics = []

    # Aggregate across workloads: use the largest workload for headline numbers
    if not comparison.workload_comparisons:
        return metrics

    # Pick the workload with most items (the "main" workload)
    main = max(comparison.workload_comparisons, key=lambda wc: wc.gpu_metrics.total_items)
    gpu = main.gpu_metrics
    tpu = main.tpu_metrics

    keys = config.backend_keys
    gpu_cost = config.get_backend(keys[0]).cost_per_hour
    tpu_cost = config.get_backend(keys[1]).cost_per_hour

    # Throughput
    tp_delta = _pct_delta(gpu.throughput_items_per_sec, tpu.throughput_items_per_sec)
    metrics.append({
        "label": "Score Throughput",
        "pytorch": f"{gpu.throughput_items_per_sec:,.0f} items/s",
        "jax": f"{tpu.throughput_items_per_sec:,.0f} items/s",
        "winner": "jax" if tpu.throughput_items_per_sec > gpu.throughput_items_per_sec else "pytorch",
        "delta": f"{tp_delta:+.1f}%",
    })

    # Latency p50
    p50_delta = _pct_delta(gpu.latency_p50_ms, tpu.latency_p50_ms)
    metrics.append({
        "label": "Latency p50",
        "pytorch": f"{gpu.latency_p50_ms:,.1f} ms",
        "jax": f"{tpu.latency_p50_ms:,.1f} ms",
        "winner": "jax" if tpu.latency_p50_ms < gpu.latency_p50_ms else "pytorch",
        "delta": f"{p50_delta:+.1f}%",
    })

    # Latency p99
    p99_delta = _pct_delta(gpu.latency_p99_ms, tpu.latency_p99_ms)
    metrics.append({
        "label": "Latency p99",
        "pytorch": f"{gpu.latency_p99_ms:,.1f} ms",
        "jax": f"{tpu.latency_p99_ms:,.1f} ms",
        "winner": "jax" if tpu.latency_p99_ms < gpu.latency_p99_ms else "pytorch",
        "delta": f"{p99_delta:+.1f}%",
    })

    # Cost efficiency (items per dollar-hour)
    gpu_eff = gpu.throughput_items_per_sec * 3600 / gpu_cost if gpu_cost > 0 else 0
    tpu_eff = tpu.throughput_items_per_sec * 3600 / tpu_cost if tpu_cost > 0 else 0
    eff_delta = _pct_delta(gpu_eff, tpu_eff)
    metrics.append({
        "label": "Cost Efficiency",
        "pytorch": f"{gpu_eff:,.0f} items/$/hr",
        "jax": f"{tpu_eff:,.0f} items/$/hr",
        "winner": "jax" if tpu_eff > gpu_eff else "pytorch",
        "delta": f"{eff_delta:+.1f}%",
    })

    return metrics


def _build_throughput_by_batch(comparison: ComparisonResult) -> list[dict]:
    """Map workloads to throughputByBatch — uses num_items as the batch axis."""
    rows = []
    for wc in sorted(comparison.workload_comparisons, key=lambda w: w.gpu_metrics.total_items):
        rows.append({
            "batch": str(wc.gpu_metrics.total_items // max(wc.gpu_metrics.num_successes, 1)),
            "pytorch": round(wc.gpu_metrics.throughput_items_per_sec, 1),
            "jax": round(wc.tpu_metrics.throughput_items_per_sec, 1),
        })
    return rows


def _build_latency_breakdown(comparison: ComparisonResult) -> list[dict]:
    """Build overall latency breakdown (p50/p99 per workload).

    Per-stage breakdown requires profiler traces — not available in Phase 1.
    """
    rows = []
    for wc in comparison.workload_comparisons:
        rows.append({
            "stage": f"{wc.workload_name} (p50)",
            "pytorch": round(wc.gpu_metrics.latency_p50_ms, 1),
            "jax": round(wc.tpu_metrics.latency_p50_ms, 1),
        })
        rows.append({
            "stage": f"{wc.workload_name} (p99)",
            "pytorch": round(wc.gpu_metrics.latency_p99_ms, 1),
            "jax": round(wc.tpu_metrics.latency_p99_ms, 1),
        })
    return rows


def _build_anomalies(comparison: ComparisonResult) -> list[dict]:
    """Build anomalies array matching JSX shape."""
    return [
        {
            "severity": a.severity,
            "backend": a.backend,
            "finding": a.finding,
            "suggestion": a.suggestion,
        }
        for a in comparison.all_anomalies
    ]


def _build_sweep_results(
    all_results: dict[str, dict[str, WorkloadMetrics]],
    config: ProfilerConfig,
) -> list[dict]:
    """Build sweep matrix from all workload × backend combinations."""
    keys = config.backend_keys
    if len(keys) < 2:
        return []

    gpu_key, tpu_key = keys[0], keys[1]
    gpu_data = all_results.get(gpu_key, {})
    tpu_data = all_results.get(tpu_key, {})

    rows = []
    for workload in config.workloads:
        wname = workload.name
        gpu_m = gpu_data.get(wname)
        tpu_m = tpu_data.get(wname)
        if gpu_m and tpu_m:
            rows.append({
                "seqLen": workload.query_tokens,
                "bs": workload.num_items,
                "pytorchTps": round(gpu_m.throughput_items_per_sec, 1),
                "jaxTps": round(tpu_m.throughput_items_per_sec, 1),
            })
    return rows


def _build_radar_data(comparison: ComparisonResult, config: ProfilerConfig) -> list[dict]:
    """Build normalized radar chart data (0-100 scale)."""
    if not comparison.workload_comparisons:
        return []

    main = max(comparison.workload_comparisons, key=lambda wc: wc.gpu_metrics.total_items)
    gpu = main.gpu_metrics
    tpu = main.tpu_metrics

    keys = config.backend_keys
    gpu_cost = config.get_backend(keys[0]).cost_per_hour
    tpu_cost = config.get_backend(keys[1]).cost_per_hour

    def normalize(a: float, b: float) -> tuple[int, int]:
        """Normalize two values to 0-100 scale where max=100."""
        m = max(a, b, 1)
        return round(a / m * 100), round(b / m * 100)

    tp_g, tp_t = normalize(gpu.throughput_items_per_sec, tpu.throughput_items_per_sec)

    # For latency, lower is better — invert
    if max(gpu.latency_p50_ms, tpu.latency_p50_ms) > 0:
        lat_g = round((1 - gpu.latency_p50_ms / max(gpu.latency_p50_ms, tpu.latency_p50_ms)) * 50 + 50)
        lat_t = round((1 - tpu.latency_p50_ms / max(gpu.latency_p50_ms, tpu.latency_p50_ms)) * 50 + 50)
    else:
        lat_g, lat_t = 50, 50

    # Consistency: lower CV is better
    gpu_cv = min(gpu.cv, 1.0)
    tpu_cv = min(tpu.cv, 1.0)
    cons_g = round((1 - gpu_cv) * 100)
    cons_t = round((1 - tpu_cv) * 100)

    # Cost efficiency
    gpu_eff = gpu.throughput_items_per_sec / gpu_cost if gpu_cost > 0 else 0
    tpu_eff = tpu.throughput_items_per_sec / tpu_cost if tpu_cost > 0 else 0
    ce_g, ce_t = normalize(gpu_eff, tpu_eff)

    return [
        {"metric": "Throughput", "pytorch": tp_g, "jax": tp_t},
        {"metric": "Latency", "pytorch": lat_g, "jax": lat_t},
        {"metric": "Consistency", "pytorch": cons_g, "jax": cons_t},
        {"metric": "Cost Eff.", "pytorch": ce_g, "jax": ce_t},
    ]


def _pct_delta(base: float, new: float) -> float:
    if base == 0:
        return 100.0 if new > 0 else 0.0
    return ((new - base) / base) * 100


# ──────────────────────────────────────────────────────────────────────
# Dashboard HTML — self-contained file with embedded React/Recharts
# ──────────────────────────────────────────────────────────────────────

def generate_dashboard_html(
    dashboard_json_path: Path,
    output_path: Path,
) -> None:
    """Generate a self-contained HTML dashboard that embeds the data and React app."""
    with open(dashboard_json_path) as f:
        dashboard_data = json.load(f)

    jsx_path = Path(__file__).parent.parent / "profiler_report.jsx"
    if jsx_path.exists():
        with open(jsx_path) as f:
            jsx_source = _prepare_jsx_for_inline(f.read())
    else:
        jsx_source = "// profiler_report.jsx not found"

    html = _build_html(dashboard_data, jsx_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def _build_html(data: dict, jsx_source: str) -> str:
    """Build self-contained HTML with embedded UMD bundles and data."""
    data_json = json.dumps(data, indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>sglang Profiling Report</title>
  <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/prop-types@15/prop-types.min.js" crossorigin></script>
  <script src="https://unpkg.com/recharts@2/umd/Recharts.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    body {{ margin: 0; padding: 0; background: #0f1117; }}
    #root {{ min-height: 100vh; }}
  </style>
</head>
<body>
  <div id="root"></div>

  <script>
    // Inject profiling data
    window.__PROFILING_DATA__ = {data_json};
  </script>

  <script type="text/babel">
    const {{ useState }} = React;
    const {{
      BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
      Tooltip, Legend, ResponsiveContainer, RadarChart, Radar,
      PolarGrid, PolarAngleAxis, PolarRadiusAxis, AreaChart, Area
    }} = Recharts;

    {jsx_source}

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(React.createElement(ProfilingReport));
  </script>
</body>
</html>"""


def _prepare_jsx_for_inline(jsx_source: str) -> str:
    """Strip ESM-only syntax so profiler_report.jsx works in inline Babel mode."""
    cleaned_lines = []
    for line in jsx_source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            continue

        if stripped.startswith("export default function "):
            line = line.replace("export default ", "", 1)
        elif stripped.startswith("export default class "):
            line = line.replace("export default ", "", 1)
        elif re.match(r"^export default [A-Za-z_$][A-Za-z0-9_$]*\s*;\s*$", stripped):
            continue
        elif stripped.startswith("export "):
            line = line.replace("export ", "", 1)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
