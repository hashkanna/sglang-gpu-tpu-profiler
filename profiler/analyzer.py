"""Cross-backend comparison and anomaly detection."""

from dataclasses import dataclass, field
from typing import Optional

from profiler.config import ProfilerConfig
from profiler.metrics import WorkloadMetrics


@dataclass
class MetricDelta:
    """Comparison of a single metric across two backends."""

    metric_name: str
    gpu_value: float
    tpu_value: float
    delta_pct: float  # Positive = TPU is higher/better for throughput
    winner: str  # "gpu", "tpu", or "tie"

    @property
    def abs_delta_pct(self) -> float:
        return abs(self.delta_pct)


@dataclass
class Anomaly:
    """An automatically detected anomaly."""

    severity: str  # "high", "medium", "low", "info"
    backend: str  # "GPU", "TPU", "Both"
    finding: str
    suggestion: str


@dataclass
class WorkloadComparison:
    """Full comparison for one workload across GPU and TPU."""

    workload_name: str
    gpu_metrics: WorkloadMetrics
    tpu_metrics: WorkloadMetrics
    deltas: list[MetricDelta] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Full comparison result across all workloads."""

    workload_comparisons: list[WorkloadComparison] = field(default_factory=list)
    global_anomalies: list[Anomaly] = field(default_factory=list)

    @property
    def all_anomalies(self) -> list[Anomaly]:
        out = list(self.global_anomalies)
        for wc in self.workload_comparisons:
            out.extend(wc.anomalies)
        return out


def _compute_delta_pct(gpu_val: float, tpu_val: float) -> float:
    """Compute percentage delta. Positive = TPU is higher."""
    if gpu_val == 0:
        return 100.0 if tpu_val > 0 else 0.0
    return ((tpu_val - gpu_val) / gpu_val) * 100


def _throughput_winner(gpu: float, tpu: float) -> str:
    """Higher throughput wins."""
    if abs(gpu - tpu) / max(gpu, tpu, 1) < 0.02:
        return "tie"
    return "tpu" if tpu > gpu else "gpu"


def _latency_winner(gpu: float, tpu: float) -> str:
    """Lower latency wins."""
    if max(gpu, tpu) == 0 or abs(gpu - tpu) / max(gpu, tpu) < 0.02:
        return "tie"
    return "tpu" if tpu < gpu else "gpu"


def _compare_workload(
    gpu: WorkloadMetrics,
    tpu: WorkloadMetrics,
) -> WorkloadComparison:
    """Compare metrics for a single workload across two backends."""
    deltas = []

    # Throughput — higher is better
    tp_delta = _compute_delta_pct(gpu.throughput_items_per_sec, tpu.throughput_items_per_sec)
    deltas.append(MetricDelta(
        "throughput_items_per_sec",
        gpu.throughput_items_per_sec,
        tpu.throughput_items_per_sec,
        tp_delta,
        _throughput_winner(gpu.throughput_items_per_sec, tpu.throughput_items_per_sec),
    ))

    # Latency metrics — lower is better
    for metric_name, gpu_val, tpu_val in [
        ("latency_p50_ms", gpu.latency_p50_ms, tpu.latency_p50_ms),
        ("latency_p90_ms", gpu.latency_p90_ms, tpu.latency_p90_ms),
        ("latency_p95_ms", gpu.latency_p95_ms, tpu.latency_p95_ms),
        ("latency_p99_ms", gpu.latency_p99_ms, tpu.latency_p99_ms),
        ("latency_mean_ms", gpu.latency_mean_ms, tpu.latency_mean_ms),
    ]:
        delta = _compute_delta_pct(gpu_val, tpu_val)
        deltas.append(MetricDelta(metric_name, gpu_val, tpu_val, delta, _latency_winner(gpu_val, tpu_val)))

    # Detect anomalies for this workload
    anomalies = _detect_workload_anomalies(gpu, tpu, deltas)

    return WorkloadComparison(
        workload_name=gpu.workload_name,
        gpu_metrics=gpu,
        tpu_metrics=tpu,
        deltas=deltas,
        anomalies=anomalies,
    )


def _detect_workload_anomalies(
    gpu: WorkloadMetrics,
    tpu: WorkloadMetrics,
    deltas: list[MetricDelta],
) -> list[Anomaly]:
    """Apply anomaly detection rules to a workload comparison."""
    anomalies = []

    # Rule 1: High failure rate (>10%)
    for label, m in [("GPU", gpu), ("TPU", tpu)]:
        if m.failure_rate > 0.10:
            anomalies.append(Anomaly(
                severity="high",
                backend=label,
                finding=f"{label} failure rate is {m.failure_rate:.0%} "
                        f"({m.num_failures}/{m.num_requests} requests failed) "
                        f"on workload '{m.workload_name}'.",
                suggestion="Check server health, timeout settings, and error logs.",
            ))

    # Rule 2: High variance (CV > 0.5)
    for label, m in [("GPU", gpu), ("TPU", tpu)]:
        if m.cv > 0.5:
            anomalies.append(Anomaly(
                severity="medium",
                backend=label,
                finding=f"{label} latency has high variance (CV={m.cv:.2f}, "
                        f"stdev={m.latency_stdev_ms:.1f}ms, mean={m.latency_mean_ms:.1f}ms) "
                        f"on workload '{m.workload_name}'.",
                suggestion="Possible XLA recompilation, GC pauses, or thermal throttling. "
                           "Increase warmup requests or pin process to cores.",
            ))

    # Rule 3: Large throughput delta (>40%)
    tp_delta = next((d for d in deltas if d.metric_name == "throughput_items_per_sec"), None)
    if tp_delta and tp_delta.abs_delta_pct > 40:
        faster = "TPU" if tp_delta.delta_pct > 0 else "GPU"
        anomalies.append(Anomaly(
            severity="info",
            backend="Both",
            finding=f"Large throughput delta ({tp_delta.delta_pct:+.1f}%) — "
                    f"{faster} is significantly faster on workload '{gpu.workload_name}'.",
            suggestion="Verify both servers are healthy and using the same model. "
                       "This may be expected for compute-bound workloads with different hardware.",
        ))

    # Rule 4: P99 tail spike (p99 > 3x p50)
    for label, m in [("GPU", gpu), ("TPU", tpu)]:
        if m.latency_p50_ms > 0 and m.latency_p99_ms > 3 * m.latency_p50_ms:
            ratio = m.latency_p99_ms / m.latency_p50_ms
            anomalies.append(Anomaly(
                severity="medium",
                backend=label,
                finding=f"{label} has a p99 tail spike: p99={m.latency_p99_ms:.1f}ms is "
                        f"{ratio:.1f}x the p50={m.latency_p50_ms:.1f}ms on '{m.workload_name}'.",
                suggestion="Check for GC pauses, scheduling jitter, or occasional XLA recompilation. "
                           "Consider increasing warmup or using request-level timeouts.",
            ))

    return anomalies


def analyze_results(
    all_results: dict[str, dict[str, WorkloadMetrics]],
    config: ProfilerConfig,
) -> ComparisonResult:
    """Compare results across backends.

    Expects all_results to have at least two backend keys (typically "gpu" and "tpu").
    """
    keys = list(all_results.keys())
    if len(keys) < 2:
        raise ValueError(
            f"Need at least 2 backends for comparison, got {len(keys)}: {keys}"
        )

    # Use first two backends as gpu/tpu pair
    gpu_key, tpu_key = keys[0], keys[1]
    gpu_results = all_results[gpu_key]
    tpu_results = all_results[tpu_key]

    comparisons = []
    for workload in config.workloads:
        wname = workload.name
        if wname in gpu_results and wname in tpu_results:
            comp = _compare_workload(gpu_results[wname], tpu_results[wname])
            comparisons.append(comp)

    return ComparisonResult(workload_comparisons=comparisons)
