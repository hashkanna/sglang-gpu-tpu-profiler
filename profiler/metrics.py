"""Metrics dataclasses and aggregation for benchmark results."""

import statistics
from dataclasses import dataclass, field
from typing import Any
from typing import Optional


@dataclass
class RequestResult:
    """Timing result from a single HTTP request."""

    request_id: int
    elapsed_ms: float
    success: bool
    num_items: int
    error: Optional[str] = None
    retries: int = 0


@dataclass
class WorkloadMetrics:
    """Aggregated metrics for a workload run against one backend."""

    workload_name: str
    backend_key: str
    num_requests: int
    num_successes: int
    num_failures: int
    total_items: int
    total_time_sec: float

    # Throughput
    throughput_items_per_sec: float

    # Latency percentiles (ms)
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_stdev_ms: float
    latency_min_ms: float
    latency_max_ms: float

    # Raw latencies for further analysis
    raw_latencies_ms: list[float] = field(default_factory=list, repr=False)
    # Per-request failure diagnostics for instability analysis
    failed_requests: list[dict[str, Any]] = field(default_factory=list, repr=False)
    error_counts: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def failure_rate(self) -> float:
        if self.num_requests == 0:
            return 0.0
        return self.num_failures / self.num_requests

    @property
    def cv(self) -> float:
        """Coefficient of variation (stdev / mean)."""
        if self.latency_mean_ms == 0:
            return 0.0
        return self.latency_stdev_ms / self.latency_mean_ms


def percentile(sorted_data: list[float], p: float) -> float:
    """Compute percentile from pre-sorted data. p in [0, 100]."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def aggregate_results(
    results: list[RequestResult],
    workload_name: str,
    backend_key: str,
    total_time_sec: float,
) -> WorkloadMetrics:
    """Aggregate per-request results into workload-level metrics."""
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    failed_requests = [
        {
            "request_id": r.request_id,
            "elapsed_ms": r.elapsed_ms,
            "error": r.error or "",
            "retries": int(r.retries),
        }
        for r in failures
    ]
    error_counts: dict[str, int] = {}
    for r in failures:
        key = (r.error or "").strip() or "unknown_error"
        error_counts[key] = error_counts.get(key, 0) + 1

    latencies = sorted(r.elapsed_ms for r in successes)
    total_items = sum(r.num_items for r in successes)

    if not latencies:
        return WorkloadMetrics(
            workload_name=workload_name,
            backend_key=backend_key,
            num_requests=len(results),
            num_successes=0,
            num_failures=len(failures),
            total_items=0,
            total_time_sec=total_time_sec,
            throughput_items_per_sec=0.0,
            latency_p50_ms=0.0,
            latency_p90_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            latency_mean_ms=0.0,
            latency_stdev_ms=0.0,
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            raw_latencies_ms=[],
            failed_requests=failed_requests,
            error_counts=error_counts,
        )

    mean_lat = statistics.mean(latencies)
    stdev_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    throughput = total_items / total_time_sec if total_time_sec > 0 else 0.0

    return WorkloadMetrics(
        workload_name=workload_name,
        backend_key=backend_key,
        num_requests=len(results),
        num_successes=len(successes),
        num_failures=len(failures),
        total_items=total_items,
        total_time_sec=total_time_sec,
        throughput_items_per_sec=throughput,
        latency_p50_ms=percentile(latencies, 50),
        latency_p90_ms=percentile(latencies, 90),
        latency_p95_ms=percentile(latencies, 95),
        latency_p99_ms=percentile(latencies, 99),
        latency_mean_ms=mean_lat,
        latency_stdev_ms=stdev_lat,
        latency_min_ms=latencies[0],
        latency_max_ms=latencies[-1],
        raw_latencies_ms=latencies,
        failed_requests=failed_requests,
        error_counts=error_counts,
    )
