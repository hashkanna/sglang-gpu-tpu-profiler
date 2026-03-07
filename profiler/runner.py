"""Async HTTP benchmark runner for the /v1/score API."""

import asyncio
import json
import statistics
import time
from typing import Optional

import aiohttp

from profiler.config import BackendConfig, BenchmarkConfig, ProfilerConfig, WorkloadConfig
from profiler.metrics import RequestResult, WorkloadMetrics, aggregate_results
from profiler.workload import build_score_request_with_shape_contract


_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_ERROR_SNIPPETS = (
    "server disconnected",
    "connection reset",
    "connection closed",
    "connection aborted",
    "temporarily unavailable",
    "timed out",
    "timeout",
)


class ScoreRunner:
    """Runs benchmark workloads against a backend server via HTTP."""

    def __init__(self, config: ProfilerConfig):
        self.config = config
        # Per-run diagnostics keyed by backend then workload.
        self.run_diagnostics: dict[str, dict[str, dict]] = {}

    async def run_workload(
        self,
        backend: BackendConfig,
        workload: WorkloadConfig,
    ) -> WorkloadMetrics:
        """Run a single workload against a single backend.

        Performs warmup requests (discarded), then timed requests.
        Handles XLA compilation warmup by detecting outlier first requests.
        """
        bench = self.config.benchmark
        url = backend.url.rstrip("/") + self.config.api.endpoint
        request_body, shape_contract = build_score_request_with_shape_contract(
            workload, self.config.experiment.model
        )

        timeout = aiohttp.ClientTimeout(total=bench.timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Warmup phase
            warmup_results = await self._run_requests(
                session, url, request_body, bench.warmup_requests, workload.num_items
            )
            warmup_first_ms, warmup_rest_median_ms, warmup_ratio = self._first_vs_rest_ratio(
                warmup_results
            )
            extra_warmup = self._detect_xla_warmup(warmup_results)
            if extra_warmup > 0:
                print(
                    f"  XLA compilation detected — running {extra_warmup} extra warmup requests"
                )
                await self._run_requests(
                    session, url, request_body, extra_warmup, workload.num_items
                )

            # Timed phase
            start_time = time.monotonic()
            results = await self._run_requests(
                session,
                url,
                request_body,
                bench.timed_requests,
                workload.num_items,
                concurrency=bench.concurrency,
            )
            total_time = time.monotonic() - start_time

        timed_first_ms, timed_rest_median_ms, timed_ratio = self._first_vs_rest_ratio(results)
        timed_first_outlier = bool(timed_ratio is not None and timed_ratio > 3.0)
        (
            timed_compile_like_request_ids,
            timed_compile_baseline_ms,
            timed_compile_threshold_ms,
        ) = self._detect_timed_compile_like_requests(results)
        if timed_first_outlier and not timed_compile_like_request_ids:
            # Backward-compatible fallback for short traces where only first-vs-rest is available.
            timed_compile_like_request_ids = [0]

        backend_diags = self.run_diagnostics.setdefault(backend.key, {})
        backend_diags[workload.name] = {
            "shape_contract": shape_contract,
            "warmup_requests": bench.warmup_requests,
            "warmup_successes": sum(1 for r in warmup_results if r.success),
            "warmup_failures": sum(1 for r in warmup_results if not r.success),
            "warmup_retry_total": sum(int(r.retries) for r in warmup_results),
            "warmup_first_success_ms": warmup_first_ms,
            "warmup_rest_median_ms": warmup_rest_median_ms,
            "warmup_first_over_rest_ratio": warmup_ratio,
            "warmup_xla_detected": extra_warmup > 0,
            "extra_warmup_requests": extra_warmup,
            "timed_requests": bench.timed_requests,
            "timed_successes": sum(1 for r in results if r.success),
            "timed_failures": sum(1 for r in results if not r.success),
            "timed_retry_total": sum(int(r.retries) for r in results),
            "timed_first_success_ms": timed_first_ms,
            "timed_rest_median_ms": timed_rest_median_ms,
            "timed_first_over_rest_ratio": timed_ratio,
            "timed_first_outlier": timed_first_outlier,
            "timed_compile_like_count": len(timed_compile_like_request_ids),
            "timed_compile_like_request_ids": timed_compile_like_request_ids,
            "timed_compile_baseline_ms": timed_compile_baseline_ms,
            "timed_compile_threshold_ms": timed_compile_threshold_ms,
        }

        return aggregate_results(results, workload.name, backend.key, total_time)

    async def run_all(
        self,
        backend_keys: Optional[list[str]] = None,
    ) -> dict[str, dict[str, WorkloadMetrics]]:
        """Run all workloads against specified backends.

        Returns: {backend_key: {workload_name: WorkloadMetrics}}
        """
        keys = backend_keys or self.config.backend_keys
        all_results: dict[str, dict[str, WorkloadMetrics]] = {}

        for bkey in keys:
            backend = self.config.get_backend(bkey)
            print(f"\n{'='*60}")
            print(f"Backend: {backend.name} ({backend.hardware})")
            print(f"URL: {backend.url}")
            print(f"{'='*60}")

            backend_results: dict[str, WorkloadMetrics] = {}
            for workload in self.config.workloads:
                print(f"\n  Workload: {workload.name}")
                print(f"    query_tokens={workload.query_tokens}, "
                      f"num_items={workload.num_items}, "
                      f"item_tokens={workload.item_tokens}")

                metrics = await self.run_workload(backend, workload)

                print(f"    Throughput: {metrics.throughput_items_per_sec:,.1f} items/sec")
                print(f"    Latency p50: {metrics.latency_p50_ms:,.1f} ms")
                print(f"    Latency p99: {metrics.latency_p99_ms:,.1f} ms")
                if metrics.num_failures > 0:
                    print(f"    Failures: {metrics.num_failures}/{metrics.num_requests}")
                    if metrics.error_counts:
                        top_errors = sorted(
                            metrics.error_counts.items(),
                            key=lambda kv: kv[1],
                            reverse=True,
                        )[:3]
                        for err, cnt in top_errors:
                            print(f"      - {cnt}x {err[:180]}")

                backend_results[workload.name] = metrics
            all_results[bkey] = backend_results

        return all_results

    async def _run_requests(
        self,
        session: aiohttp.ClientSession,
        url: str,
        request_body: dict,
        count: int,
        num_items: int,
        concurrency: int = 1,
    ) -> list[RequestResult]:
        """Send `count` requests with given concurrency."""
        if concurrency <= 1:
            return await self._run_sequential(session, url, request_body, count, num_items)
        return await self._run_concurrent(session, url, request_body, count, num_items, concurrency)

    async def _run_sequential(
        self,
        session: aiohttp.ClientSession,
        url: str,
        request_body: dict,
        count: int,
        num_items: int,
    ) -> list[RequestResult]:
        results = []
        for i in range(count):
            result = await self._send_request(session, url, request_body, i, num_items)
            results.append(result)
        return results

    async def _run_concurrent(
        self,
        session: aiohttp.ClientSession,
        url: str,
        request_body: dict,
        count: int,
        num_items: int,
        concurrency: int,
    ) -> list[RequestResult]:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(idx: int) -> RequestResult:
            async with semaphore:
                return await self._send_request(session, url, request_body, idx, num_items)

        tasks = [bounded_request(i) for i in range(count)]
        return list(await asyncio.gather(*tasks))

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        request_body: dict,
        request_id: int,
        num_items: int,
    ) -> RequestResult:
        """Send a single HTTP POST and measure latency."""
        start = time.monotonic()
        bench = self.config.benchmark
        max_attempts = max(1, int(bench.request_retry_attempts))
        backoff_sec = max(0.0, float(bench.request_retry_backoff_sec))

        for attempt_idx in range(1, max_attempts + 1):
            try:
                async with session.post(
                    url,
                    data=json.dumps(request_body),
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        error = f"HTTP {resp.status}: {body[:200]}"
                        retryable = int(resp.status) in _RETRYABLE_HTTP_STATUS
                        if retryable and attempt_idx < max_attempts:
                            await asyncio.sleep(backoff_sec * attempt_idx)
                            continue
                        return RequestResult(
                            request_id=request_id,
                            elapsed_ms=(time.monotonic() - start) * 1000,
                            success=False,
                            num_items=num_items,
                            error=error,
                            retries=attempt_idx - 1,
                        )

                    data = json.loads(body)
                    if not _validate_score_response(data):
                        return RequestResult(
                            request_id=request_id,
                            elapsed_ms=(time.monotonic() - start) * 1000,
                            success=False,
                            num_items=num_items,
                            error="Response missing 'scores' field",
                            retries=attempt_idx - 1,
                        )

                    return RequestResult(
                        request_id=request_id,
                        elapsed_ms=(time.monotonic() - start) * 1000,
                        success=True,
                        num_items=num_items,
                        retries=attempt_idx - 1,
                    )
            except Exception as e:
                error = str(e)
                retryable = self._is_retryable_transport_error(e, error)
                if retryable and attempt_idx < max_attempts:
                    await asyncio.sleep(backoff_sec * attempt_idx)
                    continue
                return RequestResult(
                    request_id=request_id,
                    elapsed_ms=(time.monotonic() - start) * 1000,
                    success=False,
                    num_items=num_items,
                    error=error,
                    retries=attempt_idx - 1,
                )

        return RequestResult(
            request_id=request_id,
            elapsed_ms=(time.monotonic() - start) * 1000,
            success=False,
            num_items=num_items,
            error="exhausted retry loop",
            retries=max_attempts - 1,
        )

    def _detect_xla_warmup(self, warmup_results: list[RequestResult]) -> int:
        """Detect if XLA compilation caused slow first request.

        If the first request is >3x the median of the rest, add extra warmup.
        """
        _, _, ratio = self._first_vs_rest_ratio(warmup_results)
        if ratio is not None and ratio > 3.0:
            return 2  # Run 2 extra warmup requests
        return 0

    def _first_vs_rest_ratio(
        self, results: list[RequestResult]
    ) -> tuple[float | None, float | None, float | None]:
        """Return first-success latency, median of remaining successes, and ratio."""
        successful = sorted(
            (r for r in results if r.success),
            key=lambda r: r.request_id,
        )
        if len(successful) < 2:
            return None, None, None

        first_ms = float(successful[0].elapsed_ms)
        rest = sorted(float(r.elapsed_ms) for r in successful[1:])
        median_rest_ms = rest[len(rest) // 2]
        if median_rest_ms <= 0:
            return first_ms, median_rest_ms, None
        return first_ms, median_rest_ms, first_ms / median_rest_ms

    def _detect_timed_compile_like_requests(
        self,
        results: list[RequestResult],
    ) -> tuple[list[int], float | None, float | None]:
        """Detect compile-like spikes in timed requests.

        Uses a conservative threshold:
        max(1500ms absolute floor, 3x median successful timed latency).
        """
        successful = sorted(
            (r for r in results if r.success and int(r.retries) == 0),
            key=lambda r: r.request_id,
        )
        if len(successful) < 3:
            successful = sorted(
                (r for r in results if r.success),
                key=lambda r: r.request_id,
            )
        if len(successful) < 3:
            return [], None, None

        latencies = [float(r.elapsed_ms) for r in successful]
        baseline_ms = float(statistics.median(latencies))
        if baseline_ms <= 0:
            return [], baseline_ms, None

        threshold_ms = max(1500.0, baseline_ms * 3.0)
        outlier_ids = [
            int(r.request_id) for r in successful if float(r.elapsed_ms) > threshold_ms
        ]
        return outlier_ids, baseline_ms, threshold_ms

    def _is_retryable_transport_error(self, exc: Exception, message: str) -> bool:
        if isinstance(
            exc,
            (
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                aiohttp.ClientOSError,
                aiohttp.ClientPayloadError,
                asyncio.TimeoutError,
                TimeoutError,
                ConnectionResetError,
                BrokenPipeError,
            ),
        ):
            return True
        lowered = (message or "").lower()
        return any(token in lowered for token in _RETRYABLE_ERROR_SNIPPETS)


def _validate_score_response(data: dict) -> bool:
    """Check that the response contains expected score fields."""
    return "scores" in data or "logprobs" in data
