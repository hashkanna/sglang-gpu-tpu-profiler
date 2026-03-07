#!/usr/bin/env python3
"""Evaluate TPU matrix summary against regression gates; exit non-zero on regressions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping at: {path}")
    return raw


def _find_workload_row(rows: list[dict[str, Any]], ips: int) -> dict[str, Any] | None:
    for row in rows:
        if int(row.get("items_per_step", -1)) == int(ips):
            return row
    return None


def _load_run_concurrency_samples(
    matrix_summary: dict[str, Any], recommended_ips: int
) -> tuple[list[int], int]:
    samples: list[int] = []
    missing_metadata = 0
    for run in matrix_summary.get("runs", []) or []:
        if int(run.get("items_per_step", -1)) != int(recommended_ips):
            continue
        if run.get("run_error"):
            continue
        run_dir = run.get("run_dir")
        if not run_dir:
            missing_metadata += 1
            continue
        metadata_path = Path(str(run_dir)) / "run_metadata.json"
        if not metadata_path.exists():
            missing_metadata += 1
            continue
        try:
            payload = json.loads(metadata_path.read_text())
            samples.append(int(payload.get("benchmark", {}).get("concurrency", 0)))
        except Exception:
            missing_metadata += 1
    return samples, missing_metadata


def evaluate(matrix_summary: dict[str, Any], gate_cfg: dict[str, Any], *, strict_workload_coverage: bool) -> dict[str, Any]:
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    global_cfg = gate_cfg.get("global", {}) or {}
    run_cfg = gate_cfg.get("run_contract", {}) or {}
    workload_cfg = gate_cfg.get("workloads", {}) or {}

    max_failure_rate = float(global_cfg.get("max_failure_rate", 0.0))
    max_timed_xla_compile_total = int(global_cfg.get("max_timed_xla_compile_total", 0))
    max_shape_contract_violation_total = int(
        global_cfg.get("max_shape_contract_violation_total", 0)
    )
    max_score_fastpath_non_success_total = int(
        global_cfg.get("max_score_fastpath_non_success_total", 0)
    )
    min_timed_requests_per_repeat = int(run_cfg.get("min_timed_requests_per_repeat", 1))
    min_repeats_requested = int(run_cfg.get("min_repeats_requested", 1))
    required_concurrency = run_cfg.get("required_concurrency")

    recommended = matrix_summary.get("recommended", {}) or {}
    recommended_ips = int(recommended.get("items_per_step") or 0)
    if recommended_ips <= 0:
        failures.append("missing/invalid recommended.items_per_step in matrix summary")
        return {"pass": False, "failures": failures, "checks": checks}

    gates = matrix_summary.get("gates", {}) or {}
    gates_by_ips = (gates.get("by_items_per_step", {}) or {})
    rec_gate = gates_by_ips.get(str(recommended_ips), {}) or gates_by_ips.get(recommended_ips, {})
    gate_pass = bool(rec_gate.get("pass", False))
    checks.append(
        {
            "name": "recommended_candidate_gate_pass",
            "value": gate_pass,
            "threshold": True,
            "pass": gate_pass,
        }
    )
    if not gate_pass:
        failures.append(
            f"recommended candidate ips={recommended_ips} failed internal matrix gates: "
            f"{rec_gate.get('failed_checks', [])}"
        )

    timed_requests_per_repeat = int(matrix_summary.get("timed_requests_per_repeat") or 0)
    timed_window_pass = timed_requests_per_repeat >= min_timed_requests_per_repeat
    checks.append(
        {
            "name": "timed_requests_per_repeat",
            "value": timed_requests_per_repeat,
            "threshold": min_timed_requests_per_repeat,
            "pass": timed_window_pass,
        }
    )
    if not timed_window_pass:
        failures.append(
            "timed_requests_per_repeat="
            f"{timed_requests_per_repeat} < min {min_timed_requests_per_repeat}"
        )

    repeats_requested = int(matrix_summary.get("repeats_requested") or 0)
    repeats_pass = repeats_requested >= min_repeats_requested
    checks.append(
        {
            "name": "repeats_requested",
            "value": repeats_requested,
            "threshold": min_repeats_requested,
            "pass": repeats_pass,
        }
    )
    if not repeats_pass:
        failures.append(f"repeats_requested={repeats_requested} < min {min_repeats_requested}")

    if required_concurrency is not None:
        expected_concurrency = int(required_concurrency)
        concurrency_samples, missing_metadata = _load_run_concurrency_samples(
            matrix_summary, recommended_ips
        )
        concurrency_pass = (
            len(concurrency_samples) > 0
            and missing_metadata == 0
            and all(int(v) == expected_concurrency for v in concurrency_samples)
        )
        checks.append(
            {
                "name": "run_concurrency",
                "value": sorted(set(concurrency_samples)),
                "threshold": expected_concurrency,
                "missing_metadata": missing_metadata,
                "pass": concurrency_pass,
            }
        )
        if not concurrency_pass:
            failures.append(
                "run_concurrency mismatch: "
                f"expected={expected_concurrency}, observed={sorted(set(concurrency_samples))}, "
                f"missing_metadata={missing_metadata}"
            )

    log_signals = (
        matrix_summary.get("aggregates", {})
        .get("log_signals_by_items_per_step", {})
        .get(str(recommended_ips), {})
    )
    timed_xla_compile_total = int(log_signals.get("timed_xla_compile_total", 0))
    timed_compile_pass = timed_xla_compile_total <= max_timed_xla_compile_total
    checks.append(
        {
            "name": "timed_compile_regression",
            "value": timed_xla_compile_total,
            "threshold": max_timed_xla_compile_total,
            "pass": timed_compile_pass,
        }
    )
    if not timed_compile_pass:
        failures.append(
            f"timed_xla_compile_total={timed_xla_compile_total} > {max_timed_xla_compile_total}"
        )

    shape_contract_violation_total = int(log_signals.get("shape_contract_violation_total", 0))
    shape_violation_pass = shape_contract_violation_total <= max_shape_contract_violation_total
    checks.append(
        {
            "name": "shape_contract_regression",
            "value": shape_contract_violation_total,
            "threshold": max_shape_contract_violation_total,
            "pass": shape_violation_pass,
        }
    )
    if not shape_violation_pass:
        failures.append(
            "shape_contract_violation_total="
            f"{shape_contract_violation_total} > {max_shape_contract_violation_total}"
        )

    fastpath_non_success_total = int(log_signals.get("fastpath_non_success_total", 0))
    fastpath_pass = fastpath_non_success_total <= max_score_fastpath_non_success_total
    checks.append(
        {
            "name": "score_fastpath_non_success_regression",
            "value": fastpath_non_success_total,
            "threshold": max_score_fastpath_non_success_total,
            "pass": fastpath_pass,
        }
    )
    if not fastpath_pass:
        failures.append(
            "fastpath_non_success_total="
            f"{fastpath_non_success_total} > {max_score_fastpath_non_success_total}"
        )

    by_workload = matrix_summary.get("aggregates", {}).get("by_workload", {}) or {}
    for wl_name, rows in by_workload.items():
        if not isinstance(rows, list):
            continue

        wl_gate_cfg = workload_cfg.get(wl_name)
        if wl_gate_cfg is None:
            if strict_workload_coverage:
                failures.append(f"missing workload gate config for '{wl_name}'")
            continue

        row = _find_workload_row(rows, recommended_ips)
        if row is None:
            failures.append(
                f"{wl_name}: no aggregate row for recommended items_per_step={recommended_ips}"
            )
            continue

        throughput = float(row.get("throughput_median_items_per_sec") or 0.0)
        failure_rate = float(row.get("failure_rate") or 0.0)
        ratio = row.get("p99_p50_ratio_median")
        ratio_v = float(ratio) if ratio is not None else None
        p99_ms = float(row.get("latency_p99_median_ms") or 0.0)

        min_tput = float(wl_gate_cfg.get("min_throughput_items_per_sec", 0.0))
        max_ratio = float(wl_gate_cfg.get("max_p99_p50_ratio", 1e9))
        max_p99_ms = wl_gate_cfg.get("max_latency_p99_ms")
        max_p99_ms_v = float(max_p99_ms) if max_p99_ms is not None else None

        tput_pass = throughput >= min_tput
        failrate_pass = failure_rate <= max_failure_rate
        ratio_pass = ratio_v is not None and ratio_v <= max_ratio
        p99_pass = True if max_p99_ms_v is None else p99_ms <= max_p99_ms_v

        checks.extend(
            [
                {
                    "name": f"{wl_name}.throughput_regression",
                    "value": throughput,
                    "threshold": min_tput,
                    "pass": tput_pass,
                },
                {
                    "name": f"{wl_name}.failure_rate_regression",
                    "value": failure_rate,
                    "threshold": max_failure_rate,
                    "pass": failrate_pass,
                },
                {
                    "name": f"{wl_name}.tail_ratio_regression",
                    "value": ratio_v,
                    "threshold": max_ratio,
                    "pass": ratio_pass,
                },
            ]
        )
        if max_p99_ms_v is not None:
            checks.append(
                {
                    "name": f"{wl_name}.p99_latency_regression",
                    "value": p99_ms,
                    "threshold": max_p99_ms_v,
                    "pass": p99_pass,
                }
            )

        if not tput_pass:
            failures.append(f"{wl_name}: throughput {throughput:.3f} < min {min_tput:.3f}")
        if not failrate_pass:
            failures.append(
                f"{wl_name}: failure_rate {failure_rate:.4f} > max {max_failure_rate:.4f}"
            )
        if not ratio_pass:
            failures.append(
                f"{wl_name}: p99/p50 {ratio_v if ratio_v is not None else 'n/a'} > max {max_ratio:.3f}"
            )
        if not p99_pass:
            failures.append(f"{wl_name}: p99 {p99_ms:.3f} > max {max_p99_ms_v:.3f}")

    return {"pass": len(failures) == 0, "failures": failures, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TPU matrix summary against regression gates.")
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument(
        "--gates-config",
        default=str(Path(__file__).resolve().parent.parent / "config" / "tpu_regression_gates.yaml"),
    )
    parser.add_argument(
        "--strict-workload-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if matrix summary contains workload names missing from gate config.",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    matrix_summary_path = Path(args.matrix_summary).expanduser().resolve()
    gates_config_path = Path(args.gates_config).expanduser().resolve()
    if not matrix_summary_path.exists():
        raise FileNotFoundError(f"matrix summary not found: {matrix_summary_path}")
    if not gates_config_path.exists():
        raise FileNotFoundError(f"gates config not found: {gates_config_path}")

    matrix_summary = json.loads(matrix_summary_path.read_text())
    gate_cfg = _load_yaml(gates_config_path)
    result = evaluate(
        matrix_summary,
        gate_cfg,
        strict_workload_coverage=bool(args.strict_workload_coverage),
    )
    result["matrix_summary"] = str(matrix_summary_path)
    result["gates_config"] = str(gates_config_path)

    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"output_json={out}")

    if result["pass"]:
        print("REGRESSION_GATES=PASS")
        sys.exit(0)

    print("REGRESSION_GATES=FAIL")
    for failure in result["failures"]:
        print(f"- {failure}")
    sys.exit(1)


if __name__ == "__main__":
    main()
