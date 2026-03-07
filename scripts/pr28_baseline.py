#!/usr/bin/env python3
"""Shared loader for canonical PR28 scoring baseline defaults."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "pr28_scoring_baseline.yaml"
)


def load_baseline(path: str | Path | None = None) -> dict[str, Any]:
    baseline_path = Path(path) if path is not None else DEFAULT_BASELINE_PATH
    baseline_path = baseline_path.expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline config not found: {baseline_path}")
    raw = yaml.safe_load(baseline_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Baseline config must be a mapping: {baseline_path}")
    _validate_baseline(raw, baseline_path)
    return raw


def baseline_workloads(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(w) for w in raw["workloads"]]


def benchmark_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "warmup_requests": int(raw["benchmark"]["warmup_requests"]),
        "timed_requests": int(raw["benchmark"]["timed_requests"]),
        "concurrency": int(raw["benchmark"]["concurrency"]),
        "timeout_sec": int(raw["benchmark"]["timeout_sec"]),
        "request_retry_attempts": int(raw["benchmark"].get("request_retry_attempts", 3)),
        "request_retry_backoff_sec": float(raw["benchmark"].get("request_retry_backoff_sec", 0.25)),
    }


def matrix_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    section = raw["tpu_matrix"]
    return {
        "items_per_step_candidates": [int(v) for v in section["items_per_step_candidates"]],
        "max_running_requests": int(section["max_running_requests"]),
        "multi_item_extend_batch_size": int(section["multi_item_extend_batch_size"]),
        "multi_item_scoring_chunk_size": int(section["multi_item_scoring_chunk_size"]),
        "precompile_token_paddings": [int(v) for v in section["precompile_token_paddings"]],
        "precompile_bs_paddings": [int(v) for v in section["precompile_bs_paddings"]],
        "stabilization_sleep_sec": float(section["stabilization_sleep_sec"]),
        "workload_warmup_attempts": int(section["workload_warmup_attempts"]),
        "workload_warmup_backoff_sec": float(section["workload_warmup_backoff_sec"]),
        "align_items_per_step_with_workloads": bool(
            section.get("align_items_per_step_with_workloads", True)
        ),
        "auto_bump_lane_capacity_with_workload": bool(
            section.get("auto_bump_lane_capacity_with_workload", False)
        ),
        "lane_capacity_bump_cap": int(section.get("lane_capacity_bump_cap", 128)),
        "warmup_all_shape_buckets": bool(section.get("warmup_all_shape_buckets", True)),
        "warmup_max_shape_bucket_requests": int(
            section.get("warmup_max_shape_bucket_requests", 64)
        ),
        "allow_score_full_vocab_fallback": bool(
            section.get("allow_score_full_vocab_fallback", False)
        ),
        "require_cache_transition_exercise": bool(
            section.get("require_cache_transition_exercise", False)
        ),
        "shape_contract": _matrix_shape_contract_defaults(section),
        "gate_max_p99_p50": {
            str(k): float(v) for k, v in section["gate_max_p99_p50"].items()
        },
    }


def _matrix_shape_contract_defaults(section: dict[str, Any]) -> dict[str, Any]:
    raw = section.get("shape_contract")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("tpu_matrix.shape_contract must be a mapping when provided")

    def _int_list(key: str, fallback: list[int]) -> list[int]:
        values = raw.get(key, fallback)
        if not isinstance(values, list) or not values:
            raise ValueError(f"tpu_matrix.shape_contract.{key} must be a non-empty list[int]")
        out = [int(v) for v in values]
        if any(v <= 0 for v in out):
            raise ValueError(f"tpu_matrix.shape_contract.{key} must contain positive integers")
        return out

    def _non_negative_int(key: str, fallback: int) -> int:
        value = int(raw.get(key, fallback))
        if value < 0:
            raise ValueError(f"tpu_matrix.shape_contract.{key} must be >= 0")
        return value

    return {
        "enabled": bool(raw.get("enabled", True)),
        "use_token_ids": bool(raw.get("use_token_ids", True)),
        "strict": bool(raw.get("strict", True)),
        "query_token_buckets": _int_list("query_token_buckets", [120, 500, 2000]),
        "item_token_buckets": _int_list("item_token_buckets", [20]),
        "num_items_buckets": _int_list("num_items_buckets", [10, 100, 500]),
        "pad_token_id": _non_negative_int("pad_token_id", 0),
        "query_fill_token_id": _non_negative_int("query_fill_token_id", 42),
        "item_fill_token_id": _non_negative_int("item_fill_token_id", 84),
    }


def tpu_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    tpu = raw["backends"]["tpu"]
    return {
        "name": str(tpu["name"]),
        "hardware": str(tpu["hardware"]),
        "cost_per_hour": float(tpu["cost_per_hour"]),
        "url": str(tpu["url"]),
        "repo_path": str(tpu["repo_path"]),
    }


def tpu_server_env(raw: dict[str, Any]) -> dict[str, str]:
    env = raw["tpu_server"]["env"]
    return {str(k): str(v) for k, v in env.items()}


def tpu_server_static_args(raw: dict[str, Any]) -> list[str]:
    return [str(v) for v in raw["tpu_server"]["static_args"]]


def gate_max_p99_p50_csv(raw: dict[str, Any]) -> str:
    workloads = [w["name"] for w in baseline_workloads(raw)]
    gates = matrix_defaults(raw)["gate_max_p99_p50"]
    parts: list[str] = []
    for wl in workloads:
        if wl in gates:
            parts.append(f"{wl}:{gates[wl]}")
    # Include any extra keys to avoid silently dropping user-provided gates.
    for wl in sorted(gates.keys()):
        if wl not in workloads:
            parts.append(f"{wl}:{gates[wl]}")
    return ",".join(parts)


def shell_default_env(raw: dict[str, Any]) -> list[tuple[str, str]]:
    workloads = {w["name"]: w for w in baseline_workloads(raw)}
    for required in ("pr28_hotshape", "small_batch", "medium_batch"):
        if required not in workloads:
            raise ValueError(f"Baseline workloads missing required key: {required}")

    hotshape = workloads["pr28_hotshape"]
    small = workloads["small_batch"]
    medium = workloads["medium_batch"]
    benchmark = benchmark_defaults(raw)
    gpu = raw["backends"]["gpu"]
    tpu = raw["backends"]["tpu"]

    label_ids = hotshape.get("label_token_ids", [198])
    label_ids_csv = ",".join(str(int(x)) for x in label_ids)
    apply_softmax = str(bool(hotshape.get("apply_softmax", False))).lower()

    gpu_server_tokens = [str(t) for t in gpu.get("default_server_args", [])]
    gpu_server_cmd = " ".join(shlex.quote(token) for token in gpu_server_tokens)

    return [
        ("MODEL_NAME", str(raw["experiment"]["model"])),
        ("API_ENDPOINT", str(raw["experiment"]["api_endpoint"])),
        ("GPU_BACKEND_NAME", str(gpu["name"])),
        ("TPU_BACKEND_NAME", str(tpu["name"])),
        ("GPU_HARDWARE", str(gpu["hardware"])),
        ("TPU_HARDWARE", str(tpu["hardware"])),
        ("GPU_COST_PER_HOUR", str(gpu["cost_per_hour"])),
        ("TPU_COST_PER_HOUR", str(tpu["cost_per_hour"])),
        ("PR28_HOTSHAPE_QUERY_TOKENS", str(hotshape["query_tokens"])),
        ("PR28_HOTSHAPE_NUM_ITEMS", str(hotshape["num_items"])),
        ("PR28_HOTSHAPE_ITEM_TOKENS", str(hotshape["item_tokens"])),
        ("SMALL_BATCH_QUERY_TOKENS", str(small["query_tokens"])),
        ("SMALL_BATCH_NUM_ITEMS", str(small["num_items"])),
        ("SMALL_BATCH_ITEM_TOKENS", str(small["item_tokens"])),
        ("MEDIUM_BATCH_QUERY_TOKENS", str(medium["query_tokens"])),
        ("MEDIUM_BATCH_NUM_ITEMS", str(medium["num_items"])),
        ("MEDIUM_BATCH_ITEM_TOKENS", str(medium["item_tokens"])),
        ("LABEL_TOKEN_IDS", label_ids_csv),
        ("APPLY_SOFTMAX", apply_softmax),
        ("WARMUP_REQUESTS", str(benchmark["warmup_requests"])),
        ("TIMED_REQUESTS", str(benchmark["timed_requests"])),
        ("CONCURRENCY", str(benchmark["concurrency"])),
        ("TIMEOUT_SEC", str(benchmark["timeout_sec"])),
        ("TPU_DEFAULT_URL", str(tpu["url"])),
        ("TPU_DEFAULT_REPO_PATH", str(tpu["repo_path"])),
        ("GPU_BASELINE_SERVER_START_CMD", gpu_server_cmd),
    ]


def emit_shell_defaults(raw: dict[str, Any]) -> str:
    lines = []
    for key, value in shell_default_env(raw):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("$", "\\$")
            .replace("`", "\\`")
        )
        lines.append(f': "${{{key}:={escaped}}}"')
    return "\n".join(lines)


def _validate_baseline(raw: dict[str, Any], path: Path) -> None:
    required_sections = [
        "experiment",
        "backends",
        "workloads",
        "benchmark",
        "tpu_matrix",
        "tpu_server",
    ]
    for key in required_sections:
        if key not in raw:
            raise ValueError(f"Baseline missing '{key}' section: {path}")

    if "gpu" not in raw["backends"] or "tpu" not in raw["backends"]:
        raise ValueError(f"Baseline backends must contain gpu and tpu sections: {path}")

    if not isinstance(raw["workloads"], list) or not raw["workloads"]:
        raise ValueError(f"Baseline workloads must be a non-empty list: {path}")

    matrix = raw["tpu_matrix"]
    for key in (
        "items_per_step_candidates",
        "max_running_requests",
        "multi_item_extend_batch_size",
        "multi_item_scoring_chunk_size",
        "precompile_token_paddings",
        "precompile_bs_paddings",
        "gate_max_p99_p50",
    ):
        if key not in matrix:
            raise ValueError(f"Baseline tpu_matrix missing '{key}': {path}")
    if "shape_contract" in matrix and not isinstance(matrix["shape_contract"], dict):
        raise ValueError(f"Baseline tpu_matrix.shape_contract must be a mapping: {path}")
    if "warmup_max_shape_bucket_requests" in matrix:
        if int(matrix["warmup_max_shape_bucket_requests"]) < 1:
            raise ValueError(
                f"Baseline tpu_matrix.warmup_max_shape_bucket_requests must be >= 1: {path}"
            )
    if "lane_capacity_bump_cap" in matrix:
        if int(matrix["lane_capacity_bump_cap"]) < 1:
            raise ValueError(
                f"Baseline tpu_matrix.lane_capacity_bump_cap must be >= 1: {path}"
            )

    tpu_server = raw["tpu_server"]
    if "env" not in tpu_server or "static_args" not in tpu_server:
        raise ValueError(f"Baseline tpu_server must include env and static_args: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect/export canonical PR28 scoring baseline.")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument(
        "--format",
        choices=["json", "shell-defaults"],
        default="json",
    )
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    if args.format == "json":
        print(json.dumps(baseline, indent=2))
    else:
        print(emit_shell_defaults(baseline))


if __name__ == "__main__":
    main()
