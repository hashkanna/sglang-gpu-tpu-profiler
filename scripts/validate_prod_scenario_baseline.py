#!/usr/bin/env python3
"""Validate production-shaped baseline config against normalized scenario contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_WORKLOAD_NAMES = [
    "track_low",
    "track_mean",
    "track_high",
    "home_low",
    "home_mean",
    "home_high",
]


def _require_mapping(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at '{key}'")
    return value


def _require_list(root: dict[str, Any], key: str) -> list[Any]:
    value = root.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected list at '{key}'")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _range_min_max(block: dict[str, Any], key: str) -> tuple[int, int]:
    sub = block.get(key)
    if not isinstance(sub, dict):
        raise ValueError(f"Missing mapping range: {key}")
    min_v = sub.get("min")
    max_v = sub.get("max")
    if not isinstance(min_v, int) or not isinstance(max_v, int):
        raise ValueError(f"Range '{key}' min/max must be integers")
    if min_v > max_v:
        raise ValueError(f"Range '{key}' min must be <= max")
    return min_v, max_v


def validate(baseline: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    workloads = _require_list(baseline, "workloads")
    if len(workloads) != 6:
        raise ValueError(f"Expected exactly 6 workloads, got {len(workloads)}")

    names = [str(w.get("name")) for w in workloads]
    if names != EXPECTED_WORKLOAD_NAMES:
        raise ValueError(f"Workload ordering mismatch: expected {EXPECTED_WORKLOAD_NAMES}, got {names}")

    reproducibility = _require_mapping(baseline, "reproducibility")
    seed = reproducibility.get("random_seed")
    if not isinstance(seed, int):
        raise ValueError("reproducibility.random_seed must be integer")

    matrix = _require_mapping(baseline, "tpu_matrix")
    shape = _require_mapping(matrix, "shape_contract")
    q_buckets = [int(v) for v in shape.get("query_token_buckets", [])]
    item_buckets = [int(v) for v in shape.get("item_token_buckets", [])]
    n_buckets = [int(v) for v in shape.get("num_items_buckets", [])]
    if not q_buckets or not item_buckets or not n_buckets:
        raise ValueError("shape_contract buckets must be non-empty")

    scenarios = _require_mapping(contract, "scenarios")

    per_workload: list[dict[str, Any]] = []
    for wl in workloads:
        name = str(wl["name"])
        scenario_name = str(wl.get("scenario"))
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario for workload {name}: {scenario_name}")
        scenario = _require_mapping(scenarios, scenario_name)
        ranges = _require_mapping(scenario, "ranges")

        q = int(wl["query_tokens"])
        n = int(wl["num_items"])
        it = int(wl["item_tokens"])
        expected_total = int(wl["total_tokens_expected"])
        computed_total = q + (n * it)
        if computed_total != expected_total:
            raise ValueError(
                f"Workload {name} total mismatch: expected {expected_total}, computed {computed_total}"
            )

        q_min, q_max = _range_min_max(ranges, "system_tokens")
        n_min, n_max = _range_min_max(ranges, "num_candidates")
        it_min, it_max = _range_min_max(ranges, "candidate_tokens")

        if not (q_min <= q <= q_max):
            raise ValueError(f"{name}.query_tokens={q} outside scenario range [{q_min}, {q_max}]")
        if not (n_min <= n <= n_max):
            raise ValueError(f"{name}.num_items={n} outside scenario range [{n_min}, {n_max}]")
        if not (it_min <= it <= it_max):
            raise ValueError(f"{name}.item_tokens={it} outside scenario range [{it_min}, {it_max}]")

        if q not in q_buckets:
            raise ValueError(f"{name}.query_tokens={q} missing from shape query bucket list")
        if it not in item_buckets:
            raise ValueError(f"{name}.item_tokens={it} missing from shape item bucket list")
        if n not in n_buckets:
            raise ValueError(f"{name}.num_items={n} missing from shape num-items bucket list")

        per_workload.append(
            {
                "name": name,
                "scenario": scenario_name,
                "profile": str(wl.get("profile")),
                "query_tokens": q,
                "num_items": n,
                "item_tokens": it,
                "total_tokens": computed_total,
            }
        )

    total_by_name = {row["name"]: row["total_tokens"] for row in per_workload}
    if total_by_name["track_mean"] != 5700:
        raise ValueError("track_mean total_tokens must be 5700")
    if total_by_name["home_mean"] != 1975:
        raise ValueError("home_mean total_tokens must be 1975")

    return {
        "pass": True,
        "workload_count": len(per_workload),
        "random_seed": seed,
        "workloads": per_workload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="config/prod_scenario_scoring_baseline.yaml",
        help="Path to production-shaped baseline YAML",
    )
    parser.add_argument(
        "--contract",
        default="config/prod_scenarios_contract.yaml",
        help="Path to normalized production contract YAML",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path for JSON validation output",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline).expanduser().resolve()
    contract_path = Path(args.contract).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract not found: {contract_path}")

    baseline = load_yaml(baseline_path)
    contract = load_yaml(contract_path)
    result = validate(baseline, contract)

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
