#!/usr/bin/env python3
"""Analyze production prewarm plan coverage and cap discipline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Ensure local imports resolve when executed as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pr28_baseline import baseline_workloads, load_baseline, matrix_defaults
from run_tpu_scoring_matrix import (
    apply_shape_contract_to_workloads,
    build_shape_bucket_warmup_plan,
)


def _normalized(raw: Any, fallback: int) -> list[int]:
    if isinstance(raw, list):
        out = sorted({int(x) for x in raw if int(x) > 0})
        if out:
            return out
    return [int(fallback)]


def _pick_bucket(logical: int, buckets: list[int]) -> int:
    logical_i = int(logical)
    for bucket in buckets:
        if logical_i <= int(bucket):
            return int(bucket)
    return logical_i


def analyze(*, baseline_path: Path, max_requests: int | None) -> dict[str, Any]:
    baseline = load_baseline(baseline_path)
    matrix = matrix_defaults(baseline)
    workloads = baseline_workloads(baseline)

    shape = dict(matrix.get("shape_contract", {}))
    workloads_shaped = apply_shape_contract_to_workloads(
        workloads=workloads,
        enabled=bool(shape.get("enabled", True)),
        use_token_ids=bool(shape.get("use_token_ids", True)),
        strict=bool(shape.get("strict", True)),
        query_token_buckets=[int(v) for v in shape.get("query_token_buckets", [])],
        item_token_buckets=[int(v) for v in shape.get("item_token_buckets", [])],
        num_items_buckets=[int(v) for v in shape.get("num_items_buckets", [])],
        pad_token_id=int(shape.get("pad_token_id", 0)),
        query_fill_token_id=int(shape.get("query_fill_token_id", 42)),
        item_fill_token_id=int(shape.get("item_fill_token_id", 84)),
    )

    cap = int(max_requests) if max_requests is not None else int(matrix.get("warmup_max_shape_bucket_requests", 64))
    plan, truncated = build_shape_bucket_warmup_plan(workloads_shaped, max_requests=cap)

    active_keys: set[tuple[int, int, int]] = set()
    plan_keys: set[tuple[int, int, int]] = set()

    cubic_upper_bound = 0
    query_union: set[int] = set()
    item_union: set[int] = set()
    num_items_union: set[int] = set()

    for wl in workloads_shaped:
        shape_enabled = bool(wl.get("enforce_shape_contract", False)) and bool(
            wl.get("use_token_ids", False)
        )
        if not shape_enabled:
            continue
        q_buckets = _normalized(wl.get("query_token_buckets"), int(wl["query_tokens"]))
        i_buckets = _normalized(wl.get("item_token_buckets"), int(wl["item_tokens"]))
        n_buckets = _normalized(wl.get("num_items_buckets"), int(wl["num_items"]))
        cubic_upper_bound += len(q_buckets) * len(i_buckets) * len(n_buckets)
        query_union.update(q_buckets)
        item_union.update(i_buckets)
        num_items_union.update(n_buckets)

        active_key = (
            _pick_bucket(int(wl["query_tokens"]), q_buckets),
            _pick_bucket(int(wl["num_items"]), n_buckets),
            _pick_bucket(int(wl["item_tokens"]), i_buckets),
        )
        active_keys.add(active_key)

    for wl in plan:
        plan_keys.add((int(wl["query_tokens"]), int(wl["num_items"]), int(wl["item_tokens"])))

    missing_active = sorted(list(active_keys - plan_keys))
    linear_estimate = len(active_keys) + len(query_union) + len(num_items_union) + len(item_union)

    summary = {
        "pass": len(missing_active) == 0 and len(plan) <= cap,
        "cap": cap,
        "plan_size": len(plan),
        "truncated": bool(truncated),
        "active_shape_count": len(active_keys),
        "active_shapes": sorted(list(active_keys)),
        "missing_active_shapes": missing_active,
        "legacy_cubic_upper_bound": int(cubic_upper_bound),
        "linear_estimate": int(linear_estimate),
        "query_bucket_union": sorted(query_union),
        "num_items_bucket_union": sorted(num_items_union),
        "item_bucket_union": sorted(item_union),
        "plan_preview": plan[: min(12, len(plan))],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="config/prod_scenario_scoring_baseline.yaml",
        help="Path to production baseline YAML",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional override for warmup request cap",
    )
    parser.add_argument("--json-out", default="", help="Optional output path")
    args = parser.parse_args()

    baseline_path = Path(args.baseline).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    summary = analyze(baseline_path=baseline_path, max_requests=args.max_requests)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n")

    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
