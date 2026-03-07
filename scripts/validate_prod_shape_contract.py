#!/usr/bin/env python3
"""Validate strict shape-contract coverage for production-shaped sample workloads."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

# Ensure local package imports resolve when executed as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiler.config import WorkloadConfig
from profiler.workload import build_score_request_with_shape_contract
from pr28_baseline import load_baseline, matrix_defaults


def _load_samples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    samples = payload.get("samples")
    if not isinstance(samples, dict):
        raise ValueError("samples JSON must contain a 'samples' mapping")

    out: list[dict[str, Any]] = []
    for scenario in ("track_scoring", "home_scoring"):
        rows = samples.get(scenario)
        if not isinstance(rows, list):
            raise ValueError(f"samples.{scenario} must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"samples.{scenario} row must be an object")
            out.append(dict(row))
    return out


def validate(
    *,
    baseline_path: Path,
    samples_path: Path,
) -> dict[str, Any]:
    baseline = load_baseline(baseline_path)
    matrix = matrix_defaults(baseline)
    shape = dict(matrix.get("shape_contract", {}))
    model = str(baseline["experiment"]["model"])

    query_buckets = [int(v) for v in shape.get("query_token_buckets", [])]
    item_buckets = [int(v) for v in shape.get("item_token_buckets", [])]
    num_items_buckets = [int(v) for v in shape.get("num_items_buckets", [])]

    if not query_buckets or not item_buckets or not num_items_buckets:
        raise ValueError("Baseline shape_contract buckets must be non-empty")

    rows = _load_samples(samples_path)

    violations: list[str] = []
    used_query_buckets: Counter[int] = Counter()
    used_item_buckets: Counter[int] = Counter()
    used_num_items_buckets: Counter[int] = Counter()

    max_logical = {
        "query_tokens": 0,
        "item_tokens": 0,
        "num_items": 0,
    }

    for idx, row in enumerate(rows):
        query_tokens = int(row["query_tokens"])
        num_items = int(row["num_items"])
        item_tokens = int(row["item_tokens"])

        max_logical["query_tokens"] = max(max_logical["query_tokens"], query_tokens)
        max_logical["num_items"] = max(max_logical["num_items"], num_items)
        max_logical["item_tokens"] = max(max_logical["item_tokens"], item_tokens)

        workload = WorkloadConfig(
            name=str(row.get("id", f"sample_{idx}")),
            query_tokens=query_tokens,
            num_items=num_items,
            item_tokens=item_tokens,
            label_token_ids=[int(v) for v in row.get("label_token_ids", [198])],
            apply_softmax=bool(row.get("apply_softmax", False)),
            use_token_ids=True,
            enforce_shape_contract=True,
            query_token_buckets=query_buckets,
            item_token_buckets=item_buckets,
            num_items_buckets=num_items_buckets,
            pad_token_id=int(shape.get("pad_token_id", 0)),
            query_fill_token_id=int(shape.get("query_fill_token_id", 42)),
            item_fill_token_id=int(shape.get("item_fill_token_id", 84)),
        )

        try:
            _payload, diag = build_score_request_with_shape_contract(workload, model)
        except Exception as exc:  # noqa: BLE001 - keep evidence for failure report
            violations.append(f"{workload.name}: exception: {exc}")
            continue

        bucket_shape = diag.get("bucket_shape", {})
        bq = int(bucket_shape.get("query_tokens", 0))
        bi = int(bucket_shape.get("item_tokens", 0))
        bn = int(bucket_shape.get("num_items", 0))
        used_query_buckets[bq] += 1
        used_item_buckets[bi] += 1
        used_num_items_buckets[bn] += 1

        if not bool(diag.get("request_matches_bucket", False)):
            violations.append(f"{workload.name}: request_matches_bucket=false")
        if not bool(diag.get("bucket_shape_is_approved", False)):
            violations.append(f"{workload.name}: bucket_shape_is_approved=false")

        dynamic = diag.get("dynamic_bucket_fallback", {})
        if any(bool(dynamic.get(key, False)) for key in ("query_tokens", "item_tokens", "num_items")):
            violations.append(f"{workload.name}: dynamic_bucket_fallback=true")

    summary = {
        "pass": len(violations) == 0,
        "sample_count": len(rows),
        "violation_count": len(violations),
        "violations": violations,
        "approved_buckets": {
            "query_tokens": query_buckets,
            "item_tokens": item_buckets,
            "num_items": num_items_buckets,
        },
        "used_buckets": {
            "query_tokens": dict(sorted(used_query_buckets.items())),
            "item_tokens": dict(sorted(used_item_buckets.items())),
            "num_items": dict(sorted(used_num_items_buckets.items())),
        },
        "max_logical_shape_seen": max_logical,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="config/prod_scenario_scoring_baseline.yaml",
        help="Production-shaped baseline YAML",
    )
    parser.add_argument(
        "--samples",
        default=(
            "results/task_artifacts/20260301_t18_monte_carlo_workload_gen/"
            "workloads_seed20260301_run1.json"
        ),
        help="Generated sample workload JSON",
    )
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    baseline_path = Path(args.baseline).expanduser().resolve()
    samples_path = Path(args.samples).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples JSON not found: {samples_path}")

    summary = validate(baseline_path=baseline_path, samples_path=samples_path)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n")

    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
