#!/usr/bin/env python3
"""Generate reproducible Monte Carlo request-shape workloads for production scenarios."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONTRACT = Path("config/prod_scenarios_contract.yaml")
DEFAULT_BASELINE = Path("config/prod_scenario_scoring_baseline.yaml")

# Keep weights explicit and deterministic for reproducible scenario mixes.
PROFILE_WEIGHTS = {
    "track_scoring": {"low": 0.20, "mean": 0.60, "high": 0.20},
    "home_scoring": {"low": 0.60, "mean": 0.30, "high": 0.10},
}


@dataclass
class ScenarioBounds:
    system_min: int
    system_max: int
    items_min: int
    items_max: int
    item_tokens_min: int
    item_tokens_max: int
    observed_total_min: int
    observed_total_max: int


@dataclass
class ProfileAnchor:
    query_tokens: int
    num_items: int
    item_tokens: int


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _scenario_bounds(contract: dict[str, Any]) -> dict[str, ScenarioBounds]:
    scenarios = contract.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError("Contract missing scenarios mapping")

    out: dict[str, ScenarioBounds] = {}
    for name in ("track_scoring", "home_scoring"):
        row = scenarios.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"Contract missing scenario: {name}")
        ranges = row.get("ranges")
        if not isinstance(ranges, dict):
            raise ValueError(f"Scenario {name} missing ranges mapping")

        def _mm(block_name: str) -> tuple[int, int]:
            block = ranges.get(block_name)
            if not isinstance(block, dict):
                raise ValueError(f"Scenario {name} missing ranges.{block_name}")
            min_v = block.get("min")
            max_v = block.get("max")
            if not isinstance(min_v, int) or not isinstance(max_v, int):
                raise ValueError(f"Scenario {name} ranges.{block_name} min/max must be int")
            return min_v, max_v

        system_min, system_max = _mm("system_tokens")
        items_min, items_max = _mm("num_candidates")
        item_tokens_min, item_tokens_max = _mm("candidate_tokens")
        obs_min, obs_max = _mm("total_tokens_observed")

        out[name] = ScenarioBounds(
            system_min=system_min,
            system_max=system_max,
            items_min=items_min,
            items_max=items_max,
            item_tokens_min=item_tokens_min,
            item_tokens_max=item_tokens_max,
            observed_total_min=obs_min,
            observed_total_max=obs_max,
        )
    return out


def _profile_anchors(baseline: dict[str, Any]) -> dict[str, dict[str, ProfileAnchor]]:
    workloads = baseline.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("Baseline missing workloads list")

    by_scenario: dict[str, dict[str, ProfileAnchor]] = {
        "track_scoring": {},
        "home_scoring": {},
    }

    for row in workloads:
        if not isinstance(row, dict):
            continue
        scenario = str(row.get("scenario", ""))
        profile = str(row.get("profile", ""))
        if scenario not in by_scenario or profile not in ("low", "mean", "high"):
            continue
        by_scenario[scenario][profile] = ProfileAnchor(
            query_tokens=int(row["query_tokens"]),
            num_items=int(row["num_items"]),
            item_tokens=int(row["item_tokens"]),
        )

    for scenario, profiles in by_scenario.items():
        missing = [p for p in ("low", "mean", "high") if p not in profiles]
        if missing:
            raise ValueError(f"Baseline missing {scenario} profiles: {missing}")

    return by_scenario


def _clamp(value: int, min_v: int, max_v: int) -> int:
    return min(max(value, min_v), max_v)


def _sample_near_anchor(
    rng: random.Random,
    *,
    anchor: int,
    min_v: int,
    max_v: int,
    std_ratio: float,
    min_std: float,
) -> int:
    std = max(min_std, abs(anchor) * std_ratio)
    sampled = int(round(rng.gauss(anchor, std)))
    return _clamp(sampled, min_v, max_v)


def _sample_profile(rng: random.Random, weights: dict[str, float]) -> str:
    # random.choices is deterministic for fixed seed and ordered input.
    ordered_profiles = ["low", "mean", "high"]
    ordered_weights = [weights[p] for p in ordered_profiles]
    return rng.choices(ordered_profiles, weights=ordered_weights, k=1)[0]


def _generate_samples_for_scenario(
    *,
    rng: random.Random,
    scenario: str,
    bounds: ScenarioBounds,
    anchors: dict[str, ProfileAnchor],
    count: int,
    label_token_ids: list[int],
    apply_softmax: bool,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for idx in range(count):
        profile = _sample_profile(rng, PROFILE_WEIGHTS[scenario])
        anchor = anchors[profile]

        query_tokens = _sample_near_anchor(
            rng,
            anchor=anchor.query_tokens,
            min_v=bounds.system_min,
            max_v=bounds.system_max,
            std_ratio=0.10,
            min_std=2.0,
        )
        item_tokens = _sample_near_anchor(
            rng,
            anchor=anchor.item_tokens,
            min_v=bounds.item_tokens_min,
            max_v=bounds.item_tokens_max,
            std_ratio=0.12,
            min_std=1.0,
        )
        num_items = _sample_near_anchor(
            rng,
            anchor=anchor.num_items,
            min_v=bounds.items_min,
            max_v=bounds.items_max,
            std_ratio=0.22,
            min_std=1.0,
        )

        total_tokens = query_tokens + (num_items * item_tokens)

        # Keep synthetic totals bounded so generated traffic remains production-shaped.
        while total_tokens > bounds.observed_total_max and num_items > bounds.items_min:
            num_items -= 1
            total_tokens = query_tokens + (num_items * item_tokens)
        while total_tokens > bounds.observed_total_max and item_tokens > bounds.item_tokens_min:
            item_tokens -= 1
            total_tokens = query_tokens + (num_items * item_tokens)

        sample = {
            "id": f"{scenario}:{idx:05d}",
            "scenario": scenario,
            "profile": profile,
            "query_tokens": int(query_tokens),
            "num_items": int(num_items),
            "item_tokens": int(item_tokens),
            "total_tokens": int(total_tokens),
            "label_token_ids": [int(v) for v in label_token_ids],
            "apply_softmax": bool(apply_softmax),
        }
        samples.append(sample)

    return samples


def _scenario_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sorted(int(s["total_tokens"]) for s in samples)
    profile_counts: dict[str, int] = {"low": 0, "mean": 0, "high": 0}
    for row in samples:
        profile_counts[str(row["profile"])] += 1

    def _pct(p: float) -> int:
        idx = min(len(total_tokens) - 1, max(0, int(round((len(total_tokens) - 1) * p))))
        return total_tokens[idx]

    return {
        "count": len(samples),
        "total_tokens_min": total_tokens[0],
        "total_tokens_p50": _pct(0.50),
        "total_tokens_p90": _pct(0.90),
        "total_tokens_p99": _pct(0.99),
        "total_tokens_max": total_tokens[-1],
        "profile_counts": profile_counts,
    }


def generate_workloads(
    *,
    contract_path: Path,
    baseline_path: Path,
    seed: int,
    track_samples: int,
    home_samples: int,
) -> dict[str, Any]:
    contract = _load_yaml(contract_path)
    baseline = _load_yaml(baseline_path)

    bounds = _scenario_bounds(contract)
    anchors = _profile_anchors(baseline)

    rng = random.Random(seed)

    label_token_ids = [198]
    apply_softmax = False
    workloads = baseline.get("workloads")
    if isinstance(workloads, list) and workloads:
        first = workloads[0]
        if isinstance(first, dict):
            label_token_ids = [int(v) for v in first.get("label_token_ids", [198])]
            apply_softmax = bool(first.get("apply_softmax", False))

    track_rows = _generate_samples_for_scenario(
        rng=rng,
        scenario="track_scoring",
        bounds=bounds["track_scoring"],
        anchors=anchors["track_scoring"],
        count=track_samples,
        label_token_ids=label_token_ids,
        apply_softmax=apply_softmax,
    )
    home_rows = _generate_samples_for_scenario(
        rng=rng,
        scenario="home_scoring",
        bounds=bounds["home_scoring"],
        anchors=anchors["home_scoring"],
        count=home_samples,
        label_token_ids=label_token_ids,
        apply_softmax=apply_softmax,
    )

    result = {
        "metadata": {
            "name": "prod_shape_workloads",
            "version": 1,
            "seed": int(seed),
            "contract_path": str(contract_path),
            "baseline_path": str(baseline_path),
            "track_samples": int(track_samples),
            "home_samples": int(home_samples),
        },
        "summaries": {
            "track_scoring": _scenario_summary(track_rows),
            "home_scoring": _scenario_summary(home_rows),
        },
        "samples": {
            "track_scoring": track_rows,
            "home_scoring": home_rows,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--seed", type=int, default=20260301)
    parser.add_argument("--track-samples", type=int, default=1000)
    parser.add_argument("--home-samples", type=int, default=1000)
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path. Default: results/generated_workloads/prod_shape_workloads_seed<seed>.json",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional compact summary JSON output path",
    )
    args = parser.parse_args()

    contract_path = Path(args.contract).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()

    if not contract_path.exists():
        raise FileNotFoundError(f"Contract not found: {contract_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    if args.track_samples < 1 or args.home_samples < 1:
        raise ValueError("track-samples and home-samples must be >= 1")

    output_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path("results/generated_workloads")
        .resolve()
        .joinpath(f"prod_shape_workloads_seed{args.seed}.json")
    )

    payload = generate_workloads(
        contract_path=contract_path,
        baseline_path=baseline_path,
        seed=args.seed,
        track_samples=args.track_samples,
        home_samples=args.home_samples,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.summary_out:
        summary_path = Path(args.summary_out).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_only = {
            "metadata": payload["metadata"],
            "summaries": payload["summaries"],
        }
        summary_path.write_text(json.dumps(summary_only, indent=2, sort_keys=True) + "\n")

    # Always print compact summary for CLI users and validation logs.
    print(
        json.dumps(
            {
                "out": str(output_path),
                "seed": payload["metadata"]["seed"],
                "summaries": payload["summaries"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
