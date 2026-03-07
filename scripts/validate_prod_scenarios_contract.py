#!/usr/bin/env python3
"""Validate normalized production scenario contract for TPU scoring benchmarks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScenarioValidation:
    scenario: str
    formula_min: int
    formula_max: int
    observed_min: int
    observed_max: int
    observed_mean: int
    observed_min_below_formula_min: bool
    observed_max_within_formula_slack: bool
    notes: list[str]


def _to_int(mapping: dict[str, Any], key: str) -> int:
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    value = mapping[key]
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for {key}, got {type(value).__name__}")
    return value


def _validate_bounds(name: str, block: dict[str, Any]) -> None:
    min_v = _to_int(block, "min")
    max_v = _to_int(block, "max")
    if min_v < 0:
        raise ValueError(f"{name}.min must be >= 0")
    if max_v < min_v:
        raise ValueError(f"{name}.max must be >= {name}.min")


def validate_contract(contract: dict[str, Any], max_slack_ratio: float = 0.05) -> dict[str, Any]:
    root_required = ["metadata", "schema", "scenarios"]
    for key in root_required:
        if key not in contract:
            raise ValueError(f"Contract missing root key: {key}")

    scenarios = contract["scenarios"]
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("scenarios must be a non-empty mapping")

    validations: list[ScenarioValidation] = []
    for scenario_name, scenario in scenarios.items():
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario {scenario_name} must be a mapping")
        ranges = scenario.get("ranges")
        if not isinstance(ranges, dict):
            raise ValueError(f"scenario {scenario_name} missing ranges mapping")

        for key in ("system_tokens", "num_candidates", "candidate_tokens", "total_tokens_observed"):
            if key not in ranges:
                raise ValueError(f"scenario {scenario_name} missing ranges.{key}")
            if not isinstance(ranges[key], dict):
                raise ValueError(f"scenario {scenario_name} ranges.{key} must be a mapping")
            _validate_bounds(f"{scenario_name}.ranges.{key}", ranges[key])

        system_min = _to_int(ranges["system_tokens"], "min")
        system_max = _to_int(ranges["system_tokens"], "max")
        cand_min = _to_int(ranges["num_candidates"], "min")
        cand_max = _to_int(ranges["num_candidates"], "max")
        tok_min = _to_int(ranges["candidate_tokens"], "min")
        tok_max = _to_int(ranges["candidate_tokens"], "max")

        observed = ranges["total_tokens_observed"]
        observed_min = _to_int(observed, "min")
        observed_max = _to_int(observed, "max")
        observed_mean = _to_int(observed, "mean")

        formula_min = system_min + (cand_min * tok_min)
        formula_max = system_max + (cand_max * tok_max)

        notes: list[str] = []
        below_formula_min = observed_min < formula_min
        if below_formula_min:
            notes.append(
                "Observed min total tokens is below strict formula min; accepted as prompt-only or partial-path logging artifact."
            )

        max_allowed = int(round(formula_max * (1.0 + max_slack_ratio)))
        within_slack = observed_max <= max_allowed
        if not within_slack:
            notes.append(
                f"Observed max total tokens ({observed_max}) exceeds formula slack cap ({max_allowed})."
            )

        if observed_mean < observed_min or observed_mean > observed_max:
            raise ValueError(
                f"scenario {scenario_name} has mean outside observed range: {observed_mean} not in [{observed_min}, {observed_max}]"
            )

        if "derived_bounds" in scenario:
            derived = scenario["derived_bounds"]
            if not isinstance(derived, dict):
                raise ValueError(f"scenario {scenario_name} derived_bounds must be a mapping")
            dmin = _to_int(derived, "total_tokens_formula_min")
            dmax = _to_int(derived, "total_tokens_formula_max")
            if dmin != formula_min or dmax != formula_max:
                raise ValueError(
                    f"scenario {scenario_name} derived_bounds mismatch: expected ({formula_min}, {formula_max}) got ({dmin}, {dmax})"
                )

        validations.append(
            ScenarioValidation(
                scenario=scenario_name,
                formula_min=formula_min,
                formula_max=formula_max,
                observed_min=observed_min,
                observed_max=observed_max,
                observed_mean=observed_mean,
                observed_min_below_formula_min=below_formula_min,
                observed_max_within_formula_slack=within_slack,
                notes=notes,
            )
        )

    home = scenarios.get("home_scoring")
    if not isinstance(home, dict):
        raise ValueError("home_scoring scenario is required")
    ambiguity = home.get("ambiguity_resolution")
    if not isinstance(ambiguity, dict):
        raise ValueError("home_scoring.ambiguity_resolution is required")

    source_values = ambiguity.get("source_values")
    if source_values != [20500, 205000]:
        raise ValueError("home_scoring ambiguity source_values must be [20500, 205000]")
    resolved_value = ambiguity.get("resolved_value")
    if resolved_value != 20500:
        raise ValueError("home_scoring ambiguity resolved_value must be 20500")

    summary = {
        "contract": contract["metadata"].get("name", "unknown"),
        "version": contract["metadata"].get("version", "unknown"),
        "pass": all(v.observed_max_within_formula_slack for v in validations),
        "scenarios": [asdict(v) for v in validations],
        "home_total_token_discrepancy_resolved_to": resolved_value,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="config/prod_scenarios_contract.yaml",
        help="Path to normalized production scenario contract YAML",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write JSON validation summary",
    )
    args = parser.parse_args()

    contract_path = Path(args.contract).expanduser().resolve()
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract file not found: {contract_path}")

    contract = yaml.safe_load(contract_path.read_text())
    if not isinstance(contract, dict):
        raise ValueError("Contract YAML root must be a mapping")

    summary = validate_contract(contract)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n")

    return 0 if summary.get("pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
