#!/usr/bin/env python3
"""Generate capacity sizing ranges from production-shaped TPU/GPU measurements."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _ceil_div(n: float, d: float) -> int:
    if d <= 0:
        return math.inf  # type: ignore[return-value]
    return int(math.ceil(n / d))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate accelerator capacity ranges.")
    parser.add_argument("--final-report", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    report = _load_json(args.final_report)
    rows = report.get("per_workload", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("final report missing per_workload rows")

    per_backend_reqps: dict[str, dict[str, float]] = {"tpu": {}, "gpu": {}}
    scenario_map: dict[str, list[str]] = {"track": [], "home": []}

    for row in rows:
        if not isinstance(row, dict):
            continue
        wl = str(row["workload"])
        num_items = int(row["shape"]["num_items"])
        if num_items <= 0:
            continue
        for backend in ("tpu", "gpu"):
            throughput = float(row[backend]["throughput_items_per_sec"])
            per_backend_reqps[backend][wl] = throughput / float(num_items)
        if wl.startswith("track_"):
            scenario_map["track"].append(wl)
        elif wl.startswith("home_"):
            scenario_map["home"].append(wl)

    scenario_capacity: dict[str, dict[str, dict[str, float]]] = {}
    for scenario, wls in scenario_map.items():
        scenario_capacity[scenario] = {}
        for backend in ("tpu", "gpu"):
            vals = [per_backend_reqps[backend][wl] for wl in wls if wl in per_backend_reqps[backend]]
            if not vals:
                continue
            scenario_capacity[scenario][backend] = {
                "req_per_sec_conservative": min(vals),
                "req_per_sec_typical": float(median(vals)),
                "req_per_sec_optimistic": max(vals),
            }

    # Scenario demand points from the production requirement notes.
    # Kept explicit so teams can re-run with modified demand assumptions.
    demands = {
        "track": {
            "production_observed_low": 2.0,
            "production_observed_high": 10.0,
            "target_table_rps": 40.0,
            "ingress_extreme_from_doc": 75000.0,
        },
        "home": {
            "production_observed": 27.0,
            "target_table_rps": 40.0,
            "ingress_extreme_from_doc": 1700.0,
        },
        "combined": {
            "production_observed_track_high_plus_home": 10.0 + 27.0,
            "target_table_track_plus_home": 40.0 + 40.0,
        },
    }

    sizing: dict[str, Any] = {}
    for scenario in ("track", "home"):
        sizing[scenario] = {}
        for backend in ("tpu", "gpu"):
            cap = scenario_capacity.get(scenario, {}).get(backend)
            if not cap:
                continue
            demand_rows = {}
            for demand_label, rps in demands[scenario].items():
                demand_rows[demand_label] = {
                    "demand_rps": rps,
                    "accelerators_conservative": _ceil_div(rps, cap["req_per_sec_conservative"]),
                    "accelerators_typical": _ceil_div(rps, cap["req_per_sec_typical"]),
                    "accelerators_optimistic": _ceil_div(rps, cap["req_per_sec_optimistic"]),
                }
            sizing[scenario][backend] = {
                "capacity_req_per_sec": cap,
                "sizing_by_demand": demand_rows,
            }

    # Combined sizing using conservative per-scenario rates added together.
    combined = {}
    for backend in ("tpu", "gpu"):
        t = scenario_capacity.get("track", {}).get(backend)
        h = scenario_capacity.get("home", {}).get(backend)
        if not t or not h:
            continue
        cap_conservative = t["req_per_sec_conservative"] + h["req_per_sec_conservative"]
        cap_typical = t["req_per_sec_typical"] + h["req_per_sec_typical"]
        cap_optimistic = t["req_per_sec_optimistic"] + h["req_per_sec_optimistic"]
        by_demand = {}
        for demand_label, rps in demands["combined"].items():
            by_demand[demand_label] = {
                "demand_rps": rps,
                "accelerators_conservative": _ceil_div(rps, cap_conservative),
                "accelerators_typical": _ceil_div(rps, cap_typical),
                "accelerators_optimistic": _ceil_div(rps, cap_optimistic),
            }
        combined[backend] = {
            "capacity_req_per_sec": {
                "conservative": cap_conservative,
                "typical": cap_typical,
                "optimistic": cap_optimistic,
            },
            "sizing_by_demand": by_demand,
        }

    output = {
        "source_final_report": str(args.final_report),
        "scenario_capacity_reqps": scenario_capacity,
        "demand_assumptions_rps": demands,
        "sizing": sizing,
        "combined_sizing": combined,
        "notes": [
            "Capacity is based on measured request/sec derived from items/sec and workload num_items.",
            "Ranges are conservative/typical/optimistic from low/median/high measured profiles per scenario.",
            "This model does not include multi-tenant queuing safety margins; add headroom before production commitments.",
        ],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2))

    lines: list[str] = []
    lines.append("# Capacity Plan (Production-Shaped Scoring)")
    lines.append("")
    lines.append("## Scenario Capacity (Measured Request/sec per Accelerator)")
    lines.append("")
    lines.append("| Scenario | Backend | Conservative req/s | Typical req/s | Optimistic req/s |")
    lines.append("|---|---|---:|---:|---:|")
    for scenario in ("track", "home"):
        for backend in ("tpu", "gpu"):
            cap = scenario_capacity.get(scenario, {}).get(backend)
            if not cap:
                continue
            lines.append(
                "| {s} | {b} | {c:.3f} | {t:.3f} | {o:.3f} |".format(
                    s=scenario,
                    b=backend,
                    c=cap["req_per_sec_conservative"],
                    t=cap["req_per_sec_typical"],
                    o=cap["req_per_sec_optimistic"],
                )
            )
    lines.append("")
    lines.append("## Accelerator Sizing by Scenario Demand")
    lines.append("")
    lines.append("| Scenario | Backend | Demand Label | Demand req/s | Conservative accel | Typical accel | Optimistic accel |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for scenario in ("track", "home"):
        for backend in ("tpu", "gpu"):
            block = sizing.get(scenario, {}).get(backend, {})
            for demand_label, row in block.get("sizing_by_demand", {}).items():
                lines.append(
                    "| {s} | {b} | {d} | {rps:.1f} | {ac} | {at} | {ao} |".format(
                        s=scenario,
                        b=backend,
                        d=demand_label,
                        rps=row["demand_rps"],
                        ac=row["accelerators_conservative"],
                        at=row["accelerators_typical"],
                        ao=row["accelerators_optimistic"],
                    )
                )
    lines.append("")
    lines.append("## Combined Track+Home Sizing")
    lines.append("")
    lines.append("| Backend | Demand Label | Demand req/s | Conservative accel | Typical accel | Optimistic accel |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for backend in ("tpu", "gpu"):
        block = combined.get(backend, {})
        for demand_label, row in block.get("sizing_by_demand", {}).items():
            lines.append(
                "| {b} | {d} | {rps:.1f} | {ac} | {at} | {ao} |".format(
                    b=backend,
                    d=demand_label,
                    rps=row["demand_rps"],
                    ac=row["accelerators_conservative"],
                    at=row["accelerators_typical"],
                    ao=row["accelerators_optimistic"],
                )
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in output["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    args.md_out.write_text("\n".join(lines))

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
