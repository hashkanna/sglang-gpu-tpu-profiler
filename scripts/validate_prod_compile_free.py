#!/usr/bin/env python3
"""Validate compile-free and shape-clean gate signals from matrix_gates.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(gates_path: Path) -> dict:
    payload = json.loads(gates_path.read_text())
    by_ips = payload.get("by_items_per_step")
    if not isinstance(by_ips, dict) or not by_ips:
        raise ValueError("matrix_gates.json missing by_items_per_step entries")

    rows = []
    failures = []
    for ips, row in sorted(by_ips.items(), key=lambda kv: int(kv[0])):
        checks = row.get("checks", {})
        timed = int(checks.get("timed_xla_compilation", {}).get("value", -1))
        shape = int(checks.get("shape_contract", {}).get("value", -1))
        gate_pass = bool(row.get("pass", False))
        rows.append(
            {
                "items_per_step": int(ips),
                "gate_pass": gate_pass,
                "timed_xla_compile_total": timed,
                "shape_contract_violation_total": shape,
                "failed_checks": [str(x) for x in row.get("failed_checks", [])],
            }
        )
        if timed != 0:
            failures.append(f"ips={ips}: timed_xla_compile_total={timed}")
        if shape != 0:
            failures.append(f"ips={ips}: shape_contract_violation_total={shape}")
        if not gate_pass:
            failures.append(f"ips={ips}: gate_pass=false")

    return {
        "pass": len(failures) == 0,
        "matrix_gates_path": str(gates_path),
        "rows": rows,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-gates", required=True, help="Path to matrix_gates.json")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    gates_path = Path(args.matrix_gates).expanduser().resolve()
    if not gates_path.exists():
        raise FileNotFoundError(f"matrix_gates.json not found: {gates_path}")

    summary = validate(gates_path)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n")

    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
