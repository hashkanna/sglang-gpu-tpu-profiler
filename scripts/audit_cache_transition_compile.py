#!/usr/bin/env python3
"""Audit cache-transition compile stability from matrix summaries and raw TPU logs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_tpu_scoring_matrix import parse_log_signals


XLA_MARKER_RE = re.compile(r"XLA compilation detected")


def _audit_matrix_summary(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text())
    rows: list[dict[str, Any]] = []
    for run in payload.get("runs", []):
        run_dir = Path(str(run.get("run_dir", ""))).expanduser().resolve()
        log_path = run_dir / "artifacts" / "tpu" / "tpu_server.log"
        if not log_path.exists():
            continue
        signals = parse_log_signals(log_path.read_text(errors="ignore"))
        compile_signals = run.get("compile_signals", {}) or {}
        rows.append(
            {
                "source": "matrix_run",
                "summary_path": str(summary_path),
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "items_per_step": int(run.get("items_per_step", 0)),
                "repeat_idx": int(run.get("repeat_idx", 0)),
                "missing_cache_handle_total": int(signals.get("missing_cache_handle", 0)),
                "timed_xla_compile_count": int(
                    compile_signals.get("timed_xla_compile_count", 0)
                ),
                "warmup_xla_detect_count": int(
                    compile_signals.get("warmup_xla_detect_count", 0)
                ),
                "xla_marker_count": len(XLA_MARKER_RE.findall(log_path.read_text(errors="ignore"))),
            }
        )
    return rows


def _audit_raw_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(errors="ignore")
    signals = parse_log_signals(text)
    return {
        "source": "raw_log",
        "log_path": str(log_path),
        "missing_cache_handle_total": int(signals.get("missing_cache_handle", 0)),
        "timed_xla_compile_count": None,
        "warmup_xla_detect_count": None,
        "xla_marker_count": len(XLA_MARKER_RE.findall(text)),
    }


def _summarize(rows: list[dict[str, Any]], require_transition_observation: bool) -> dict[str, Any]:
    transitions_observed = [r for r in rows if int(r["missing_cache_handle_total"]) > 0]
    transition_with_timed_compile = [
        r
        for r in rows
        if int(r["missing_cache_handle_total"]) > 0
        and r.get("timed_xla_compile_count") is not None
        and int(r.get("timed_xla_compile_count") or 0) > 0
    ]
    transition_with_xla_marker = [
        r
        for r in rows
        if int(r["missing_cache_handle_total"]) > 0 and int(r.get("xla_marker_count") or 0) > 0
    ]

    checks = {
        "transition_compile_stability_timed": len(transition_with_timed_compile) == 0,
        "transition_compile_stability_marker": len(transition_with_xla_marker) == 0,
        "transition_observed": (len(transitions_observed) > 0)
        if require_transition_observation
        else True,
    }
    return {
        "rows_total": len(rows),
        "transitions_observed_total": len(transitions_observed),
        "transition_with_timed_compile_total": len(transition_with_timed_compile),
        "transition_with_xla_marker_total": len(transition_with_xla_marker),
        "checks": checks,
        "pass": all(bool(v) for v in checks.values()),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Cache Transition Compile Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    s = report["summary"]
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- rows_total: `{s['rows_total']}`")
    lines.append(f"- transitions_observed_total: `{s['transitions_observed_total']}`")
    lines.append(
        f"- transition_with_timed_compile_total: `{s['transition_with_timed_compile_total']}`"
    )
    lines.append(
        f"- transition_with_xla_marker_total: `{s['transition_with_xla_marker_total']}`"
    )
    lines.append(f"- audit_pass: `{str(s['pass']).lower()}`")
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    for name, passed in s["checks"].items():
        lines.append(f"- {name}: `{str(bool(passed)).lower()}`")
    lines.append("")

    lines.append("## Rows")
    lines.append("")
    lines.append(
        "| source | items_per_step | repeat | missing_cache_handle_total | timed_xla_compile_count | xla_marker_count | path |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in report["rows"]:
        path = row.get("run_dir") or row.get("log_path") or "n/a"
        lines.append(
            f"| {row['source']} | {row.get('items_per_step', 'n/a')} | {row.get('repeat_idx', 'n/a')} | "
            f"{row['missing_cache_handle_total']} | {row.get('timed_xla_compile_count')} | "
            f"{row['xla_marker_count']} | {path} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cache transition compile stability.")
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Path to matrix_summary.json (repeatable).",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Path to raw TPU server log (repeatable).",
    )
    parser.add_argument(
        "--require-transition-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require at least one cache-transition observation in provided evidence.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    summary_paths = [Path(p).expanduser().resolve() for p in args.summary]
    log_paths = [Path(p).expanduser().resolve() for p in args.log]
    missing = [str(p) for p in [*summary_paths, *log_paths] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input paths: {missing}")

    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        rows.extend(_audit_matrix_summary(summary_path))
    for log_path in log_paths:
        rows.append(_audit_raw_log(log_path))

    if not rows:
        raise RuntimeError("No evidence rows were produced (check inputs).")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_summaries": [str(p) for p in summary_paths],
        "input_logs": [str(p) for p in log_paths],
        "rows": rows,
    }
    report["summary"] = _summarize(rows, bool(args.require_transition_observation))

    out_json = Path(args.output_json).expanduser().resolve()
    out_md = Path(args.output_md).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text(render_markdown(report))
    print(f"output_json={out_json}")
    print(f"output_md={out_md}")


if __name__ == "__main__":
    main()
