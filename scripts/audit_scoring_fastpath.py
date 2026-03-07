#!/usr/bin/env python3
"""Audit score-path fastpath behavior from prior matrix summary artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_tpu_scoring_matrix import parse_log_signals


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total_score_samples = sum(int(r["score_samples"]) for r in runs)
    total_fastpath_samples = sum(int(r["fastpath_metrics_samples"]) for r in runs)
    total_attempted = sum(int(r["fastpath_attempted_total"]) for r in runs)
    total_succeeded = sum(int(r["fastpath_succeeded_total"]) for r in runs)
    total_non_success = sum(int(r["fastpath_non_success_total"]) for r in runs)
    total_fallback = sum(int(r["fastpath_fallback_total"]) for r in runs)
    fallback_reasons = sorted(
        {reason for r in runs for reason in r.get("fastpath_fallback_reasons", [])}
    )
    server_label_only_disabled = sum(
        1 for r in runs if r.get("server_label_only_enabled") is False
    )
    server_fastpath_metrics_disabled = sum(
        1 for r in runs if r.get("server_fastpath_metrics_enabled") is False
    )

    checks = {
        "fastpath_metrics_coverage": total_fastpath_samples >= total_score_samples,
        "fastpath_attempted_nonzero": total_attempted > 0,
        "fastpath_success_equals_attempted": total_succeeded >= total_attempted,
        "fastpath_non_success_zero": total_non_success == 0,
        "fastpath_fallback_zero": total_fallback == 0,
        "server_label_only_not_disabled": server_label_only_disabled == 0,
        "server_fastpath_metrics_not_disabled": server_fastpath_metrics_disabled == 0,
    }
    return {
        "total_runs": len(runs),
        "total_score_samples": total_score_samples,
        "total_fastpath_metrics_samples": total_fastpath_samples,
        "total_fastpath_attempted": total_attempted,
        "total_fastpath_succeeded": total_succeeded,
        "total_fastpath_non_success": total_non_success,
        "total_fastpath_fallback": total_fallback,
        "fastpath_fallback_reasons": fallback_reasons,
        "server_label_only_disabled_runs": server_label_only_disabled,
        "server_fastpath_metrics_disabled_runs": server_fastpath_metrics_disabled,
        "checks": checks,
        "pass": all(bool(v) for v in checks.values()),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Scoring Fastpath Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    s = report["summary"]
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- matrix_summaries: `{len(report['input_summaries'])}`")
    lines.append(f"- total_runs: `{s['total_runs']}`")
    lines.append(f"- total_score_samples: `{s['total_score_samples']}`")
    lines.append(f"- total_fastpath_metrics_samples: `{s['total_fastpath_metrics_samples']}`")
    lines.append(f"- total_fastpath_attempted: `{s['total_fastpath_attempted']}`")
    lines.append(f"- total_fastpath_succeeded: `{s['total_fastpath_succeeded']}`")
    lines.append(f"- total_fastpath_non_success: `{s['total_fastpath_non_success']}`")
    lines.append(f"- total_fastpath_fallback: `{s['total_fastpath_fallback']}`")
    lines.append(
        "- fallback_reasons: "
        + (", ".join(s["fastpath_fallback_reasons"]) if s["fastpath_fallback_reasons"] else "none")
    )
    lines.append(f"- audit_pass: `{str(s['pass']).lower()}`")
    lines.append("")

    lines.append("## Gate Checks")
    lines.append("")
    for name, passed in s["checks"].items():
        lines.append(f"- {name}: `{str(bool(passed)).lower()}`")
    lines.append("")

    lines.append("## Per-Run")
    lines.append("")
    lines.append(
        "| items_per_step | repeat | score_samples | fastpath_samples | attempted | succeeded | non_success | fallback | label_only_enabled | fastpath_metrics_enabled | run_dir |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for run in report["runs"]:
        lines.append(
            f"| {run['items_per_step']} | {run['repeat_idx']} | {run['score_samples']} | "
            f"{run['fastpath_metrics_samples']} | {run['fastpath_attempted_total']} | "
            f"{run['fastpath_succeeded_total']} | {run['fastpath_non_success_total']} | "
            f"{run['fastpath_fallback_total']} | {run['server_label_only_enabled']} | "
            f"{run['server_fastpath_metrics_enabled']} | {run['run_dir']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit scoring fastpath behavior from matrix runs.")
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Path to matrix_summary.json (can be passed multiple times).",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    summary_paths = [Path(p).expanduser().resolve() for p in args.summary]
    missing = [str(p) for p in summary_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing summary paths: {missing}")

    runs: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text())
        for run in payload.get("runs", []):
            run_dir = Path(str(run.get("run_dir", ""))).expanduser().resolve()
            log_path = run_dir / "artifacts" / "tpu" / "tpu_server.log"
            if not log_path.exists():
                continue
            signals = parse_log_signals(log_path.read_text(errors="ignore"))
            runs.append(
                {
                    "summary_path": str(summary_path),
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                    "items_per_step": int(run.get("items_per_step", 0)),
                    "repeat_idx": int(run.get("repeat_idx", 0)),
                    "score_samples": int(signals.get("score_samples", 0)),
                    "fastpath_metrics_samples": int(signals.get("fastpath_metrics_samples", 0)),
                    "fastpath_attempted_total": int(signals.get("fastpath_attempted_total", 0)),
                    "fastpath_succeeded_total": int(signals.get("fastpath_succeeded_total", 0)),
                    "fastpath_non_success_total": int(signals.get("fastpath_non_success_total", 0)),
                    "fastpath_fallback_total": int(signals.get("fastpath_fallback_total", 0)),
                    "fastpath_fallback_reasons": [
                        str(x)
                        for x in signals.get("fastpath_fallback_reasons", [])
                        if str(x)
                    ],
                    "server_label_only_enabled": signals.get("server_label_only_enabled"),
                    "server_fastpath_metrics_enabled": signals.get(
                        "server_fastpath_metrics_enabled"
                    ),
                }
            )

    if not runs:
        raise RuntimeError("No run logs found to audit.")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_summaries": [str(p) for p in summary_paths],
        "runs": sorted(runs, key=lambda r: (int(r["items_per_step"]), int(r["repeat_idx"]))),
    }
    report["summary"] = _summarize_runs(report["runs"])

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
