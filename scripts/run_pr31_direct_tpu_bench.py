#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiler.config import WorkloadConfig
from profiler.workload import build_score_request_with_shape_contract


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    idx = (len(arr) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    if lo == hi:
        return arr[lo]
    return arr[lo] * (hi - idx) + arr[hi] * (idx - lo)


def run(
    cmd: list[str],
    check: bool = True,
    capture: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
        input=input_text,
    )


def build_workloads(cfg: dict[str, Any], workload_filter: set[str] | None) -> list[tuple[str, WorkloadConfig]]:
    shape = cfg["tpu_matrix"]["shape_contract"]
    out: list[tuple[str, WorkloadConfig]] = []
    for raw in cfg["workloads"]:
        name = str(raw["name"])
        if workload_filter and name not in workload_filter:
            continue
        wl = WorkloadConfig(
            name=name,
            query_tokens=int(raw["query_tokens"]),
            num_items=int(raw["num_items"]),
            item_tokens=int(raw["item_tokens"]),
            label_token_ids=[int(x) for x in raw.get("label_token_ids", [198])],
            apply_softmax=bool(raw.get("apply_softmax", False)),
            use_token_ids=bool(shape.get("use_token_ids", True)),
            enforce_shape_contract=bool(shape.get("strict", True)),
            query_token_buckets=[int(x) for x in shape.get("query_token_buckets", [])],
            item_token_buckets=[int(x) for x in shape.get("item_token_buckets", [])],
            num_items_buckets=[int(x) for x in shape.get("num_items_buckets", [])],
            pad_token_id=int(shape.get("pad_token_id", 0)),
            query_fill_token_id=int(shape.get("query_fill_token_id", 42)),
            item_fill_token_id=int(shape.get("item_fill_token_id", 84)),
        )
        out.append((name, wl))
    return out


def ensure_tunnel(local_port: int, host: str, user: str, key: str, remote_port: int) -> int:
    probe = run(["pgrep", "-f", f"{local_port}:127.0.0.1:{remote_port}"], check=False)
    if probe.returncode == 0 and probe.stdout.strip():
        return int(probe.stdout.strip().splitlines()[0])
    cmd = [
        "ssh",
        "-f",
        "-N",
        "-L",
        f"{local_port}:127.0.0.1:{remote_port}",
        "-i",
        key,
        "-o",
        "StrictHostKeyChecking=no",
        f"{user}@{host}",
    ]
    run(cmd, capture=True)
    probe = run(["pgrep", "-f", f"{local_port}:127.0.0.1:{remote_port}"], check=True)
    return int(probe.stdout.strip().splitlines()[0])


def remote_ssh(host: str, user: str, key: str, script: str) -> str:
    result = run(
        [
            "ssh",
            "-i",
            key,
            "-o",
            "StrictHostKeyChecking=no",
            f"{user}@{host}",
            "bash",
            "-s",
        ],
        input_text=script,
    )
    return result.stdout


def remote_scp_from(host: str, user: str, key: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "scp",
            "-i",
            key,
            "-o",
            "StrictHostKeyChecking=no",
            f"{user}@{host}:{remote_path}",
            str(local_path),
        ]
    )


def start_server(cfg: dict[str, Any], host: str, user: str, key: str, remote_log_path: str) -> None:
    model = str(cfg["experiment"]["model"])
    repo_path = str(cfg["backends"]["tpu"]["repo_path"])
    port = int(cfg["backends"]["tpu"]["url"].rsplit(":", 1)[1])
    static_args = list(cfg["tpu_server"]["static_args"])
    env = {str(k): str(v) for k, v in cfg["tpu_server"].get("env", {}).items()}
    matrix = cfg["tpu_matrix"]
    items_per_step = int(matrix["items_per_step_candidates"][0])
    max_running_requests = int(matrix["max_running_requests"])
    multi_item_extend_batch_size = int(matrix["multi_item_extend_batch_size"])
    multi_item_scoring_chunk_size = int(matrix["multi_item_scoring_chunk_size"])
    precompile_token_paddings = [int(x) for x in matrix["precompile_token_paddings"]]
    precompile_bs_paddings = [int(x) for x in matrix["precompile_bs_paddings"]]
    env_lines = "\n".join(
        f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items())
    )
    launch_cmd = [
        ".venv/bin/python",
        "-m",
        "sgl_jax.launch_server",
        "--model-path",
        model,
        "--port",
        str(port),
        *static_args,
        "--max-running-requests",
        str(max_running_requests),
        "--precompile-token-paddings",
        *[str(x) for x in precompile_token_paddings],
        "--precompile-bs-paddings",
        *[str(x) for x in precompile_bs_paddings],
        "--multi-item-scoring-chunk-size",
        str(multi_item_scoring_chunk_size),
        "--multi-item-extend-batch-size",
        str(multi_item_extend_batch_size),
        "--multi-item-score-from-cache-v2-items-per-step",
        str(items_per_step),
    ]
    launch_str = " ".join(shlex.quote(x) for x in launch_cmd)
    script = f"""
set -euo pipefail
mkdir -p {shlex.quote(str(Path(remote_log_path).parent))}
cd {shlex.quote(repo_path)}
pkill -f 'sgl_jax.launch_server.*--port {port}' >/dev/null 2>&1 || true
sleep 2
{env_lines}
export MULTI_ITEM_EXTEND_BATCH_SIZE={multi_item_extend_batch_size}
export MULTI_ITEM_EXTEND_MAX_RUNNING_REQUESTS={max_running_requests}
export MULTI_ITEM_SCORE_FROM_CACHE_V2_ITEMS_PER_STEP={items_per_step}
export MULTI_ITEM_EXTEND_PRECOMPILE_BS_PADDINGS={','.join(str(x) for x in precompile_bs_paddings)}
nohup {launch_str} > {shlex.quote(remote_log_path)} 2>&1 &
echo $! 
"""
    remote_ssh(host, user, key, script)


def stop_server(cfg: dict[str, Any], host: str, user: str, key: str) -> None:
    port = int(cfg["backends"]["tpu"]["url"].rsplit(":", 1)[1])
    run(
        [
            "ssh",
            "-i",
            key,
            "-o",
            "StrictHostKeyChecking=no",
            f"{user}@{host}",
            "bash",
            "-s",
        ],
        check=False,
        input_text=f"pkill -f 'sgl_jax.launch_server.*--port {port}' >/dev/null 2>&1 || true\n",
    )


def wait_ready(base_url: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    models_url = base_url.rstrip("/") + "/v1/models"
    while time.time() < deadline:
        try:
            r = requests.get(models_url, timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"TPU server did not become ready within {timeout_sec}s")


def benchmark(cfg: dict[str, Any], workloads: list[tuple[str, WorkloadConfig]], base_url: str) -> dict[str, Any]:
    endpoint = cfg.get("experiment", {}).get("api_endpoint", "/v1/score")
    model = str(cfg["experiment"]["model"])
    bench = cfg["benchmark"]
    out: dict[str, Any] = {}
    for name, wl in workloads:
        payload, shape_contract = build_score_request_with_shape_contract(wl, model)
        warmup_s: list[float] = []
        for _ in range(int(bench["warmup_requests"])):
            t0 = time.perf_counter()
            r = requests.post(base_url.rstrip("/") + endpoint, json=payload, timeout=int(bench["timeout_sec"]))
            r.raise_for_status()
            warmup_s.append(time.perf_counter() - t0)
        timed_s: list[float] = []
        t_start = time.perf_counter()
        for _ in range(int(bench["timed_requests"])):
            t0 = time.perf_counter()
            r = requests.post(base_url.rstrip("/") + endpoint, json=payload, timeout=int(bench["timeout_sec"]))
            r.raise_for_status()
            timed_s.append(time.perf_counter() - t0)
        elapsed = max(time.perf_counter() - t_start, 1e-9)
        out[name] = {
            "items": wl.num_items,
            "query_tokens": wl.query_tokens,
            "item_tokens": wl.item_tokens,
            "shape_contract": shape_contract,
            "warmup_s": warmup_s,
            "timed_s": timed_s,
            "avg_latency_s": statistics.mean(timed_s) if timed_s else None,
            "p50_latency_s": statistics.median(timed_s) if timed_s else None,
            "p99_latency_s": percentile(timed_s, 0.99),
            "throughput_items_per_s": (len(timed_s) * wl.num_items) / elapsed,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--ssh-key", required=True)
    ap.add_argument("--workload-filter", default="")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--local-port", type=int, default=30001)
    ap.add_argument("--remote-port", type=int, default=30001)
    ap.add_argument("--ready-timeout-sec", type=int, default=240)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    filt = {x.strip() for x in args.workload_filter.split(",") if x.strip()} or None
    workloads = build_workloads(cfg, filt)
    name = str(cfg["metadata"]["name"])
    run_dir = args.out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    remote_log = f"/home/kanna/bench_logs/{name}_{int(time.time())}.log"
    tunnel_pid = ensure_tunnel(args.local_port, args.host, args.user, os.path.expanduser(args.ssh_key), args.remote_port)
    try:
        start_server(cfg, args.host, args.user, os.path.expanduser(args.ssh_key), remote_log)
        wait_ready(f"http://127.0.0.1:{args.local_port}", args.ready_timeout_sec)
        results = benchmark(cfg, workloads, f"http://127.0.0.1:{args.local_port}")
        payload = {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": str(args.config),
            "metadata": cfg.get("metadata", {}),
            "results": results,
            "remote_log": remote_log,
            "tunnel_pid": tunnel_pid,
        }
        args.out.write_text(json.dumps(payload, indent=2))
        remote_scp_from(args.host, args.user, os.path.expanduser(args.ssh_key), remote_log, args.out.parent / "tpu_server_full.log")
    finally:
        stop_server(cfg, args.host, args.user, os.path.expanduser(args.ssh_key))


if __name__ == "__main__":
    main()
