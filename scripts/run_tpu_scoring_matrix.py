#!/usr/bin/env python3
"""Run an automated TPU scoring tuning matrix and emit a ranked report."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

# Ensure local package imports (e.g., `profiler.*`) resolve when executed as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pr28_baseline import (
    DEFAULT_BASELINE_PATH,
    baseline_workloads,
    benchmark_defaults,
    gate_max_p99_p50_csv,
    load_baseline,
    matrix_defaults,
    tpu_defaults,
    tpu_server_env,
    tpu_server_static_args,
)


WORKLOADS = [
    {
        "name": "pr28_hotshape",
        "query_tokens": 2000,
        "num_items": 500,
        "item_tokens": 20,
        "label_token_ids": [198],
        "apply_softmax": False,
    },
    {
        "name": "small_batch",
        "query_tokens": 120,
        "num_items": 10,
        "item_tokens": 20,
        "label_token_ids": [198],
        "apply_softmax": False,
    },
    {
        "name": "medium_batch",
        "query_tokens": 500,
        "num_items": 100,
        "item_tokens": 20,
        "label_token_ids": [198],
        "apply_softmax": False,
    },
]
WORKLOAD_BY_NAME = {w["name"]: w for w in WORKLOADS}
WORKLOAD_ITEMS = {w["name"]: int(w["num_items"]) for w in WORKLOADS}
DEFAULT_P99_P50_GATES = {
    "pr28_hotshape": 3.5,
    "small_batch": 5.0,
    "medium_batch": 4.0,
}

SCORE_METRICS_RE = re.compile(
    r"ScorePathMetrics path=(\S+) items=(\d+) dispatches=(\d+) .*? "
    r"queue_wait_s=([0-9.]+) device_compute_s=([0-9.]+) host_orchestration_s=([0-9.]+)"
)
SCORE_FASTPATH_RE = re.compile(
    r"fastpath_attempted=(True|False)\s+"
    r"fastpath_succeeded=(True|False)\s+"
    r"fastpath_fallback_reason=([^\s]+)"
)
PREFILL_RE = re.compile(
    r"Prefill batch\. #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
    r"token usage: ([0-9.]+), #running-req: (\d+), #queue-req: (\d+)"
)
SERVER_LABEL_ONLY_RE = re.compile(r"multi_item_score_label_only_logprob=(True|False)")
SERVER_FASTPATH_METRICS_RE = re.compile(r"multi_item_score_fastpath_log_metrics=(True|False)")


@dataclass
class MatrixRun:
    items_per_step: int
    repeat_idx: int
    run_dir: Path
    raw_results: dict[str, Any]
    tpu_log_path: Path
    compile_signals: dict[str, Any] = field(default_factory=dict)
    shape_signals: dict[str, Any] = field(default_factory=dict)
    run_error: str | None = None


@dataclass
class GateConfig:
    max_failure_rate: float
    max_p99_p50: dict[str, float]
    max_throughput_cv: float
    max_p99_cv: float
    min_score_samples: int
    allow_missing_cache_handle: bool
    allow_timed_xla_compilation: bool
    allow_shape_contract_violations: bool
    allow_score_full_vocab_fallback: bool
    require_cache_transition_exercise: bool


@dataclass
class TunnelHandle:
    mode: str
    local_port: int
    remote_port: int
    proc: subprocess.Popen[str] | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def resolve_baseline_path(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE_PATH))
    args, _ = parser.parse_known_args(argv)
    return Path(args.baseline_config).expanduser().resolve()


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(p) for p in cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=capture_output,
        check=check,
        env=env,
    )


def run_cmd_retry(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
    attempts: int = 5,
    base_sleep_sec: float = 2.0,
) -> subprocess.CompletedProcess[str]:
    last_error: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return run_cmd(cmd, cwd=cwd, capture_output=capture_output, check=True)
        except subprocess.CalledProcessError as e:
            last_error = e
            if i == attempts:
                break
            sleep_s = base_sleep_sec * (2 ** (i - 1))
            print(
                f"[retry] command failed (attempt {i}/{attempts}, rc={e.returncode}); "
                f"sleeping {sleep_s:.1f}s before retry"
            )
            time.sleep(sleep_s)
    assert last_error is not None
    raise last_error


def run_direct_ssh_script(
    *,
    tpu_host: str,
    ssh_user: str,
    ssh_key: str,
    script: str,
    attempts: int = 5,
    base_sleep_sec: float = 2.0,
) -> None:
    cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        f"{ssh_user}@{tpu_host}",
        "bash -s",
    ]
    last_error: Exception | None = None
    for i in range(1, attempts + 1):
        print("+", " ".join(shlex.quote(p) for p in cmd), f"(stdin script, attempt {i}/{attempts})")
        try:
            subprocess.run(cmd, input=script, text=True, check=True)
            return
        except subprocess.CalledProcessError as e:
            last_error = e
            if i == attempts:
                break
            sleep_s = base_sleep_sec * (2 ** (i - 1))
            print(
                f"[retry] direct ssh script failed (attempt {i}/{attempts}, rc={e.returncode}); "
                f"sleeping {sleep_s:.1f}s before retry"
            )
            time.sleep(sleep_s)
    assert last_error is not None
    raise last_error


def gcloud_tpu_ssh_cmd(name: str, zone: str, command: str, project: str | None) -> list[str]:
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "ssh", name, f"--zone={zone}"]
    if project:
        cmd.append(f"--project={project}")
    cmd.extend(
        [
            "--ssh-flag=-o",
            "--ssh-flag=ConnectTimeout=10",
            "--ssh-flag=-o",
            "--ssh-flag=ConnectionAttempts=1",
        ]
    )
    cmd.extend(["--command", command])
    return cmd


def gcloud_tpu_scp_cmd(name: str, zone: str, remote: str, local: Path, project: str | None) -> list[str]:
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "scp", f"--zone={zone}"]
    if project:
        cmd.append(f"--project={project}")
    cmd.extend(
        [
            "--scp-flag=-o",
            "--scp-flag=ConnectTimeout=10",
            "--scp-flag=-o",
            "--scp-flag=ConnectionAttempts=1",
        ]
    )
    cmd.extend([f"{name}:{remote}", str(local)])
    return cmd


def gcloud_tpu_scp_upload_cmd(
    name: str,
    zone: str,
    local: Path,
    remote: str,
    project: str | None,
) -> list[str]:
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "scp", f"--zone={zone}"]
    if project:
        cmd.append(f"--project={project}")
    cmd.extend(
        [
            "--scp-flag=-o",
            "--scp-flag=ConnectTimeout=10",
            "--scp-flag=-o",
            "--scp-flag=ConnectionAttempts=1",
        ]
    )
    cmd.extend([str(local), f"{name}:{remote}"])
    return cmd


def resolve_tpu_external_ip(name: str, zone: str, project: str | None) -> str:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        name,
        f"--zone={zone}",
        "--format=value(networkEndpoints[0].accessConfig.externalIp)",
    ]
    if project:
        cmd.append(f"--project={project}")
    proc = run_cmd(cmd, capture_output=True, check=True)
    host = proc.stdout.strip()
    if not host:
        raise RuntimeError("Failed to resolve TPU external IP.")
    return host


def tpu_ssh_cmd(
    *,
    ssh_mode: str,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
    command: str,
) -> list[str]:
    if ssh_mode == "direct":
        if not tpu_host:
            raise ValueError("tpu_host is required for direct SSH mode.")
        return [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            f"{ssh_user}@{tpu_host}",
            command,
        ]
    return gcloud_tpu_ssh_cmd(tpu_name, tpu_zone, command, tpu_project)


def tpu_scp_cmd(
    *,
    ssh_mode: str,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
    remote_path: str,
    local_path: Path,
) -> list[str]:
    if ssh_mode == "direct":
        if not tpu_host:
            raise ValueError("tpu_host is required for direct SCP mode.")
        return [
            "scp",
            "-i",
            ssh_key,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            f"{ssh_user}@{tpu_host}:{remote_path}",
            str(local_path),
        ]
    return gcloud_tpu_scp_cmd(tpu_name, tpu_zone, remote_path, local_path, tpu_project)


def tpu_scp_upload_cmd(
    *,
    ssh_mode: str,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
    local_path: Path,
    remote_path: str,
) -> list[str]:
    if ssh_mode == "direct":
        if not tpu_host:
            raise ValueError("tpu_host is required for direct SCP upload mode.")
        return [
            "scp",
            "-i",
            ssh_key,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            str(local_path),
            f"{ssh_user}@{tpu_host}:{remote_path}",
        ]
    return gcloud_tpu_scp_upload_cmd(tpu_name, tpu_zone, local_path, remote_path, tpu_project)


def parse_base_url(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported TPU URL scheme in '{base_url}' (expected http/https).")
    host = parsed.hostname
    if not host:
        raise ValueError(f"TPU URL missing hostname: '{base_url}'")
    if parsed.port is not None:
        return host, int(parsed.port)
    return host, 443 if parsed.scheme == "https" else 80


def is_loopback_host(host: str) -> bool:
    h = host.strip().lower()
    return h in {"localhost", "::1"} or h.startswith("127.")


def resolve_connection_mode(base_url: str, requested: str) -> str:
    host, _ = parse_base_url(base_url)
    inferred = "tunnel" if is_loopback_host(host) else "direct"
    if requested == "auto":
        return inferred
    if requested == "tunnel" and not is_loopback_host(host):
        raise ValueError(
            f"--tpu-connection-mode=tunnel requires loopback TPU URL host (got '{host}' from {base_url})."
        )
    if requested == "direct" and is_loopback_host(host):
        raise ValueError(
            f"--tpu-connection-mode=direct requires non-loopback TPU URL host (got '{host}' from {base_url})."
        )
    return requested


def local_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def wait_for_local_port(port: int, timeout_sec: float = 10.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if local_port_open(port):
            return True
        time.sleep(0.25)
    return local_port_open(port)


def build_tpu_tunnel_cmd(
    *,
    ssh_mode: str,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
    local_port: int,
    remote_port: int,
) -> list[str]:
    if ssh_mode == "direct":
        if not tpu_host:
            raise ValueError("tpu_host is required for direct tunnel mode.")
        return [
            "ssh",
            "-i",
            ssh_key,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-N",
            "-L",
            f"{local_port}:127.0.0.1:{remote_port}",
            f"{ssh_user}@{tpu_host}",
        ]
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_name, f"--zone={tpu_zone}"]
    if tpu_project:
        cmd.append(f"--project={tpu_project}")
    cmd.extend(
        [
            "--ssh-flag=-o",
            "--ssh-flag=ExitOnForwardFailure=yes",
            "--ssh-flag=-o",
            "--ssh-flag=ConnectTimeout=10",
            "--ssh-flag=-o",
            "--ssh-flag=ServerAliveInterval=30",
            "--ssh-flag=-o",
            "--ssh-flag=ServerAliveCountMax=3",
            "--ssh-flag=-N",
            "--ssh-flag=-L",
            f"--ssh-flag={local_port}:127.0.0.1:{remote_port}",
        ]
    )
    return cmd


def stop_tpu_tunnel(handle: TunnelHandle) -> None:
    if not handle.is_running():
        return
    assert handle.proc is not None
    handle.proc.terminate()
    try:
        handle.proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        handle.proc.kill()
        handle.proc.wait(timeout=5)
    handle.proc = None


def start_tpu_tunnel(
    *,
    handle: TunnelHandle,
    ssh_mode: str,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
) -> None:
    if local_port_open(handle.local_port):
        # Existing local tunnel/process is already bound to this port.
        return
    cmd = build_tpu_tunnel_cmd(
        ssh_mode=ssh_mode,
        tpu_name=tpu_name,
        tpu_zone=tpu_zone,
        tpu_project=tpu_project,
        tpu_host=tpu_host,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        local_port=handle.local_port,
        remote_port=handle.remote_port,
    )
    print("+", " ".join(shlex.quote(p) for p in cmd))
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if wait_for_local_port(handle.local_port, timeout_sec=12.0):
        handle.proc = proc
        return
    stdout, stderr = proc.communicate(timeout=1) if proc.poll() is not None else ("", "")
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    raise RuntimeError(
        "Failed to establish TPU tunnel "
        f"(local_port={handle.local_port}, remote_port={handle.remote_port}). "
        f"stdout={stdout[-300:]} stderr={stderr[-300:]}"
    )


def ensure_tpu_tunnel(
    *,
    handle: TunnelHandle | None,
    autostart: bool,
    ssh_mode: str,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
) -> None:
    if handle is None:
        return
    if local_port_open(handle.local_port):
        return
    if not autostart:
        raise RuntimeError(
            f"TPU tunnel mode selected but local port {handle.local_port} is not reachable. "
            "Enable --tpu-tunnel-autostart or start the tunnel manually."
        )
    if handle.is_running():
        stop_tpu_tunnel(handle)
    start_tpu_tunnel(
        handle=handle,
        ssh_mode=ssh_mode,
        tpu_name=tpu_name,
        tpu_zone=tpu_zone,
        tpu_project=tpu_project,
        tpu_host=tpu_host,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
    )


def precheck_tpu_endpoint(
    *,
    base_url: str,
    model: str,
    health_timeout_sec: int,
    score_timeout_sec: int,
    attempts: int,
    backoff_sec: float,
    tunnel_handle: TunnelHandle | None,
    tunnel_autostart: bool,
    ssh_mode: str,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
) -> None:
    attempts = max(1, int(attempts))
    backoff_sec = max(0.0, float(backoff_sec))
    last_err: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            ensure_tpu_tunnel(
                handle=tunnel_handle,
                autostart=tunnel_autostart,
                ssh_mode=ssh_mode,
                tpu_name=tpu_name,
                tpu_zone=tpu_zone,
                tpu_project=tpu_project,
                tpu_host=tpu_host,
                ssh_user=ssh_user,
                ssh_key=ssh_key,
            )
            wait_for_health(base_url, timeout_sec=health_timeout_sec)
            wait_for_score_ready(base_url, model, timeout_sec=score_timeout_sec)
            return
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            if attempt < attempts:
                print(
                    f"[precheck] TPU precheck failed (attempt {attempt}/{attempts}): {last_err}"
                )
                if tunnel_handle is not None and tunnel_autostart:
                    try:
                        stop_tpu_tunnel(tunnel_handle)
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(backoff_sec * attempt)
    raise RuntimeError(
        f"TPU precheck failed after {attempts} attempt(s): {last_err}"
    )


def wait_for_health(base_url: str, timeout_sec: int = 240) -> None:
    import urllib.request

    deadline = time.time() + timeout_sec
    url = base_url.rstrip("/") + "/v1/models"
    last_err = "unknown"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(2)
    raise RuntimeError(f"TPU endpoint did not become healthy within {timeout_sec}s: {last_err}")


def wait_for_score_ready(base_url: str, model: str, timeout_sec: int = 600) -> None:
    import json as _json
    import urllib.request

    payload = {
        "model": model,
        "query": "hello ",
        "items": ["hello "],
        "label_token_ids": [198],
        "apply_softmax": False,
    }
    body = _json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/v1/score"

    deadline = time.time() + timeout_sec
    last_err = "unknown"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=20) as resp:
                txt = resp.read().decode("utf-8", errors="ignore")
                if resp.status == 200 and ("scores" in txt or "logprobs" in txt):
                    return
                last_err = f"HTTP {resp.status}: {txt[:120]}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(3)
    raise RuntimeError(
        f"TPU score endpoint not ready within {timeout_sec}s (last_error={last_err})"
    )


def _build_warmup_payload(workload: dict[str, Any], model: str) -> dict[str, Any]:
    """Build warmup payload from the same shape-contract path used by timed runs."""
    from profiler.config import WorkloadConfig
    from profiler.workload import build_score_request_with_shape_contract

    wl = WorkloadConfig(
        name=str(workload["name"]),
        query_tokens=int(workload["query_tokens"]),
        num_items=int(workload["num_items"]),
        item_tokens=int(workload["item_tokens"]),
        label_token_ids=[int(x) for x in workload.get("label_token_ids", [198])],
        apply_softmax=bool(workload.get("apply_softmax", False)),
        use_token_ids=bool(workload.get("use_token_ids", False)),
        enforce_shape_contract=bool(workload.get("enforce_shape_contract", False)),
        query_token_buckets=[int(x) for x in workload.get("query_token_buckets", [])],
        item_token_buckets=[int(x) for x in workload.get("item_token_buckets", [])],
        num_items_buckets=[int(x) for x in workload.get("num_items_buckets", [])],
        pad_token_id=int(workload.get("pad_token_id", 0)),
        query_fill_token_id=int(workload.get("query_fill_token_id", 42)),
        item_fill_token_id=int(workload.get("item_fill_token_id", 84)),
    )
    payload, _ = build_score_request_with_shape_contract(wl, model)
    return payload


def _normalized_bucket_values(raw: Any, fallback: int) -> list[int]:
    if isinstance(raw, list):
        out = sorted({int(x) for x in raw if int(x) > 0})
        if out:
            return out
    return [int(fallback)]


def _pick_bucket_value(logical: int, buckets: list[int]) -> int:
    logical_i = int(logical)
    for bucket in buckets:
        if logical_i <= int(bucket):
            return int(bucket)
    return logical_i


def _shape_warmup_entry(
    *,
    base_name: str,
    query_tokens: int,
    num_items: int,
    item_tokens: int,
    label_ids: list[int],
    apply_softmax: bool,
    pad_token_id: int,
    query_fill_token_id: int,
    item_fill_token_id: int,
) -> dict[str, Any]:
    return {
        "name": base_name,
        "query_tokens": int(query_tokens),
        "num_items": int(num_items),
        "item_tokens": int(item_tokens),
        "label_token_ids": list(label_ids),
        "apply_softmax": bool(apply_softmax),
        "use_token_ids": True,
        "enforce_shape_contract": True,
        "query_token_buckets": [int(query_tokens)],
        "item_token_buckets": [int(item_tokens)],
        "num_items_buckets": [int(num_items)],
        "pad_token_id": int(pad_token_id),
        "query_fill_token_id": int(query_fill_token_id),
        "item_fill_token_id": int(item_fill_token_id),
    }


def build_shape_bucket_warmup_plan(
    workloads: list[dict[str, Any]],
    *,
    max_requests: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Build a bounded warmup plan: active shapes first, then bucket sweeps."""
    limit = max(1, int(max_requests))
    plan: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    truncated = False

    def _append(entry: dict[str, Any], key: tuple[Any, ...]) -> bool:
        nonlocal truncated
        if key in seen:
            return True
        if len(plan) >= limit:
            truncated = True
            return False
        seen.add(key)
        plan.append(entry)
        return True

    shape_profiles: list[dict[str, Any]] = []
    for wl in workloads:
        shape_enabled = bool(wl.get("enforce_shape_contract", False)) and bool(
            wl.get("use_token_ids", False)
        )
        label_ids = [int(x) for x in wl.get("label_token_ids", [198])]
        apply_softmax = bool(wl.get("apply_softmax", False))
        pad_token_id = int(wl.get("pad_token_id", 0))
        query_fill_token_id = int(wl.get("query_fill_token_id", 42))
        item_fill_token_id = int(wl.get("item_fill_token_id", 84))

        if not shape_enabled:
            key = (
                "direct",
                int(wl["query_tokens"]),
                int(wl["num_items"]),
                int(wl["item_tokens"]),
                tuple(label_ids),
                apply_softmax,
            )
            if not _append(dict(wl), key):
                break
            continue

        query_buckets = _normalized_bucket_values(wl.get("query_token_buckets"), int(wl["query_tokens"]))
        num_items_buckets = _normalized_bucket_values(wl.get("num_items_buckets"), int(wl["num_items"]))
        item_buckets = _normalized_bucket_values(wl.get("item_token_buckets"), int(wl["item_tokens"]))
        active_query = _pick_bucket_value(int(wl["query_tokens"]), query_buckets)
        active_num_items = _pick_bucket_value(int(wl["num_items"]), num_items_buckets)
        active_item_tokens = _pick_bucket_value(int(wl["item_tokens"]), item_buckets)

        active_entry = _shape_warmup_entry(
            base_name=f"{wl['name']}__warmup_active_q{active_query}_n{active_num_items}_i{active_item_tokens}",
            query_tokens=active_query,
            num_items=active_num_items,
            item_tokens=active_item_tokens,
            label_ids=label_ids,
            apply_softmax=apply_softmax,
            pad_token_id=pad_token_id,
            query_fill_token_id=query_fill_token_id,
            item_fill_token_id=item_fill_token_id,
        )
        active_key = (
            active_query,
            active_num_items,
            active_item_tokens,
            tuple(label_ids),
            apply_softmax,
            pad_token_id,
            query_fill_token_id,
            item_fill_token_id,
        )
        if not _append(active_entry, active_key):
            break

        shape_profiles.append(
            {
                "name": str(wl["name"]),
                "query_buckets": query_buckets,
                "num_items_buckets": num_items_buckets,
                "item_buckets": item_buckets,
                "active_query": active_query,
                "active_num_items": active_num_items,
                "active_item_tokens": active_item_tokens,
                "label_ids": label_ids,
                "apply_softmax": apply_softmax,
                "pad_token_id": pad_token_id,
                "query_fill_token_id": query_fill_token_id,
                "item_fill_token_id": item_fill_token_id,
            }
        )

    if truncated or not shape_profiles:
        return plan, truncated

    # Prioritize linear bucket sweeps around active shapes instead of cubic expansion.
    query_union: list[int] = sorted(
        {
            bucket
            for profile in shape_profiles
            for bucket in profile["query_buckets"]
        }
    )
    num_items_union: list[int] = sorted(
        {
            bucket
            for profile in shape_profiles
            for bucket in profile["num_items_buckets"]
        }
    )
    item_union: list[int] = sorted(
        {
            bucket
            for profile in shape_profiles
            for bucket in profile["item_buckets"]
        }
    )

    def _representative(profile: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(profile["active_query"]),
            int(profile["active_num_items"]),
            int(profile["active_item_tokens"]),
        )

    rep_query = max(_representative(profile)[0] for profile in shape_profiles)
    rep_num_items = max(_representative(profile)[1] for profile in shape_profiles)
    rep_item_tokens = max(_representative(profile)[2] for profile in shape_profiles)
    rep_profile = max(shape_profiles, key=lambda p: int(p["active_num_items"]))
    rep_label_ids = [int(x) for x in rep_profile["label_ids"]]
    rep_apply_softmax = bool(rep_profile["apply_softmax"])
    rep_pad_token_id = int(rep_profile["pad_token_id"])
    rep_query_fill_token_id = int(rep_profile["query_fill_token_id"])
    rep_item_fill_token_id = int(rep_profile["item_fill_token_id"])

    for query_tokens in query_union:
        entry = _shape_warmup_entry(
            base_name=f"shape_warmup_query_q{query_tokens}_n{rep_num_items}_i{rep_item_tokens}",
            query_tokens=query_tokens,
            num_items=rep_num_items,
            item_tokens=rep_item_tokens,
            label_ids=rep_label_ids,
            apply_softmax=rep_apply_softmax,
            pad_token_id=rep_pad_token_id,
            query_fill_token_id=rep_query_fill_token_id,
            item_fill_token_id=rep_item_fill_token_id,
        )
        key = (
            query_tokens,
            rep_num_items,
            rep_item_tokens,
            tuple(rep_label_ids),
            rep_apply_softmax,
            rep_pad_token_id,
            rep_query_fill_token_id,
            rep_item_fill_token_id,
        )
        if not _append(entry, key):
            break
    if truncated:
        return plan, truncated

    for num_items in num_items_union:
        entry = _shape_warmup_entry(
            base_name=f"shape_warmup_items_q{rep_query}_n{num_items}_i{rep_item_tokens}",
            query_tokens=rep_query,
            num_items=num_items,
            item_tokens=rep_item_tokens,
            label_ids=rep_label_ids,
            apply_softmax=rep_apply_softmax,
            pad_token_id=rep_pad_token_id,
            query_fill_token_id=rep_query_fill_token_id,
            item_fill_token_id=rep_item_fill_token_id,
        )
        key = (
            rep_query,
            num_items,
            rep_item_tokens,
            tuple(rep_label_ids),
            rep_apply_softmax,
            rep_pad_token_id,
            rep_query_fill_token_id,
            rep_item_fill_token_id,
        )
        if not _append(entry, key):
            break
    if truncated:
        return plan, truncated

    for item_tokens in item_union:
        entry = _shape_warmup_entry(
            base_name=f"shape_warmup_item_tokens_q{rep_query}_n{rep_num_items}_i{item_tokens}",
            query_tokens=rep_query,
            num_items=rep_num_items,
            item_tokens=item_tokens,
            label_ids=rep_label_ids,
            apply_softmax=rep_apply_softmax,
            pad_token_id=rep_pad_token_id,
            query_fill_token_id=rep_query_fill_token_id,
            item_fill_token_id=rep_item_fill_token_id,
        )
        key = (
            rep_query,
            rep_num_items,
            item_tokens,
            tuple(rep_label_ids),
            rep_apply_softmax,
            rep_pad_token_id,
            rep_query_fill_token_id,
            rep_item_fill_token_id,
        )
        if not _append(entry, key):
            break

    return plan, truncated


def run_workload_shape_warmup(
    *,
    base_url: str,
    model: str,
    workloads: list[dict[str, Any]],
    warmup_all_shape_buckets: bool = True,
    max_shape_bucket_requests: int = 64,
    timeout_sec: int = 300,
    attempts: int = 3,
    backoff_sec: float = 3.0,
    fail_on_error: bool = False,
) -> None:
    import json as _json
    import urllib.request

    url = base_url.rstrip("/") + "/v1/score"
    headers = {"Content-Type": "application/json"}
    attempts = max(1, int(attempts))
    if warmup_all_shape_buckets:
        warmup_plan, truncated = build_shape_bucket_warmup_plan(
            workloads,
            max_requests=max_shape_bucket_requests,
        )
    else:
        warmup_plan = [dict(w) for w in workloads]
        truncated = False

    print(
        f"[warmup] issuing {len(warmup_plan)} shape warmup request(s) "
        f"(expanded={str(warmup_all_shape_buckets).lower()}, "
        f"max={max_shape_bucket_requests}, truncated={str(truncated).lower()})"
    )

    for wl in warmup_plan:
        payload = _build_warmup_payload(wl, model)
        body = _json.dumps(payload).encode("utf-8")
        last_err: str | None = None
        for attempt_idx in range(1, attempts + 1):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    txt = resp.read().decode("utf-8", errors="ignore")
                    if resp.status != 200:
                        raise RuntimeError(
                            f"HTTP={resp.status}: {txt[:200]}"
                        )
                    if "scores" not in txt and "logprobs" not in txt:
                        raise RuntimeError(f"missing score fields: {txt[:200]}")
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                if attempt_idx < attempts:
                    time.sleep(max(0.0, backoff_sec))
        if last_err is not None:
            msg = (
                f"Warmup request failed for workload={wl['name']} after {attempts} attempt(s): "
                f"{last_err}"
            )
            if fail_on_error:
                raise RuntimeError(msg)
            print(f"[warn] {msg}; continuing")


def launch_tpu_server(
    *,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    ssh_mode: str,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
    repo_path: str,
    port: int,
    model: str,
    items_per_step: int,
    max_running_requests: int,
    multi_item_extend_batch_size: int,
    multi_item_scoring_chunk_size: int,
    precompile_token_paddings: list[int],
    precompile_bs_paddings: list[int],
    remote_log_path: str,
    server_env: dict[str, str],
    server_static_args: list[str],
) -> None:
    env_precompile_bs = ",".join(str(x) for x in precompile_bs_paddings)
    env_lines = "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in sorted(server_env.items())
    )
    launch_cmd = [
        ".venv/bin/python",
        "-m",
        "sgl_jax.launch_server",
        "--model-path",
        model,
        "--port",
        str(port),
        *server_static_args,
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
    launch_cmd_str = " ".join(shlex.quote(token) for token in launch_cmd)
    script = f"""
set -euo pipefail
cd {shlex.quote(repo_path)}
if pgrep -f "sgl_jax.launch_server.*--port {port}" >/dev/null; then
  pkill -f "sgl_jax.launch_server.*--port {port}" || true
  sleep 2
fi
{env_lines}
export MULTI_ITEM_EXTEND_BATCH_SIZE={multi_item_extend_batch_size}
export MULTI_ITEM_EXTEND_MAX_RUNNING_REQUESTS={max_running_requests}
export MULTI_ITEM_SCORE_FROM_CACHE_V2_ITEMS_PER_STEP={items_per_step}
export MULTI_ITEM_EXTEND_PRECOMPILE_BS_PADDINGS={env_precompile_bs}
nohup {launch_cmd_str} > {shlex.quote(remote_log_path)} 2>&1 &
echo started_pid=$!
"""
    if ssh_mode == "direct":
        assert tpu_host is not None
        run_direct_ssh_script(
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            script=script,
        )
    else:
        cmd = tpu_ssh_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            command=f"bash -lc {shlex.quote(script)}",
        )
        run_cmd_retry(cmd)


def stop_tpu_server(
    *,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    ssh_mode: str,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
    port: int,
) -> None:
    script = f"""
set -euo pipefail
if pgrep -f "sgl_jax.launch_server.*--port {port}" >/dev/null; then
  pkill -f "sgl_jax.launch_server.*--port {port}" || true
  sleep 2
fi
"""
    if ssh_mode == "direct":
        assert tpu_host is not None
        run_direct_ssh_script(
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            script=script,
        )
    else:
        cmd = tpu_ssh_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            command=f"bash -lc {shlex.quote(script)}",
        )
        run_cmd_retry(cmd)


def write_config(
    *,
    path: Path,
    experiment_name: str,
    model: str,
    tpu_url: str,
    hardware: str,
    cost_per_hour: float,
    warmup_requests: int,
    timed_requests: int,
    concurrency: int,
    timeout_sec: int,
    request_retry_attempts: int,
    request_retry_backoff_sec: float,
    workloads: list[dict[str, Any]],
) -> None:
    cfg = {
        "experiment": {"name": experiment_name, "model": model},
        "backends": {
            "tpu": {
                "name": "JAX/TPU matrix run",
                "url": tpu_url,
                "hardware": hardware,
                "cost_per_hour": cost_per_hour,
            }
        },
        "api": {"endpoint": "/v1/score"},
        "workloads": workloads,
        "benchmark": {
            "warmup_requests": warmup_requests,
            "timed_requests": timed_requests,
            "concurrency": concurrency,
            "timeout_sec": timeout_sec,
            "request_retry_attempts": request_retry_attempts,
            "request_retry_backoff_sec": request_retry_backoff_sec,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def parse_results_dir_text(stdout: str) -> str:
    m = re.search(r"Results directory:\s*(.+)", stdout)
    if not m:
        raise RuntimeError("Could not parse 'Results directory' from profiler output")
    return m.group(1).strip()


def parse_results_dir(stdout: str) -> Path:
    return Path(parse_results_dir_text(stdout)).resolve()


def _extract_compile_signals(run_dir: Path, profiler_stdout: str) -> dict[str, Any]:
    stdout_markers = len(re.findall(r"XLA compilation detected", profiler_stdout))
    signals: dict[str, Any] = {
        # Timed-phase instability signal (first timed request is >3x median of remaining timed requests).
        "timed_xla_compile_count": 0,
        "timed_xla_compile_workloads": [],
        "timed_xla_compile_requests": [],
        # Warmup-phase signal (same heuristic applied on warmup requests).
        "warmup_xla_detect_count": 0,
        "warmup_xla_detect_workloads": [],
        # Backward-compatible fallback signal from plain profiler stdout.
        "profiler_stdout_xla_marker_count": stdout_markers,
    }

    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        if stdout_markers > 0:
            signals["warmup_xla_detect_count"] = stdout_markers
        return signals

    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:  # noqa: BLE001
        if stdout_markers > 0:
            signals["warmup_xla_detect_count"] = stdout_markers
        return signals

    run_diags = metadata.get("run_diagnostics", {})
    tpu_diags: dict[str, Any] | None = None
    if isinstance(run_diags, dict):
        candidate = run_diags.get("tpu")
        if isinstance(candidate, dict):
            tpu_diags = candidate
        else:
            for value in run_diags.values():
                if isinstance(value, dict):
                    tpu_diags = value
                    break

    if not isinstance(tpu_diags, dict):
        if stdout_markers > 0:
            signals["warmup_xla_detect_count"] = stdout_markers
        return signals

    timed_workloads: list[str] = []
    timed_request_markers: list[str] = []
    timed_compile_events = 0
    warmup_workloads: list[str] = []
    for wl_name, diag_raw in tpu_diags.items():
        if not isinstance(diag_raw, dict):
            continue
        if bool(diag_raw.get("warmup_xla_detected", False)):
            warmup_workloads.append(str(wl_name))
        timed_compile_count = int(diag_raw.get("timed_compile_like_count", 0) or 0)
        if timed_compile_count <= 0 and bool(diag_raw.get("timed_first_outlier", False)):
            timed_compile_count = 1
        if timed_compile_count > 0:
            timed_workloads.append(str(wl_name))
            timed_compile_events += timed_compile_count
            request_ids = diag_raw.get("timed_compile_like_request_ids", [])
            request_id_list = (
                [int(x) for x in request_ids if isinstance(x, (int, float))]
                if isinstance(request_ids, list)
                else []
            )
            if request_id_list:
                timed_request_markers.extend(
                    f"{wl_name}:{request_id}" for request_id in sorted(set(request_id_list))
                )
            else:
                timed_request_markers.append(f"{wl_name}:?")

    warmup_workloads.sort()
    timed_workloads.sort()
    timed_request_markers.sort()
    signals["warmup_xla_detect_workloads"] = warmup_workloads
    signals["timed_xla_compile_workloads"] = timed_workloads
    signals["timed_xla_compile_requests"] = timed_request_markers
    signals["warmup_xla_detect_count"] = len(warmup_workloads)
    signals["timed_xla_compile_count"] = timed_compile_events
    if signals["warmup_xla_detect_count"] == 0 and stdout_markers > 0:
        signals["warmup_xla_detect_count"] = stdout_markers
    return signals


def run_profiler_for_config(
    profiler_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    client_location: str,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    ssh_mode: str,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
    tpu_repo_path: str,
    tpu_client_runtime_dir: str,
    tpu_client_python: str | None,
) -> tuple[Path, dict[str, Any]]:
    if client_location == "local":
        py_env = dict(os.environ)
        existing_pythonpath = py_env.get("PYTHONPATH", "")
        py_env["PYTHONPATH"] = (
            f"{profiler_dir}:{existing_pythonpath}"
            if existing_pythonpath
            else str(profiler_dir)
        )
        cmd = [
            "python3",
            "-m",
            "profiler.cli",
            "run",
            "-c",
            str(config_path),
            "-b",
            "tpu",
            "-o",
            str(output_dir),
        ]
        proc = run_cmd(
            cmd,
            cwd=profiler_dir,
            capture_output=True,
            check=True,
            env=py_env,
        )
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        run_dir = parse_results_dir(proc.stdout)
        profiler_artifacts_dir = run_dir / "artifacts" / "profiler"
        profiler_artifacts_dir.mkdir(parents=True, exist_ok=True)
        (profiler_artifacts_dir / "profiler_stdout.log").write_text(proc.stdout or "")
        (profiler_artifacts_dir / "profiler_stderr.log").write_text(proc.stderr or "")
        compile_signals = _extract_compile_signals(run_dir, proc.stdout or "")
        return run_dir, compile_signals

    if client_location != "tpu_vm":
        raise ValueError(f"Unsupported --tpu-client-location: {client_location}")

    runtime_dir = tpu_client_runtime_dir.rstrip("/")
    remote_cfg = f"{runtime_dir}/configs/{config_path.name}"
    remote_output_base = f"{runtime_dir}/runs"
    remote_python = (
        tpu_client_python.strip()
        if isinstance(tpu_client_python, str) and tpu_client_python.strip()
        else f"{tpu_repo_path.rstrip('/')}/.venv/bin/python"
    )

    run_cmd_retry(
        tpu_scp_upload_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            local_path=config_path,
            remote_path=remote_cfg,
        )
    )

    remote_run_script = f"""
set -euo pipefail
mkdir -p {shlex.quote(runtime_dir)}/configs {shlex.quote(runtime_dir)}/runs
PYTHONPATH={shlex.quote(runtime_dir)} {shlex.quote(remote_python)} -m profiler.cli run -c {shlex.quote(remote_cfg)} -b tpu -o {shlex.quote(remote_output_base)}
"""
    remote_proc = run_cmd_retry(
        tpu_ssh_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            command=f"bash -lc {shlex.quote(remote_run_script)}",
        ),
        capture_output=True,
    )
    if remote_proc.stdout:
        print(remote_proc.stdout)
    if remote_proc.stderr:
        print(remote_proc.stderr, file=sys.stderr)

    remote_run_dir = parse_results_dir_text(remote_proc.stdout or "")
    remote_run_name = Path(remote_run_dir).name
    remote_tar = f"{runtime_dir}/artifacts/{remote_run_name}.tar.gz"
    local_tar = output_dir / f"{remote_run_name}.tar.gz"

    pack_script = f"""
set -euo pipefail
mkdir -p {shlex.quote(runtime_dir)}/artifacts
tar -C {shlex.quote(str(Path(remote_run_dir).parent))} -czf {shlex.quote(remote_tar)} {shlex.quote(remote_run_name)}
"""
    run_cmd_retry(
        tpu_ssh_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            command=f"bash -lc {shlex.quote(pack_script)}",
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd_retry(
        tpu_scp_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            remote_path=remote_tar,
            local_path=local_tar,
        )
    )
    run_cmd(["tar", "-xzf", str(local_tar), "-C", str(output_dir)])
    if local_tar.exists():
        local_tar.unlink()

    run_cmd_retry(
        tpu_ssh_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            command=f"bash -lc {shlex.quote(f'rm -f {shlex.quote(remote_tar)}')}",
        )
    )

    run_dir = (output_dir / remote_run_name).resolve()
    if not run_dir.exists():
        raise RuntimeError(f"Expected downloaded run directory not found: {run_dir}")

    profiler_artifacts_dir = run_dir / "artifacts" / "profiler"
    profiler_artifacts_dir.mkdir(parents=True, exist_ok=True)
    (profiler_artifacts_dir / "profiler_stdout.log").write_text(remote_proc.stdout or "")
    (profiler_artifacts_dir / "profiler_stderr.log").write_text(remote_proc.stderr or "")
    compile_signals = _extract_compile_signals(run_dir, remote_proc.stdout or "")
    return run_dir, compile_signals


def prepare_tpu_vm_profiler_runtime(
    *,
    profiler_dir: Path,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    ssh_mode: str,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
    tpu_client_runtime_dir: str,
) -> None:
    runtime_dir = tpu_client_runtime_dir.rstrip("/")
    if not runtime_dir:
        raise ValueError("--tpu-client-runtime-dir must not be empty")

    mkdir_script = f"mkdir -p {shlex.quote(runtime_dir)}"
    run_cmd_retry(
        tpu_ssh_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            command=f"bash -lc {shlex.quote(mkdir_script)}",
        )
    )

    with tempfile.TemporaryDirectory(prefix="tpu_profiler_runtime_") as tmp_dir:
        tar_path = Path(tmp_dir) / "profiler_runtime.tgz"
        run_cmd(
            [
                "tar",
                "-czf",
                str(tar_path),
                "-C",
                str(profiler_dir),
                "profiler",
            ]
        )
        remote_tar_path = f"{runtime_dir}/profiler_runtime.tgz"
        run_cmd_retry(
            tpu_scp_upload_cmd(
                ssh_mode=ssh_mode,
                tpu_name=tpu_name,
                tpu_zone=tpu_zone,
                tpu_project=tpu_project,
                tpu_host=tpu_host,
                ssh_user=ssh_user,
                ssh_key=ssh_key,
                local_path=tar_path,
                remote_path=remote_tar_path,
            )
        )

    unpack_script = f"""
set -euo pipefail
mkdir -p {shlex.quote(runtime_dir)}
rm -rf {shlex.quote(runtime_dir)}/profiler
tar -xzf {shlex.quote(remote_tar_path)} -C {shlex.quote(runtime_dir)}
rm -f {shlex.quote(remote_tar_path)}
mkdir -p {shlex.quote(runtime_dir)}/configs {shlex.quote(runtime_dir)}/runs {shlex.quote(runtime_dir)}/artifacts
"""
    run_cmd_retry(
        tpu_ssh_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            command=f"bash -lc {shlex.quote(unpack_script)}",
        )
    )


def collect_tpu_log(
    *,
    tpu_name: str,
    tpu_zone: str,
    tpu_project: str | None,
    ssh_mode: str,
    tpu_host: str | None,
    ssh_user: str,
    ssh_key: str,
    remote_log_path: str,
    run_dir: Path,
    start_offset_bytes: int = 0,
) -> tuple[Path, int]:
    local_log_dir = run_dir / "artifacts" / "tpu"
    local_log_dir.mkdir(parents=True, exist_ok=True)
    local_log_full = local_log_dir / "tpu_server_full.log"
    local_log = local_log_dir / "tpu_server.log"
    run_cmd_retry(
        tpu_scp_cmd(
            ssh_mode=ssh_mode,
            tpu_name=tpu_name,
            tpu_zone=tpu_zone,
            tpu_project=tpu_project,
            tpu_host=tpu_host,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            remote_path=remote_log_path,
            local_path=local_log_full,
        )
    )
    data = local_log_full.read_bytes()
    total_size = len(data)
    if start_offset_bytes <= 0:
        sliced = data
    elif start_offset_bytes >= total_size:
        sliced = b""
    else:
        sliced = data[start_offset_bytes:]
    local_log.write_bytes(sliced)
    return local_log, total_size


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    return statistics.median(vals)


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return statistics.mean(vals)


def _cv(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    mean_v = statistics.mean(vals)
    if mean_v == 0:
        return 0.0
    return statistics.pstdev(vals) / mean_v


def _fmt(v: float | None, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _mode_int(vals: list[int]) -> int | None:
    if not vals:
        return None
    counts: dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts.keys()), key=lambda k: counts[k])


def _mode_bool(vals: list[bool]) -> bool | None:
    if not vals:
        return None
    true_count = sum(1 for v in vals if bool(v))
    false_count = len(vals) - true_count
    return true_count >= false_count


def _parse_optional_server_bool(log_text: str, pattern: re.Pattern[str]) -> bool | None:
    matches = pattern.findall(log_text)
    if not matches:
        return None
    vals = [str(m).strip() == "True" for m in matches]
    return _mode_bool(vals)


def parse_p99_p50_gates(raw: str) -> dict[str, float]:
    out = dict(DEFAULT_P99_P50_GATES)
    if not raw.strip():
        return out
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid --gate-max-p99-p50 token '{token}'. Expected '<workload>:<float>'."
            )
        name, value = token.split(":", 1)
        wl_name = name.strip()
        if wl_name not in WORKLOAD_ITEMS:
            raise ValueError(
                f"Unknown workload in --gate-max-p99-p50: '{wl_name}'. "
                f"Allowed={sorted(WORKLOAD_ITEMS.keys())}"
            )
        out[wl_name] = float(value.strip())
    return out


def parse_int_csv(raw: str, *, name: str) -> list[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"--{name} must contain at least one integer value.")
    try:
        out = [int(v) for v in vals]
    except ValueError as e:
        raise ValueError(f"Invalid integer in --{name}: {raw}") from e
    return out


def maybe_align_items_per_step_candidates(
    *,
    candidates: list[int],
    workloads: list[dict[str, Any]],
    enabled: bool,
) -> list[int]:
    if not enabled:
        return sorted({int(x) for x in candidates if int(x) > 0})
    if len(workloads) != 1:
        return sorted({int(x) for x in candidates if int(x) > 0})
    if not candidates:
        return []

    unique = sorted({int(x) for x in candidates if int(x) > 0})
    if not unique:
        return []
    wl_items = int(workloads[0]["num_items"])
    max_candidate = max(unique)
    if 0 < wl_items <= (max_candidate * 2):
        unique = sorted(set(unique + [wl_items]))
    return unique


def maybe_align_lane_capacity_for_single_workload(
    *,
    max_running_requests: int,
    multi_item_extend_batch_size: int,
    workloads: list[dict[str, Any]],
    enabled: bool,
    cap: int,
) -> tuple[int, int]:
    if not enabled or len(workloads) != 1:
        return int(max_running_requests), int(multi_item_extend_batch_size)
    target = int(workloads[0]["num_items"])
    target = max(1, min(target, int(cap)))
    return (
        max(int(max_running_requests), target),
        max(int(multi_item_extend_batch_size), target),
    )


def parse_workload_filter(raw: str | None) -> list[dict[str, Any]]:
    if raw is None or not raw.strip():
        return list(WORKLOADS)
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("--workload-filter parsed to empty list.")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in names:
        if name not in WORKLOAD_BY_NAME:
            raise ValueError(
                f"Unknown workload '{name}' in --workload-filter. "
                f"Allowed={sorted(WORKLOAD_BY_NAME.keys())}"
            )
        if name in seen:
            continue
        seen.add(name)
        selected.append(dict(WORKLOAD_BY_NAME[name]))
    return selected


def apply_shape_contract_to_workloads(
    *,
    workloads: list[dict[str, Any]],
    enabled: bool,
    use_token_ids: bool,
    strict: bool,
    query_token_buckets: list[int],
    item_token_buckets: list[int],
    num_items_buckets: list[int],
    pad_token_id: int,
    query_fill_token_id: int,
    item_fill_token_id: int,
) -> list[dict[str, Any]]:
    if not enabled:
        return [dict(w) for w in workloads]
    out: list[dict[str, Any]] = []
    for wl in workloads:
        w = dict(wl)
        w["use_token_ids"] = bool(use_token_ids)
        w["enforce_shape_contract"] = bool(strict)
        w["query_token_buckets"] = list(query_token_buckets)
        w["item_token_buckets"] = list(item_token_buckets)
        w["num_items_buckets"] = list(num_items_buckets)
        w["pad_token_id"] = int(pad_token_id)
        w["query_fill_token_id"] = int(query_fill_token_id)
        w["item_fill_token_id"] = int(item_fill_token_id)
        out.append(w)
    return out


def extract_shape_contract_signals(
    run_dir: Path,
    workloads: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "shape_contract_violation_count": 0,
        "shape_contract_violations": [],
        "shape_contract_checked_workloads": [str(w["name"]) for w in workloads],
        "shape_contract_passed_workloads": [],
    }
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        out["shape_contract_violations"] = ["missing run_metadata.json"]
        out["shape_contract_violation_count"] = 1
        return out
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as e:  # noqa: BLE001
        out["shape_contract_violations"] = [f"failed to parse run_metadata.json: {e}"]
        out["shape_contract_violation_count"] = 1
        return out

    run_diags = metadata.get("run_diagnostics", {})
    tpu_diags: dict[str, Any] | None = None
    if isinstance(run_diags, dict):
        if isinstance(run_diags.get("tpu"), dict):
            tpu_diags = run_diags.get("tpu")
        else:
            for value in run_diags.values():
                if isinstance(value, dict):
                    tpu_diags = value
                    break
    if not isinstance(tpu_diags, dict):
        out["shape_contract_violations"] = ["missing tpu run diagnostics"]
        out["shape_contract_violation_count"] = 1
        return out

    violations: list[str] = []
    passed: list[str] = []
    for wl in workloads:
        wl_name = str(wl["name"])
        wl_diag = tpu_diags.get(wl_name)
        if not isinstance(wl_diag, dict):
            violations.append(f"{wl_name}: missing workload diagnostics")
            continue
        shape = wl_diag.get("shape_contract")
        if not isinstance(shape, dict):
            violations.append(f"{wl_name}: missing shape_contract block")
            continue
        enabled = bool(shape.get("enabled", False))
        matches = bool(shape.get("request_matches_bucket", False))
        use_token_ids = bool(shape.get("use_token_ids", False))
        shape_violations = [str(v) for v in shape.get("violations", []) if str(v)]

        if not enabled:
            violations.append(f"{wl_name}: shape_contract disabled")
            continue
        if not use_token_ids:
            violations.append(f"{wl_name}: use_token_ids=false")
            continue
        if not matches:
            violations.append(f"{wl_name}: request shape != selected bucket")
            continue
        if shape_violations:
            violations.append(f"{wl_name}: {', '.join(shape_violations)}")
            continue
        passed.append(wl_name)

    out["shape_contract_passed_workloads"] = sorted(passed)
    out["shape_contract_violations"] = sorted(violations)
    out["shape_contract_violation_count"] = len(violations)
    return out


def parse_score_rows(log_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        m = SCORE_METRICS_RE.search(line)
        if m is None:
            continue
        fastpath_match = SCORE_FASTPATH_RE.search(line)
        if fastpath_match is not None:
            fastpath_attempted = fastpath_match.group(1) == "True"
            fastpath_succeeded = fastpath_match.group(2) == "True"
            fastpath_fallback_reason = fastpath_match.group(3)
        else:
            fastpath_attempted = None
            fastpath_succeeded = None
            fastpath_fallback_reason = None
        rows.append(
            {
                "path": m.group(1),
                "items": int(m.group(2)),
                "dispatches": int(m.group(3)),
                "queue_wait_s": float(m.group(4)),
                "device_compute_s": float(m.group(5)),
                "host_orchestration_s": float(m.group(6)),
                "fastpath_attempted": fastpath_attempted,
                "fastpath_succeeded": fastpath_succeeded,
                "fastpath_fallback_reason": fastpath_fallback_reason,
            }
        )
    return rows


def parse_log_signals(log_text: str) -> dict[str, Any]:
    rows = parse_score_rows(log_text)
    missing_cache_handle = len(re.findall(r"Missing scoring cache handle", log_text))
    prefill_rows = PREFILL_RE.findall(log_text)
    dominant_new_token = None
    if prefill_rows:
        dominant_new_token = _mode_int([int(r[1]) for r in prefill_rows])
    outliers = len([r for r in rows if float(r["device_compute_s"]) >= 1.0])
    fastpath_rows = [
        r
        for r in rows
        if r.get("fastpath_attempted") is not None and r.get("fastpath_succeeded") is not None
    ]
    fastpath_attempted_total = sum(1 for r in fastpath_rows if bool(r["fastpath_attempted"]))
    fastpath_succeeded_total = sum(1 for r in fastpath_rows if bool(r["fastpath_succeeded"]))
    fastpath_non_success_total = sum(
        1
        for r in fastpath_rows
        if bool(r["fastpath_attempted"]) and (not bool(r["fastpath_succeeded"]))
    )
    fallback_reason_events = [
        str(r["fastpath_fallback_reason"])
        for r in fastpath_rows
        if str(r.get("fastpath_fallback_reason"))
        and str(r.get("fastpath_fallback_reason")) not in {"None", "none", "null"}
    ]
    fallback_reasons = sorted(
        {
            reason for reason in fallback_reason_events
        }
    )
    server_label_only = _parse_optional_server_bool(log_text, SERVER_LABEL_ONLY_RE)
    server_fastpath_metrics = _parse_optional_server_bool(log_text, SERVER_FASTPATH_METRICS_RE)
    return {
        "score_samples": len(rows),
        "device_outliers_ge_1s": outliers,
        "missing_cache_handle": missing_cache_handle,
        "prefill_samples": len(prefill_rows),
        "dominant_new_token": dominant_new_token,
        "fastpath_metrics_samples": len(fastpath_rows),
        "fastpath_attempted_total": fastpath_attempted_total,
        "fastpath_succeeded_total": fastpath_succeeded_total,
        "fastpath_non_success_total": fastpath_non_success_total,
        "fastpath_fallback_total": len(fallback_reason_events),
        "fastpath_fallback_reasons": fallback_reasons,
        "server_label_only_enabled": server_label_only,
        "server_fastpath_metrics_enabled": server_fastpath_metrics,
        "timed_xla_compile_count": 0,
        "timed_xla_compile_workloads": [],
        "timed_xla_compile_requests": [],
        "warmup_xla_detect_count": 0,
        "warmup_xla_detect_workloads": [],
        "profiler_stdout_xla_marker_count": 0,
    }


def summarize_score_phase(
    score_rows: list[dict[str, Any]],
    *,
    items: int,
    measured_tput: float,
) -> dict[str, float | int | None]:
    subset = [r for r in score_rows if int(r["items"]) == items]
    steady = [r for r in subset if float(r["device_compute_s"]) < 1.0]

    queue_med = _median([float(r["queue_wait_s"]) for r in steady])
    device_med = _median([float(r["device_compute_s"]) for r in steady])
    host_med = _median([float(r["host_orchestration_s"]) for r in steady])
    dispatches_med = _median([float(r["dispatches"]) for r in subset])

    score_phase_s = None
    if queue_med is not None and device_med is not None and host_med is not None:
        score_phase_s = queue_med + device_med + host_med

    theoretical = None
    if score_phase_s and score_phase_s > 0:
        theoretical = items / score_phase_s
    utilization = None
    if theoretical and theoretical > 0:
        utilization = (measured_tput / theoretical) * 100.0

    return {
        "samples": len(subset),
        "steady_samples": len(steady),
        "dispatches_median": dispatches_med,
        "queue_wait_median_ms": queue_med * 1000.0 if queue_med is not None else None,
        "device_compute_median_ms": device_med * 1000.0 if device_med is not None else None,
        "host_orchestration_median_ms": host_med * 1000.0 if host_med is not None else None,
        "score_phase_median_s": score_phase_s,
        "theoretical_items_per_sec": theoretical,
        "utilization_pct": utilization,
    }


def build_failed_raw_results(
    timed_requests: int,
    workloads: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tpu": {
            wl["name"]: {
                "throughput_items_per_sec": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p99_ms": 0.0,
                "num_failures": timed_requests,
                "raw_latencies_ms": [],
                "failed_requests": [],
                "error_counts": {},
            }
            for wl in workloads
        }
    }


def build_ranked_report(
    group_dir: Path,
    runs: list[MatrixRun],
    *,
    repeats: int,
    timed_requests: int,
    gate_config: GateConfig,
    workloads: list[dict[str, Any]],
    workload_items: dict[str, int],
) -> Path:
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repeats_requested": repeats,
        "timed_requests_per_repeat": timed_requests,
        "runs": [],
    }

    by_workload: dict[str, dict[int, list[dict[str, Any]]]] = {wl["name"]: {} for wl in workloads}
    by_ips_log_signals: dict[int, list[dict[str, Any]]] = {}

    for run in runs:
        raw_tpu = run.raw_results.get("tpu", {})
        log_text = run.tpu_log_path.read_text(errors="ignore") if run.tpu_log_path.exists() else ""
        log_signals = parse_log_signals(log_text)
        compile_signals = dict(run.compile_signals or {})
        shape_signals = dict(run.shape_signals or {})
        timed_xla_compile_count = int(compile_signals.get("timed_xla_compile_count", 0))
        warmup_xla_detect_count = int(compile_signals.get("warmup_xla_detect_count", 0))
        profiler_stdout_xla_marker_count = int(
            compile_signals.get("profiler_stdout_xla_marker_count", 0)
        )
        timed_xla_workloads = [
            str(x) for x in compile_signals.get("timed_xla_compile_workloads", []) if str(x)
        ]
        timed_xla_requests = [
            str(x) for x in compile_signals.get("timed_xla_compile_requests", []) if str(x)
        ]
        warmup_xla_workloads = [
            str(x) for x in compile_signals.get("warmup_xla_detect_workloads", []) if str(x)
        ]
        shape_violation_count = int(shape_signals.get("shape_contract_violation_count", 0))
        shape_violations = [
            str(x) for x in shape_signals.get("shape_contract_violations", []) if str(x)
        ]
        shape_passed_workloads = [
            str(x) for x in shape_signals.get("shape_contract_passed_workloads", []) if str(x)
        ]
        log_signals["timed_xla_compile_count"] = timed_xla_compile_count
        log_signals["timed_xla_compile_workloads"] = timed_xla_workloads
        log_signals["timed_xla_compile_requests"] = timed_xla_requests
        log_signals["warmup_xla_detect_count"] = warmup_xla_detect_count
        log_signals["warmup_xla_detect_workloads"] = warmup_xla_workloads
        log_signals["profiler_stdout_xla_marker_count"] = profiler_stdout_xla_marker_count
        log_signals["shape_contract_violation_count"] = shape_violation_count
        log_signals["shape_contract_violations"] = shape_violations
        log_signals["shape_contract_passed_workloads"] = shape_passed_workloads
        score_rows = parse_score_rows(log_text)
        by_ips_log_signals.setdefault(run.items_per_step, []).append(log_signals)

        run_entry = {
            "items_per_step": run.items_per_step,
            "repeat_idx": run.repeat_idx,
            "run_dir": str(run.run_dir),
            "run_error": run.run_error,
            "compile_signals": {
                "timed_xla_compile_count": timed_xla_compile_count,
                "timed_xla_compile_workloads": timed_xla_workloads,
                "timed_xla_compile_requests": timed_xla_requests,
                "warmup_xla_detect_count": warmup_xla_detect_count,
                "warmup_xla_detect_workloads": warmup_xla_workloads,
                "profiler_stdout_xla_marker_count": profiler_stdout_xla_marker_count,
            },
            "shape_signals": {
                "shape_contract_violation_count": shape_violation_count,
                "shape_contract_violations": shape_violations,
                "shape_contract_passed_workloads": shape_passed_workloads,
            },
            "log_signals": log_signals,
            "workloads": {},
        }
        for wl in workloads:
            wl_name = wl["name"]
            items = workload_items[wl_name]
            m = raw_tpu.get(wl_name, {})
            throughput = float(m.get("throughput_items_per_sec", 0.0))
            p50 = float(m.get("latency_p50_ms", 0.0))
            p99 = float(m.get("latency_p99_ms", 0.0))
            failures = int(m.get("num_failures", timed_requests))
            error_counts_raw = m.get("error_counts", {}) or {}
            error_counts = {
                str(k): int(v)
                for k, v in error_counts_raw.items()
                if isinstance(v, (int, float))
            }
            failed_requests = m.get("failed_requests", []) or []
            p99_p50 = (p99 / p50) if p50 > 0 else None
            phase = summarize_score_phase(score_rows, items=items, measured_tput=throughput)

            wl_entry = {
                "throughput_items_per_sec": throughput,
                "latency_p50_ms": p50,
                "latency_p99_ms": p99,
                "p99_p50_ratio": p99_p50,
                "num_failures": failures,
                "error_counts": error_counts,
                "failed_requests_sample": failed_requests[:3],
                "score_phase": phase,
            }
            run_entry["workloads"][wl_name] = wl_entry
            by_workload[wl_name].setdefault(run.items_per_step, []).append(wl_entry)
        summary["runs"].append(run_entry)

    rankings: dict[str, list[dict[str, Any]]] = {}
    aggregate_by_workload: dict[str, dict[int, dict[str, Any]]] = {}
    overall_scores: dict[int, float] = {}

    for wl_name, by_ips in by_workload.items():
        aggregate_by_workload[wl_name] = {}
        rows: list[dict[str, Any]] = []
        for ips, entries in by_ips.items():
            tputs = [float(e["throughput_items_per_sec"]) for e in entries]
            p50s = [float(e["latency_p50_ms"]) for e in entries]
            p99s = [float(e["latency_p99_ms"]) for e in entries]
            fails = [int(e["num_failures"]) for e in entries]
            ratios = [
                float(e["p99_p50_ratio"])
                for e in entries
                if e["p99_p50_ratio"] is not None
            ]

            util_vals = [
                float(e["score_phase"]["utilization_pct"])
                for e in entries
                if e["score_phase"]["utilization_pct"] is not None
            ]
            dispatch_vals = [
                float(e["score_phase"]["dispatches_median"])
                for e in entries
                if e["score_phase"]["dispatches_median"] is not None
            ]
            queue_wait_vals = [
                float(e["score_phase"]["queue_wait_median_ms"])
                for e in entries
                if e["score_phase"]["queue_wait_median_ms"] is not None
            ]
            host_orch_vals = [
                float(e["score_phase"]["host_orchestration_median_ms"])
                for e in entries
                if e["score_phase"]["host_orchestration_median_ms"] is not None
            ]
            error_counts_agg: dict[str, int] = {}
            for e in entries:
                for err, cnt in e.get("error_counts", {}).items():
                    error_counts_agg[str(err)] = error_counts_agg.get(str(err), 0) + int(cnt)
            top_error_counts = sorted(
                error_counts_agg.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5]
            score_samples = [int(e["score_phase"]["samples"]) for e in entries]
            steady_samples = [int(e["score_phase"]["steady_samples"]) for e in entries]

            denom = max(1, len(entries) * timed_requests)
            row = {
                "items_per_step": ips,
                "repeats_collected": len(entries),
                "throughput_median_items_per_sec": _median(tputs),
                "throughput_mean_items_per_sec": _mean(tputs),
                "throughput_cv": _cv(tputs),
                "latency_p50_median_ms": _median(p50s),
                "latency_p99_median_ms": _median(p99s),
                "latency_p99_cv": _cv(p99s),
                "p99_p50_ratio_median": _median(ratios),
                "num_failures_total": sum(fails),
                "failure_rate": sum(fails) / denom,
                "score_utilization_pct_median": _median(util_vals),
                "dispatches_median": _median(dispatch_vals),
                "queue_wait_median_ms_median": _median(queue_wait_vals),
                "host_orchestration_median_ms_median": _median(host_orch_vals),
                "score_samples_total": sum(score_samples),
                "steady_score_samples_total": sum(steady_samples),
                "error_counts_total": error_counts_agg,
                "top_error_counts": top_error_counts,
            }
            rows.append(row)
            aggregate_by_workload[wl_name][ips] = row

        max_tput = max(
            [float(r["throughput_median_items_per_sec"] or 0.0) for r in rows], default=0.0
        )
        p99_candidates = [float(r["latency_p99_median_ms"] or 0.0) for r in rows if float(r["latency_p99_median_ms"] or 0.0) > 0]
        min_p99 = min(p99_candidates) if p99_candidates else 0.0
        util_candidates = [float(r["score_utilization_pct_median"] or 0.0) for r in rows]
        max_util = max(util_candidates) if util_candidates else 0.0
        host_overhead_candidates = []
        for r in rows:
            queue_ms = r.get("queue_wait_median_ms_median")
            host_ms = r.get("host_orchestration_median_ms_median")
            if queue_ms is None or host_ms is None:
                continue
            host_overhead_candidates.append(float(queue_ms) + float(host_ms))
        min_host_overhead_ms = (
            min(host_overhead_candidates) if host_overhead_candidates else None
        )

        for row in rows:
            tput = float(row["throughput_median_items_per_sec"] or 0.0)
            p99 = float(row["latency_p99_median_ms"] or 0.0)
            util = float(row["score_utilization_pct_median"] or 0.0)
            tput_term = (tput / max_tput) if max_tput > 0 else 0.0
            p99_term = (min_p99 / p99) if min_p99 > 0 and p99 > 0 else 0.0
            util_term = (util / max_util) if max_util > 0 else 0.0
            host_overhead_term = 0.0
            queue_ms = row.get("queue_wait_median_ms_median")
            host_ms = row.get("host_orchestration_median_ms_median")
            if (
                min_host_overhead_ms is not None
                and queue_ms is not None
                and host_ms is not None
            ):
                denom = float(queue_ms) + float(host_ms)
                if denom > 0:
                    host_overhead_term = min_host_overhead_ms / denom

            failure_penalty = min(0.8, float(row["failure_rate"]) * 2.0)

            stability_penalty = 0.0
            t_cv = row["throughput_cv"]
            if t_cv is not None and float(t_cv) > gate_config.max_throughput_cv:
                stability_penalty += min(0.2, (float(t_cv) - gate_config.max_throughput_cv) * 0.5)
            p99_cv = row["latency_p99_cv"]
            if p99_cv is not None and float(p99_cv) > gate_config.max_p99_cv:
                stability_penalty += min(0.2, (float(p99_cv) - gate_config.max_p99_cv) * 0.5)

            dispatch_penalty = 0.0
            dispatches_median = row.get("dispatches_median")
            if dispatches_median is not None:
                ideal_dispatches = max(
                    1.0,
                    math.ceil(float(workload_items[wl_name]) / float(row["items_per_step"])),
                )
                dispatch_over_ideal = max(0.0, float(dispatches_median) - ideal_dispatches)
                dispatch_penalty = min(0.20, dispatch_over_ideal * 0.05)

            row["dispatch_fragmentation_penalty"] = dispatch_penalty
            row["host_overhead_term"] = host_overhead_term
            row["score"] = (
                (0.50 * tput_term)
                + (0.25 * p99_term)
                + (0.15 * util_term)
                + (0.10 * host_overhead_term)
                - failure_penalty
                - stability_penalty
                - dispatch_penalty
            )
            overall_scores[row["items_per_step"]] = overall_scores.get(row["items_per_step"], 0.0) + float(row["score"])

        rows.sort(key=lambda x: float(x["score"]), reverse=True)
        rankings[wl_name] = rows

    overall_sorted = [
        {"items_per_step": k, "score_sum": v}
        for k, v in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    log_signal_summary: dict[int, dict[str, Any]] = {}
    for ips, logs in by_ips_log_signals.items():
        dominant_vals = [int(l["dominant_new_token"]) for l in logs if l["dominant_new_token"] is not None]
        fallback_reasons: set[str] = set()
        for l in logs:
            for reason in l.get("fastpath_fallback_reasons", []):
                if str(reason):
                    fallback_reasons.add(str(reason))
        log_signal_summary[ips] = {
            "repeats_collected": len(logs),
            "score_samples_total": sum(int(l["score_samples"]) for l in logs),
            "score_samples_min": min((int(l["score_samples"]) for l in logs), default=0),
            "device_outliers_ge_1s_total": sum(int(l["device_outliers_ge_1s"]) for l in logs),
            "missing_cache_handle_total": sum(int(l["missing_cache_handle"]) for l in logs),
            "cache_transition_repeats": sum(
                1 for l in logs if int(l.get("missing_cache_handle", 0)) > 0
            ),
            "prefill_samples_total": sum(int(l["prefill_samples"]) for l in logs),
            "dominant_new_token": _mode_int(dominant_vals),
            "fastpath_metrics_samples_total": sum(
                int(l.get("fastpath_metrics_samples", 0)) for l in logs
            ),
            "fastpath_attempted_total": sum(
                int(l.get("fastpath_attempted_total", 0)) for l in logs
            ),
            "fastpath_succeeded_total": sum(
                int(l.get("fastpath_succeeded_total", 0)) for l in logs
            ),
            "fastpath_non_success_total": sum(
                int(l.get("fastpath_non_success_total", 0)) for l in logs
            ),
            "fastpath_fallback_total": sum(
                int(l.get("fastpath_fallback_total", 0)) for l in logs
            ),
            "fastpath_fallback_reasons": sorted(fallback_reasons),
            "server_label_only_enabled_repeats": sum(
                1 for l in logs if l.get("server_label_only_enabled") is True
            ),
            "server_label_only_disabled_repeats": sum(
                1 for l in logs if l.get("server_label_only_enabled") is False
            ),
            "server_fastpath_metrics_enabled_repeats": sum(
                1 for l in logs if l.get("server_fastpath_metrics_enabled") is True
            ),
            "server_fastpath_metrics_disabled_repeats": sum(
                1 for l in logs if l.get("server_fastpath_metrics_enabled") is False
            ),
            "timed_xla_compile_total": sum(int(l.get("timed_xla_compile_count", 0)) for l in logs),
            "timed_xla_compile_repeats": sum(
                1 for l in logs if int(l.get("timed_xla_compile_count", 0)) > 0
            ),
            "warmup_xla_detect_total": sum(int(l.get("warmup_xla_detect_count", 0)) for l in logs),
            "warmup_xla_detect_repeats": sum(
                1 for l in logs if int(l.get("warmup_xla_detect_count", 0)) > 0
            ),
            "profiler_stdout_xla_marker_total": sum(
                int(l.get("profiler_stdout_xla_marker_count", 0)) for l in logs
            ),
            "shape_contract_violation_total": sum(
                int(l.get("shape_contract_violation_count", 0)) for l in logs
            ),
            "shape_contract_violation_repeats": sum(
                1 for l in logs if int(l.get("shape_contract_violation_count", 0)) > 0
            ),
        }

    gate_by_ips: dict[int, dict[str, Any]] = {}
    for row in overall_sorted:
        ips = int(row["items_per_step"])
        failures: list[str] = []
        checks: dict[str, Any] = {}

        log_summary = log_signal_summary.get(ips, {})
        score_samples_min = int(log_summary.get("score_samples_min", 0))
        score_samples_total = int(log_summary.get("score_samples_total", 0))
        missing_cache_total = int(log_summary.get("missing_cache_handle_total", 0))
        timed_xla_compile_total = int(log_summary.get("timed_xla_compile_total", 0))
        shape_contract_violation_total = int(log_summary.get("shape_contract_violation_total", 0))
        fastpath_metrics_samples_total = int(
            log_summary.get("fastpath_metrics_samples_total", 0)
        )
        fastpath_attempted_total = int(log_summary.get("fastpath_attempted_total", 0))
        fastpath_succeeded_total = int(log_summary.get("fastpath_succeeded_total", 0))
        fastpath_non_success_total = int(log_summary.get("fastpath_non_success_total", 0))
        fastpath_fallback_total = int(log_summary.get("fastpath_fallback_total", 0))
        fastpath_fallback_reasons = [
            str(x) for x in log_summary.get("fastpath_fallback_reasons", []) if str(x)
        ]
        server_label_only_disabled_repeats = int(
            log_summary.get("server_label_only_disabled_repeats", 0)
        )
        server_fastpath_metrics_disabled_repeats = int(
            log_summary.get("server_fastpath_metrics_disabled_repeats", 0)
        )

        checks["min_score_samples"] = {
            "value": score_samples_min,
            "threshold": gate_config.min_score_samples,
            "pass": score_samples_min >= gate_config.min_score_samples,
        }
        if not checks["min_score_samples"]["pass"]:
            failures.append(
                f"score_samples_min={score_samples_min} < {gate_config.min_score_samples}"
            )

        checks["missing_cache_handle"] = {
            "value": missing_cache_total,
            "threshold": 0,
            "pass": gate_config.allow_missing_cache_handle or missing_cache_total == 0,
        }
        if not checks["missing_cache_handle"]["pass"]:
            failures.append(f"missing_cache_handle_total={missing_cache_total} > 0")

        checks["timed_xla_compilation"] = {
            "value": timed_xla_compile_total,
            "threshold": 0,
            "pass": gate_config.allow_timed_xla_compilation or timed_xla_compile_total == 0,
        }
        if not checks["timed_xla_compilation"]["pass"]:
            failures.append(f"timed_xla_compile_total={timed_xla_compile_total} > 0")

        checks["shape_contract"] = {
            "value": shape_contract_violation_total,
            "threshold": 0,
            "pass": gate_config.allow_shape_contract_violations
            or shape_contract_violation_total == 0,
        }
        if not checks["shape_contract"]["pass"]:
            failures.append(f"shape_contract_violation_total={shape_contract_violation_total} > 0")

        cache_transition_observed = missing_cache_total > 0
        checks["cache_path_compile_stability"] = {
            "value": {
                "cache_transition_observed": cache_transition_observed,
                "missing_cache_handle_total": missing_cache_total,
                "timed_xla_compile_total": timed_xla_compile_total,
            },
            "threshold": {
                "timed_xla_compile_total_when_transition_observed": 0,
            },
            "pass": (not cache_transition_observed) or timed_xla_compile_total == 0,
        }
        if not checks["cache_path_compile_stability"]["pass"]:
            failures.append(
                "cache transition triggered timed compile "
                f"(missing_cache_handle_total={missing_cache_total}, "
                f"timed_xla_compile_total={timed_xla_compile_total})"
            )

        checks["cache_transition_exercised"] = {
            "value": cache_transition_observed,
            "threshold": True,
            "pass": (not gate_config.require_cache_transition_exercise)
            or cache_transition_observed,
        }
        if not checks["cache_transition_exercised"]["pass"]:
            failures.append(
                "cache transition not exercised (missing_cache_handle_total=0)"
            )

        checks["score_fastpath_metrics_coverage"] = {
            "value": fastpath_metrics_samples_total,
            "threshold": score_samples_total,
            "pass": gate_config.allow_score_full_vocab_fallback
            or fastpath_metrics_samples_total >= score_samples_total,
        }
        if not checks["score_fastpath_metrics_coverage"]["pass"]:
            failures.append(
                "fastpath_metrics_samples_total="
                f"{fastpath_metrics_samples_total} < score_samples_total={score_samples_total}"
            )

        checks["score_fastpath_success"] = {
            "value": {
                "attempted_total": fastpath_attempted_total,
                "succeeded_total": fastpath_succeeded_total,
                "non_success_total": fastpath_non_success_total,
                "fallback_total": fastpath_fallback_total,
                "fallback_reasons": fastpath_fallback_reasons,
            },
            "threshold": {
                "non_success_total": 0,
                "fallback_total": 0,
            },
            "pass": gate_config.allow_score_full_vocab_fallback
            or (
                fastpath_attempted_total > 0
                and fastpath_succeeded_total >= fastpath_attempted_total
                and fastpath_non_success_total == 0
                and fastpath_fallback_total == 0
            ),
        }
        if not checks["score_fastpath_success"]["pass"]:
            failures.append(
                "fastpath fallback/non-success detected "
                f"(attempted={fastpath_attempted_total}, succeeded={fastpath_succeeded_total}, "
                f"non_success={fastpath_non_success_total}, fallback={fastpath_fallback_total}, "
                f"reasons={fastpath_fallback_reasons or ['none']})"
            )

        checks["score_mode_label_only"] = {
            "value": {
                "server_label_only_disabled_repeats": server_label_only_disabled_repeats,
                "server_fastpath_metrics_disabled_repeats": (
                    server_fastpath_metrics_disabled_repeats
                ),
            },
            "threshold": {
                "server_label_only_disabled_repeats": 0,
                "server_fastpath_metrics_disabled_repeats": 0,
            },
            "pass": gate_config.allow_score_full_vocab_fallback
            or (
                server_label_only_disabled_repeats == 0
                and server_fastpath_metrics_disabled_repeats == 0
            ),
        }
        if not checks["score_mode_label_only"]["pass"]:
            failures.append(
                "server score mode flags indicate non-label-only/fallback-prone mode "
                f"(label_only_disabled_repeats={server_label_only_disabled_repeats}, "
                f"fastpath_metrics_disabled_repeats={server_fastpath_metrics_disabled_repeats})"
            )

        for wl in workloads:
            wl_name = wl["name"]
            agg = aggregate_by_workload.get(wl_name, {}).get(ips)
            if agg is None:
                failures.append(f"{wl_name}: missing aggregate")
                continue

            wl_checks: dict[str, Any] = {}
            wl_failure_rate = float(agg["failure_rate"])
            wl_ratio = agg["p99_p50_ratio_median"]
            wl_tput_cv = agg["throughput_cv"]
            wl_p99_cv = agg["latency_p99_cv"]
            wl_tput_med = float(agg["throughput_median_items_per_sec"] or 0.0)
            wl_tail_gate = gate_config.max_p99_p50.get(wl_name, 5.0)

            wl_checks["failure_rate"] = {
                "value": wl_failure_rate,
                "threshold": gate_config.max_failure_rate,
                "pass": wl_failure_rate <= gate_config.max_failure_rate,
            }
            wl_checks["tail_ratio_p99_p50"] = {
                "value": wl_ratio,
                "threshold": wl_tail_gate,
                "pass": wl_ratio is not None and float(wl_ratio) <= wl_tail_gate,
            }
            wl_checks["throughput_cv"] = {
                "value": wl_tput_cv,
                "threshold": gate_config.max_throughput_cv,
                "pass": wl_tput_cv is None or float(wl_tput_cv) <= gate_config.max_throughput_cv,
            }
            wl_checks["p99_cv"] = {
                "value": wl_p99_cv,
                "threshold": gate_config.max_p99_cv,
                "pass": wl_p99_cv is None or float(wl_p99_cv) <= gate_config.max_p99_cv,
            }
            wl_checks["throughput_nonzero"] = {
                "value": wl_tput_med,
                "threshold": 0.0,
                "pass": wl_tput_med > 0.0,
            }

            if not wl_checks["failure_rate"]["pass"]:
                failures.append(
                    f"{wl_name}: failure_rate={wl_failure_rate:.3f} > {gate_config.max_failure_rate:.3f}"
                )
            if not wl_checks["tail_ratio_p99_p50"]["pass"]:
                failures.append(
                    f"{wl_name}: p99/p50={_fmt(wl_ratio, 2)} > {wl_tail_gate:.2f}"
                )
            if not wl_checks["throughput_cv"]["pass"]:
                failures.append(
                    f"{wl_name}: throughput_cv={_fmt(wl_tput_cv, 3)} > {gate_config.max_throughput_cv:.3f}"
                )
            if not wl_checks["p99_cv"]["pass"]:
                failures.append(
                    f"{wl_name}: p99_cv={_fmt(wl_p99_cv, 3)} > {gate_config.max_p99_cv:.3f}"
                )
            if not wl_checks["throughput_nonzero"]["pass"]:
                failures.append(f"{wl_name}: throughput_median=0")

            checks[wl_name] = wl_checks

        gate_by_ips[ips] = {
            "pass": len(failures) == 0,
            "failed_checks": failures,
            "checks": checks,
        }

    passing = [r for r in overall_sorted if gate_by_ips.get(int(r["items_per_step"]), {}).get("pass")]
    recommended = passing[0] if passing else (overall_sorted[0] if overall_sorted else None)
    recommended_reason = (
        "highest overall score among gate-passing candidates"
        if passing
        else "no candidate passed all gates; selected highest overall score"
    )

    summary["aggregates"] = {
        "by_workload": rankings,
        "log_signals_by_items_per_step": log_signal_summary,
    }
    summary["overall"] = overall_sorted
    summary["gates"] = {
        "config": {
            "max_failure_rate": gate_config.max_failure_rate,
            "max_p99_p50": gate_config.max_p99_p50,
            "max_throughput_cv": gate_config.max_throughput_cv,
            "max_p99_cv": gate_config.max_p99_cv,
            "min_score_samples": gate_config.min_score_samples,
            "allow_missing_cache_handle": gate_config.allow_missing_cache_handle,
            "allow_timed_xla_compilation": gate_config.allow_timed_xla_compilation,
            "allow_shape_contract_violations": gate_config.allow_shape_contract_violations,
            "allow_score_full_vocab_fallback": gate_config.allow_score_full_vocab_fallback,
            "require_cache_transition_exercise": gate_config.require_cache_transition_exercise,
        },
        "by_items_per_step": gate_by_ips,
    }
    summary["recommended"] = {
        "items_per_step": recommended["items_per_step"] if recommended else None,
        "reason": recommended_reason if recommended else "no runs collected",
    }

    summary_path = group_dir / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    gates_path = group_dir / "matrix_gates.json"
    gates_path.write_text(json.dumps(summary["gates"], indent=2))

    md_lines: list[str] = []
    md_lines.append(f"# TPU Scoring Matrix: {group_dir.name}")
    md_lines.append("")
    md_lines.append("## Metadata")
    md_lines.append("")
    md_lines.append(f"- repeats_requested: `{repeats}`")
    md_lines.append(f"- timed_requests_per_repeat: `{timed_requests}`")
    md_lines.append(f"- recommended_items_per_step: `{summary['recommended']['items_per_step']}` ({recommended_reason})")
    md_lines.append("")
    md_lines.append("## Overall Ranking")
    md_lines.append("")
    md_lines.append("| Rank | items_per_step | score_sum | gate_pass |")
    md_lines.append("|---|---:|---:|---:|")
    for i, row in enumerate(overall_sorted, start=1):
        ips = int(row["items_per_step"])
        gate_pass = gate_by_ips.get(ips, {}).get("pass", False)
        md_lines.append(f"| {i} | {ips} | {row['score_sum']:.4f} | {str(gate_pass).lower()} |")
    md_lines.append("")

    for wl_name in sorted(rankings.keys()):
        md_lines.append(f"## Workload: {wl_name}")
        md_lines.append("")
        md_lines.append(
            "| Rank | items_per_step | repeats | tput_med (items/s) | tput_cv | "
            "p50_med (ms) | p99_med (ms) | p99/p50 | queue_wait_med (ms) | host_orch_med (ms) | "
            "host_overhead_term | dispatches_med | dispatch_frag_penalty | "
            "failure_rate | util_med (%) | score | top_failure_reason |"
        )
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for i, row in enumerate(rankings[wl_name], start=1):
            top_failure_reason = "none"
            if row["top_error_counts"]:
                err, cnt = row["top_error_counts"][0]
                top_failure_reason = f"{cnt}x {str(err)[:120]}"
            md_lines.append(
                f"| {i} | {row['items_per_step']} | {row['repeats_collected']} | "
                f"{_fmt(row['throughput_median_items_per_sec'], 1)} | {_fmt(row['throughput_cv'], 3)} | "
                f"{_fmt(row['latency_p50_median_ms'], 1)} | {_fmt(row['latency_p99_median_ms'], 1)} | "
                f"{_fmt(row['p99_p50_ratio_median'], 2)} | "
                f"{_fmt(row.get('queue_wait_median_ms_median'), 3)} | "
                f"{_fmt(row.get('host_orchestration_median_ms_median'), 3)} | "
                f"{_fmt(row.get('host_overhead_term'), 3)} | "
                f"{_fmt(row['dispatches_median'], 2)} | "
                f"{_fmt(row.get('dispatch_fragmentation_penalty'), 3)} | "
                f"{_fmt(row['failure_rate'] * 100.0, 1)}% | "
                f"{_fmt(row['score_utilization_pct_median'], 1)} | {row['score']:.4f} | {top_failure_reason} |"
            )
        md_lines.append("")

    md_lines.append("## Gate Evaluation")
    md_lines.append("")
    md_lines.append("| items_per_step | gate_pass | failed_checks |")
    md_lines.append("|---:|---:|---|")
    for row in overall_sorted:
        ips = int(row["items_per_step"])
        g = gate_by_ips.get(ips, {"pass": False, "failed_checks": ["missing gate data"]})
        failed = "; ".join(g["failed_checks"]) if g["failed_checks"] else "none"
        md_lines.append(f"| {ips} | {str(g['pass']).lower()} | {failed} |")
    md_lines.append("")

    md_lines.append("## Log Signals by items_per_step")
    md_lines.append("")
    md_lines.append(
        "| items_per_step | repeats | score_samples_total | score_samples_min | "
        "fastpath_metrics_samples_total | fastpath_attempted_total | fastpath_succeeded_total | "
        "fastpath_non_success_total | fastpath_fallback_total | fastpath_fallback_reasons | "
        "server_label_only_disabled_repeats | server_fastpath_metrics_disabled_repeats | "
        "device_outliers_ge_1s_total | missing_cache_handle_total | cache_transition_repeats | "
        "timed_xla_compile_total | timed_xla_compile_repeats | "
        "warmup_xla_detect_total | warmup_xla_detect_repeats | "
        "profiler_stdout_xla_marker_total | shape_contract_violation_total | "
        "shape_contract_violation_repeats | prefill_samples_total | dominant_new_token |"
    )
    md_lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for ips in sorted(log_signal_summary.keys()):
        s = log_signal_summary[ips]
        md_lines.append(
            f"| {ips} | {s['repeats_collected']} | {s['score_samples_total']} | {s['score_samples_min']} | "
            f"{s['fastpath_metrics_samples_total']} | {s['fastpath_attempted_total']} | "
            f"{s['fastpath_succeeded_total']} | {s['fastpath_non_success_total']} | "
            f"{s['fastpath_fallback_total']} | "
            f"{','.join(s['fastpath_fallback_reasons']) if s['fastpath_fallback_reasons'] else 'none'} | "
            f"{s['server_label_only_disabled_repeats']} | "
            f"{s['server_fastpath_metrics_disabled_repeats']} | "
            f"{s['device_outliers_ge_1s_total']} | {s['missing_cache_handle_total']} | "
            f"{s['cache_transition_repeats']} | "
            f"{s['timed_xla_compile_total']} | {s['timed_xla_compile_repeats']} | "
            f"{s['warmup_xla_detect_total']} | {s['warmup_xla_detect_repeats']} | "
            f"{s['profiler_stdout_xla_marker_total']} | {s['shape_contract_violation_total']} | "
            f"{s['shape_contract_violation_repeats']} | "
            f"{s['prefill_samples_total']} | {s['dominant_new_token']} |"
        )
    md_lines.append("")

    md_lines.append("## Compile Signals by Run")
    md_lines.append("")
    md_lines.append(
        "| items_per_step | repeat | timed_xla_compile_count | timed_workloads | "
        "timed_request_ids | warmup_xla_detect_count | warmup_workloads | profiler_stdout_xla_markers |"
    )
    md_lines.append("|---:|---:|---:|---|---|---:|---|---:|")
    for run in sorted(runs, key=lambda x: (x.items_per_step, x.repeat_idx)):
        c = run.compile_signals or {}
        timed_workloads = ",".join(str(x) for x in c.get("timed_xla_compile_workloads", []) if str(x))
        timed_requests = ",".join(str(x) for x in c.get("timed_xla_compile_requests", []) if str(x))
        warmup_workloads = ",".join(str(x) for x in c.get("warmup_xla_detect_workloads", []) if str(x))
        md_lines.append(
            f"| {run.items_per_step} | {run.repeat_idx} | "
            f"{int(c.get('timed_xla_compile_count', 0))} | {timed_workloads or 'none'} | "
            f"{timed_requests or 'none'} | "
            f"{int(c.get('warmup_xla_detect_count', 0))} | {warmup_workloads or 'none'} | "
            f"{int(c.get('profiler_stdout_xla_marker_count', 0))} |"
        )
    md_lines.append("")

    md_lines.append("## Shape Contract Signals by Run")
    md_lines.append("")
    md_lines.append("| items_per_step | repeat | violations | passed_workloads | details |")
    md_lines.append("|---:|---:|---:|---|---|")
    for run in sorted(runs, key=lambda x: (x.items_per_step, x.repeat_idx)):
        s = run.shape_signals or {}
        violation_count = int(s.get("shape_contract_violation_count", 0))
        passed = ",".join(str(x) for x in s.get("shape_contract_passed_workloads", []) if str(x))
        details = "; ".join(str(x) for x in s.get("shape_contract_violations", []) if str(x))
        md_lines.append(
            f"| {run.items_per_step} | {run.repeat_idx} | {violation_count} | "
            f"{passed or 'none'} | {details or 'none'} |"
        )
    md_lines.append("")

    md_lines.append("## Failure Diagnostics (Top Errors)")
    md_lines.append("")
    md_lines.append("| Workload | items_per_step | top_errors |")
    md_lines.append("|---|---:|---|")
    for wl_name in sorted(rankings.keys()):
        for row in rankings[wl_name]:
            errs = row.get("top_error_counts", [])
            if errs:
                err_txt = "; ".join(f"{cnt}x {str(err)[:80]}" for err, cnt in errs[:3])
            else:
                err_txt = "none"
            md_lines.append(f"| {wl_name} | {row['items_per_step']} | {err_txt} |")
    md_lines.append("")

    md_lines.append("## Run Directories")
    md_lines.append("")
    for run in sorted(runs, key=lambda x: (x.items_per_step, x.repeat_idx)):
        status = "ok" if run.run_error is None else f"error={run.run_error}"
        md_lines.append(
            f"- ips={run.items_per_step}, repeat={run.repeat_idx}: `{run.run_dir}` ({status})"
        )
    md_lines.append("")

    report_path = group_dir / "matrix_report.md"
    report_path.write_text("\n".join(md_lines))
    return report_path


def main() -> None:
    baseline_path = resolve_baseline_path(sys.argv[1:])
    baseline = load_baseline(baseline_path)
    baseline_benchmark = benchmark_defaults(baseline)
    baseline_matrix = matrix_defaults(baseline)
    baseline_tpu = tpu_defaults(baseline)
    baseline_gate_csv = gate_max_p99_p50_csv(baseline)
    baseline_workload_rows = baseline_workloads(baseline)
    baseline_shape = dict(baseline_matrix.get("shape_contract", {}))
    shape_query_default = ",".join(
        str(v) for v in baseline_shape.get("query_token_buckets", [120, 500, 2000])
    )
    shape_item_default = ",".join(
        str(v) for v in baseline_shape.get("item_token_buckets", [20])
    )
    shape_num_items_default = ",".join(
        str(v) for v in baseline_shape.get("num_items_buckets", [10, 100, 500])
    )

    global WORKLOADS, WORKLOAD_BY_NAME, WORKLOAD_ITEMS, DEFAULT_P99_P50_GATES
    WORKLOADS = [dict(w) for w in baseline_workload_rows]
    WORKLOAD_BY_NAME = {w["name"]: w for w in WORKLOADS}
    WORKLOAD_ITEMS = {w["name"]: int(w["num_items"]) for w in WORKLOADS}
    DEFAULT_P99_P50_GATES = {
        str(k): float(v) for k, v in baseline_matrix["gate_max_p99_p50"].items()
    }

    parser = argparse.ArgumentParser(description="Run TPU scoring items_per_step tuning sweep.")
    parser.add_argument(
        "--baseline-config",
        default=str(baseline_path),
        help="Path to canonical baseline YAML.",
    )
    parser.add_argument("--tpu-name", required=True)
    parser.add_argument("--tpu-zone", required=True)
    parser.add_argument("--tpu-project", default=None)
    parser.add_argument("--ssh-mode", choices=["direct", "gcloud"], default="direct")
    parser.add_argument("--tpu-host", default=None, help="TPU external IP (required for --ssh-mode=direct unless resolvable)")
    parser.add_argument("--ssh-user", default="kanna")
    parser.add_argument("--ssh-key", default=str(Path.home() / ".ssh/google_compute_engine"))
    parser.add_argument("--tpu-repo-path", default=baseline_tpu["repo_path"])
    parser.add_argument("--tpu-port", type=int, default=30001)
    parser.add_argument("--tpu-url", default=baseline_tpu["url"])
    parser.add_argument(
        "--tpu-client-location",
        choices=["local", "tpu_vm"],
        default="local",
        help=(
            "Where benchmark HTTP client runs. "
            "local=run profiler client on this machine, "
            "tpu_vm=run profiler client on TPU VM for client-path parity."
        ),
    )
    parser.add_argument(
        "--tpu-client-runtime-dir",
        default="/tmp/sglang-gpu-tpu-profiler-runtime",
        help="Remote TPU VM runtime directory used when --tpu-client-location=tpu_vm.",
    )
    parser.add_argument(
        "--tpu-client-python",
        default=None,
        help=(
            "Optional python binary path on TPU VM for co-located client mode. "
            "Default: <tpu-repo-path>/.venv/bin/python."
        ),
    )
    parser.add_argument(
        "--tpu-connection-mode",
        choices=["auto", "direct", "tunnel"],
        default="auto",
        help=(
            "Connection policy for TPU HTTP traffic. "
            "auto=infer from --tpu-url host (loopback=tunnel, non-loopback=direct)."
        ),
    )
    parser.add_argument(
        "--tpu-tunnel-autostart",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When connection mode is tunnel, start/restart a local SSH tunnel if missing. "
            "(default: true)"
        ),
    )
    parser.add_argument(
        "--tpu-tunnel-local-port",
        type=int,
        default=None,
        help="Local tunnel port. Defaults to port parsed from --tpu-url.",
    )
    parser.add_argument(
        "--tpu-tunnel-remote-port",
        type=int,
        default=None,
        help="Remote TPU port for tunnel forwarding. Defaults to --tpu-port.",
    )
    parser.add_argument("--model", default=str(baseline["experiment"]["model"]))
    parser.add_argument(
        "--shape-contract-enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(baseline_shape.get("enabled", True)),
        help="Enable strict request-shape bucketing/padding contract at API boundary.",
    )
    parser.add_argument(
        "--shape-contract-use-token-ids",
        action=argparse.BooleanOptionalAction,
        default=bool(baseline_shape.get("use_token_ids", True)),
        help="Send query/items as token IDs for exact shape control.",
    )
    parser.add_argument(
        "--shape-contract-strict",
        action=argparse.BooleanOptionalAction,
        default=bool(baseline_shape.get("strict", True)),
        help="Fail if workload dims do not fit any approved shape bucket.",
    )
    parser.add_argument(
        "--shape-query-token-buckets",
        default=shape_query_default,
        help="Approved query token buckets (comma-separated).",
    )
    parser.add_argument(
        "--shape-item-token-buckets",
        default=shape_item_default,
        help="Approved per-item token buckets (comma-separated).",
    )
    parser.add_argument(
        "--shape-num-items-buckets",
        default=shape_num_items_default,
        help="Approved item-count buckets per request (comma-separated).",
    )
    parser.add_argument(
        "--shape-pad-token-id",
        type=int,
        default=int(baseline_shape.get("pad_token_id", 0)),
        help="Token ID used for right-padding query/items in token-ID mode.",
    )
    parser.add_argument(
        "--shape-query-fill-token-id",
        type=int,
        default=int(baseline_shape.get("query_fill_token_id", 42)),
        help="Fill token ID used for logical query tokens in token-ID mode.",
    )
    parser.add_argument(
        "--shape-item-fill-token-id",
        type=int,
        default=int(baseline_shape.get("item_fill_token_id", 84)),
        help="Fill token ID used for logical item tokens in token-ID mode.",
    )
    parser.add_argument(
        "--workload-filter",
        default=None,
        help="Optional comma-separated workload names (e.g. pr28_hotshape,medium_batch).",
    )
    parser.add_argument(
        "--items-per-step",
        default=",".join(str(v) for v in baseline_matrix["items_per_step_candidates"]),
    )
    parser.add_argument(
        "--align-items-per-step-with-workloads",
        action=argparse.BooleanOptionalAction,
        default=bool(baseline_matrix.get("align_items_per_step_with_workloads", True)),
        help=(
            "For single-workload sweeps, inject workload num_items as an extra items_per_step "
            "candidate (bounded by max provided candidate) to reduce dispatch fragmentation."
        ),
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=baseline_matrix["max_running_requests"],
    )
    parser.add_argument(
        "--multi-item-extend-batch-size",
        type=int,
        default=baseline_matrix["multi_item_extend_batch_size"],
    )
    parser.add_argument(
        "--auto-bump-lane-capacity-with-workload",
        action=argparse.BooleanOptionalAction,
        default=bool(baseline_matrix.get("auto_bump_lane_capacity_with_workload", False)),
        help=(
            "For single-workload sweeps, bump max_running_requests and extend_batch_size up to "
            "workload num_items (capped) to reduce host orchestration/queue overhead."
        ),
    )
    parser.add_argument(
        "--lane-capacity-bump-cap",
        type=int,
        default=int(baseline_matrix.get("lane_capacity_bump_cap", 128)),
        help="Upper bound for auto-bumped lane capacity.",
    )
    parser.add_argument(
        "--multi-item-scoring-chunk-size",
        type=int,
        default=baseline_matrix["multi_item_scoring_chunk_size"],
    )
    parser.add_argument(
        "--precompile-token-paddings",
        default=",".join(str(v) for v in baseline_matrix["precompile_token_paddings"]),
    )
    parser.add_argument(
        "--precompile-bs-paddings",
        default=",".join(str(v) for v in baseline_matrix["precompile_bs_paddings"]),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--reuse-server-per-candidate",
        action="store_true",
        help=(
            "Start TPU server once per items_per_step candidate and reuse it across repeats. "
            "This avoids per-repeat precompile/startup churn."
        ),
    )
    parser.add_argument(
        "--warmup-requests", type=int, default=baseline_benchmark["warmup_requests"]
    )
    parser.add_argument(
        "--timed-requests", type=int, default=baseline_benchmark["timed_requests"]
    )
    parser.add_argument(
        "--concurrency", type=int, default=baseline_benchmark["concurrency"]
    )
    parser.add_argument(
        "--timeout-sec", type=int, default=baseline_benchmark["timeout_sec"]
    )
    parser.add_argument(
        "--request-retry-attempts",
        type=int,
        default=int(baseline_benchmark.get("request_retry_attempts", 3)),
        help="Per-request attempts for transient transport/HTTP failures (>=1).",
    )
    parser.add_argument(
        "--request-retry-backoff-sec",
        type=float,
        default=float(baseline_benchmark.get("request_retry_backoff_sec", 0.25)),
        help="Linear backoff base between request retries.",
    )
    parser.add_argument(
        "--stabilize-with-workload-warmup",
        action="store_true",
        help="Before profiler timed run, issue one /v1/score request per workload shape.",
    )
    parser.add_argument(
        "--stabilization-sleep-sec",
        type=float,
        default=baseline_matrix["stabilization_sleep_sec"],
        help="Optional sleep after readiness/warmup before starting profiler run.",
    )
    parser.add_argument(
        "--workload-warmup-attempts",
        type=int,
        default=baseline_matrix["workload_warmup_attempts"],
        help="Retry count per workload warmup request when --stabilize-with-workload-warmup is enabled.",
    )
    parser.add_argument(
        "--workload-warmup-backoff-sec",
        type=float,
        default=baseline_matrix["workload_warmup_backoff_sec"],
        help="Sleep between workload warmup retry attempts.",
    )
    parser.add_argument(
        "--warmup-all-shape-buckets",
        action=argparse.BooleanOptionalAction,
        default=bool(baseline_matrix.get("warmup_all_shape_buckets", True)),
        help=(
            "When workload warmup is enabled, prewarm all active shape bucket combinations "
            "derived from selected workloads."
        ),
    )
    parser.add_argument(
        "--warmup-max-shape-bucket-requests",
        type=int,
        default=int(baseline_matrix.get("warmup_max_shape_bucket_requests", 64)),
        help="Cap on number of expanded shape warmup requests per run.",
    )
    parser.add_argument(
        "--fail-on-workload-warmup-error",
        action="store_true",
        help="Fail the repeat if any workload warmup request keeps failing after retries.",
    )
    parser.add_argument("--tpu-hardware", default=baseline_tpu["hardware"])
    parser.add_argument("--tpu-cost-per-hour", type=float, default=baseline_tpu["cost_per_hour"])
    parser.add_argument("--output-base", default="results/tpu_matrix")
    parser.add_argument("--profiler-dir", default=".")
    parser.add_argument("--gate-max-failure-rate", type=float, default=0.0)
    parser.add_argument(
        "--gate-max-p99-p50",
        default=baseline_gate_csv,
        help="Comma separated list: <workload>:<max_ratio>",
    )
    parser.add_argument("--gate-max-throughput-cv", type=float, default=0.30)
    parser.add_argument("--gate-max-p99-cv", type=float, default=0.50)
    parser.add_argument("--gate-min-score-samples", type=int, default=1)
    parser.add_argument(
        "--allow-missing-cache-handle",
        action="store_true",
        help="Allow candidates with Missing scoring cache handle signals to pass gates.",
    )
    parser.add_argument(
        "--require-cache-transition-exercise",
        action=argparse.BooleanOptionalAction,
        default=bool(baseline_matrix.get("require_cache_transition_exercise", False)),
        help=(
            "Require each candidate to exercise cache-transition events "
            "(Missing scoring cache handle > 0) for compile-stability validation."
        ),
    )
    parser.add_argument(
        "--allow-timed-xla-compilation",
        action="store_true",
        help=(
            "Allow candidates with timed first-request outlier signals (first successful timed request "
            ">3x median of remaining timed requests)."
        ),
    )
    parser.add_argument(
        "--allow-shape-contract-violations",
        action="store_true",
        help="Allow shape-contract violations to pass candidate gates.",
    )
    parser.add_argument(
        "--allow-score-full-vocab-fallback",
        action=argparse.BooleanOptionalAction,
        default=bool(baseline_matrix.get("allow_score_full_vocab_fallback", False)),
        help=(
            "Allow score-path candidates that show label-only fastpath fallback or missing "
            "fastpath success coverage."
        ),
    )
    parser.add_argument(
        "--precheck-attempts",
        type=int,
        default=4,
        help="Retry attempts for per-run TPU precheck (health + /v1/score).",
    )
    parser.add_argument(
        "--precheck-backoff-sec",
        type=float,
        default=2.0,
        help="Linear backoff base between TPU precheck attempts.",
    )
    parser.add_argument(
        "--precheck-health-timeout-sec",
        type=int,
        default=90,
        help="Per-attempt timeout for /v1/models health precheck.",
    )
    parser.add_argument(
        "--precheck-score-timeout-sec",
        type=int,
        default=180,
        help="Per-attempt timeout for /v1/score precheck.",
    )
    parser.add_argument(
        "--multi-item-prefill-extend-cache-timeout-sec",
        type=float,
        default=None,
        help=(
            "Optional override for MULTI_ITEM_PREFILL_EXTEND_CACHE_TIMEOUT to stress cache "
            "hit/miss transition behavior."
        ),
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.workload_warmup_attempts < 1:
        raise ValueError("--workload-warmup-attempts must be >= 1")
    if args.warmup_max_shape_bucket_requests < 1:
        raise ValueError("--warmup-max-shape-bucket-requests must be >= 1")
    if args.lane_capacity_bump_cap < 1:
        raise ValueError("--lane-capacity-bump-cap must be >= 1")
    if args.request_retry_attempts < 1:
        raise ValueError("--request-retry-attempts must be >= 1")
    if args.request_retry_backoff_sec < 0:
        raise ValueError("--request-retry-backoff-sec must be >= 0")
    if args.precheck_attempts < 1:
        raise ValueError("--precheck-attempts must be >= 1")
    if args.precheck_health_timeout_sec < 1 or args.precheck_score_timeout_sec < 1:
        raise ValueError("--precheck-*-timeout-sec values must be >= 1")
    if args.shape_pad_token_id < 0:
        raise ValueError("--shape-pad-token-id must be >= 0")
    if args.shape_query_fill_token_id < 0 or args.shape_item_fill_token_id < 0:
        raise ValueError("--shape-*-fill-token-id must be >= 0")
    if (
        args.multi_item_prefill_extend_cache_timeout_sec is not None
        and float(args.multi_item_prefill_extend_cache_timeout_sec) < 0
    ):
        raise ValueError("--multi-item-prefill-extend-cache-timeout-sec must be >= 0")
    if args.tpu_client_location == "tpu_vm" and not str(args.tpu_client_runtime_dir).strip():
        raise ValueError("--tpu-client-runtime-dir must not be empty")

    profiler_dir = Path(args.profiler_dir).resolve()
    group_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_tpu-items-per-step-matrix")
    group_dir = (profiler_dir / args.output_base / group_name).resolve()
    cfg_dir = group_dir / "configs"
    group_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    raw_candidates = [int(x.strip()) for x in args.items_per_step.split(",") if x.strip()]
    candidates = list(raw_candidates)
    if not candidates:
        raise ValueError("No items_per_step candidates provided.")
    shape_query_buckets = parse_int_csv(
        args.shape_query_token_buckets,
        name="shape-query-token-buckets",
    )
    shape_item_buckets = parse_int_csv(
        args.shape_item_token_buckets,
        name="shape-item-token-buckets",
    )
    shape_num_items_buckets = parse_int_csv(
        args.shape_num_items_buckets,
        name="shape-num-items-buckets",
    )
    workloads = apply_shape_contract_to_workloads(
        workloads=parse_workload_filter(args.workload_filter),
        enabled=args.shape_contract_enabled,
        use_token_ids=args.shape_contract_use_token_ids,
        strict=args.shape_contract_strict,
        query_token_buckets=shape_query_buckets,
        item_token_buckets=shape_item_buckets,
        num_items_buckets=shape_num_items_buckets,
        pad_token_id=args.shape_pad_token_id,
        query_fill_token_id=args.shape_query_fill_token_id,
        item_fill_token_id=args.shape_item_fill_token_id,
    )
    candidates = maybe_align_items_per_step_candidates(
        candidates=candidates,
        workloads=workloads,
        enabled=args.align_items_per_step_with_workloads,
    )
    if not candidates:
        raise ValueError("No items_per_step candidates after alignment.")
    workload_items = {w["name"]: int(w["num_items"]) for w in workloads}
    max_running_requests = int(args.max_running_requests)
    multi_item_extend_batch_size = int(args.multi_item_extend_batch_size)
    aligned_mrr, aligned_ebs = maybe_align_lane_capacity_for_single_workload(
        max_running_requests=max_running_requests,
        multi_item_extend_batch_size=multi_item_extend_batch_size,
        workloads=workloads,
        enabled=args.auto_bump_lane_capacity_with_workload,
        cap=args.lane_capacity_bump_cap,
    )
    if (aligned_mrr, aligned_ebs) != (max_running_requests, multi_item_extend_batch_size):
        print(
            "Aligned lane capacity for single workload: "
            f"max_running_requests {max_running_requests}->{aligned_mrr}, "
            f"extend_batch_size {multi_item_extend_batch_size}->{aligned_ebs}"
        )
        max_running_requests = aligned_mrr
        multi_item_extend_batch_size = aligned_ebs
    print(f"Selected workloads: {[w['name'] for w in workloads]}")
    if candidates != sorted({int(x) for x in raw_candidates if int(x) > 0}):
        print(
            f"Aligned items_per_step candidates from {sorted(set(raw_candidates))} to {candidates}"
        )
    precompile_token_paddings = parse_int_csv(
        args.precompile_token_paddings, name="precompile-token-paddings"
    )
    precompile_bs_paddings = parse_int_csv(
        args.precompile_bs_paddings, name="precompile-bs-paddings"
    )

    gate_config = GateConfig(
        max_failure_rate=args.gate_max_failure_rate,
        max_p99_p50=parse_p99_p50_gates(args.gate_max_p99_p50),
        max_throughput_cv=args.gate_max_throughput_cv,
        max_p99_cv=args.gate_max_p99_cv,
        min_score_samples=args.gate_min_score_samples,
        allow_missing_cache_handle=args.allow_missing_cache_handle,
        allow_timed_xla_compilation=args.allow_timed_xla_compilation,
        allow_shape_contract_violations=args.allow_shape_contract_violations,
        allow_score_full_vocab_fallback=args.allow_score_full_vocab_fallback,
        require_cache_transition_exercise=args.require_cache_transition_exercise,
    )
    server_env = tpu_server_env(baseline)
    if args.multi_item_prefill_extend_cache_timeout_sec is not None:
        server_env["MULTI_ITEM_PREFILL_EXTEND_CACHE_TIMEOUT"] = str(
            float(args.multi_item_prefill_extend_cache_timeout_sec)
        )
        print(
            "Overriding MULTI_ITEM_PREFILL_EXTEND_CACHE_TIMEOUT="
            f"{server_env['MULTI_ITEM_PREFILL_EXTEND_CACHE_TIMEOUT']}"
        )
    server_static_args = tpu_server_static_args(baseline)

    tpu_url_host, tpu_url_port = parse_base_url(args.tpu_url)
    profiler_tpu_url = args.tpu_url
    if args.tpu_client_location == "tpu_vm":
        profiler_tpu_url = f"http://127.0.0.1:{int(args.tpu_port)}"
        print(
            "Using TPU VM co-located client mode: profiler backend URL set to "
            f"{profiler_tpu_url} (server precheck URL remains {args.tpu_url})"
        )
    connection_mode = resolve_connection_mode(args.tpu_url, args.tpu_connection_mode)
    tunnel_local_port = (
        int(args.tpu_tunnel_local_port)
        if args.tpu_tunnel_local_port is not None
        else int(tpu_url_port)
    )
    if tunnel_local_port != int(tpu_url_port):
        raise ValueError(
            "--tpu-tunnel-local-port must match the port used by --tpu-url "
            f"(url_port={tpu_url_port}, tunnel_local_port={tunnel_local_port})."
        )
    tunnel_remote_port = (
        int(args.tpu_tunnel_remote_port)
        if args.tpu_tunnel_remote_port is not None
        else int(args.tpu_port)
    )
    tunnel_handle = (
        TunnelHandle(mode="tunnel", local_port=tunnel_local_port, remote_port=tunnel_remote_port)
        if connection_mode == "tunnel"
        else None
    )
    print(
        f"TPU connection mode: {connection_mode} "
        f"(url_host={tpu_url_host}, url_port={tpu_url_port})"
    )

    tpu_host = args.tpu_host
    if args.ssh_mode == "direct" and not tpu_host:
        tpu_host = resolve_tpu_external_ip(args.tpu_name, args.tpu_zone, args.tpu_project)
        print(f"Resolved TPU host: {tpu_host}")

    if connection_mode == "tunnel":
        ensure_tpu_tunnel(
            handle=tunnel_handle,
            autostart=args.tpu_tunnel_autostart,
            ssh_mode=args.ssh_mode,
            tpu_name=args.tpu_name,
            tpu_zone=args.tpu_zone,
            tpu_project=args.tpu_project,
            tpu_host=tpu_host,
            ssh_user=args.ssh_user,
            ssh_key=args.ssh_key,
        )
    if args.tpu_client_location == "tpu_vm":
        prepare_tpu_vm_profiler_runtime(
            profiler_dir=profiler_dir,
            tpu_name=args.tpu_name,
            tpu_zone=args.tpu_zone,
            tpu_project=args.tpu_project,
            ssh_mode=args.ssh_mode,
            tpu_host=tpu_host,
            ssh_user=args.ssh_user,
            ssh_key=args.ssh_key,
            tpu_client_runtime_dir=args.tpu_client_runtime_dir,
        )

    runs: list[MatrixRun] = []
    for ips in candidates:
        candidate_log = f"/tmp/sgl_jax_pr28_matrix_ips{ips}_shared.log"
        candidate_log_bytes_collected = 0
        candidate_launch_error: str | None = None
        if args.reuse_server_per_candidate:
            print(f"\n=== Candidate Setup: items_per_step={ips} (reuse server across repeats) ===")
            try:
                launch_tpu_server(
                    tpu_name=args.tpu_name,
                    tpu_zone=args.tpu_zone,
                    tpu_project=args.tpu_project,
                    ssh_mode=args.ssh_mode,
                    tpu_host=tpu_host,
                    ssh_user=args.ssh_user,
                    ssh_key=args.ssh_key,
                    repo_path=args.tpu_repo_path,
                    port=args.tpu_port,
                    model=args.model,
                    items_per_step=ips,
                    max_running_requests=max_running_requests,
                    multi_item_extend_batch_size=multi_item_extend_batch_size,
                    multi_item_scoring_chunk_size=args.multi_item_scoring_chunk_size,
                    precompile_token_paddings=precompile_token_paddings,
                    precompile_bs_paddings=precompile_bs_paddings,
                    remote_log_path=candidate_log,
                    server_env=server_env,
                    server_static_args=server_static_args,
                )
                precheck_tpu_endpoint(
                    base_url=args.tpu_url,
                    model=args.model,
                    health_timeout_sec=args.precheck_health_timeout_sec,
                    score_timeout_sec=args.precheck_score_timeout_sec,
                    attempts=args.precheck_attempts,
                    backoff_sec=args.precheck_backoff_sec,
                    tunnel_handle=tunnel_handle,
                    tunnel_autostart=args.tpu_tunnel_autostart,
                    ssh_mode=args.ssh_mode,
                    tpu_name=args.tpu_name,
                    tpu_zone=args.tpu_zone,
                    tpu_project=args.tpu_project,
                    tpu_host=tpu_host,
                    ssh_user=args.ssh_user,
                    ssh_key=args.ssh_key,
                )
            except Exception as e:  # noqa: BLE001
                candidate_launch_error = str(e)
                print(
                    f"[warn] candidate setup failed for ips={ips}: "
                    f"{candidate_launch_error}"
                )

        for repeat_idx in range(1, args.repeats + 1):
            print(f"\n=== Matrix Run: items_per_step={ips}, repeat={repeat_idx}/{args.repeats} ===")
            remote_log = (
                candidate_log
                if args.reuse_server_per_candidate
                else f"/tmp/sgl_jax_pr28_matrix_ips{ips}_r{repeat_idx}.log"
            )
            cfg_path = cfg_dir / f"tpu_matrix_ips{ips}_r{repeat_idx}.yaml"
            run_error: str | None = None
            run_dir: Path | None = None
            tpu_log_path: Path | None = None
            raw_results: dict[str, Any] | None = None
            compile_signals: dict[str, Any] = {}
            shape_signals: dict[str, Any] = {}
            log_start_offset = candidate_log_bytes_collected if args.reuse_server_per_candidate else 0
            try:
                if args.reuse_server_per_candidate:
                    if candidate_launch_error is not None:
                        raise RuntimeError(
                            f"candidate setup failed for ips={ips}: {candidate_launch_error}"
                        )
                    precheck_tpu_endpoint(
                        base_url=args.tpu_url,
                        model=args.model,
                        health_timeout_sec=args.precheck_health_timeout_sec,
                        score_timeout_sec=args.precheck_score_timeout_sec,
                        attempts=args.precheck_attempts,
                        backoff_sec=args.precheck_backoff_sec,
                        tunnel_handle=tunnel_handle,
                        tunnel_autostart=args.tpu_tunnel_autostart,
                        ssh_mode=args.ssh_mode,
                        tpu_name=args.tpu_name,
                        tpu_zone=args.tpu_zone,
                        tpu_project=args.tpu_project,
                        tpu_host=tpu_host,
                        ssh_user=args.ssh_user,
                        ssh_key=args.ssh_key,
                    )
                else:
                    launch_tpu_server(
                        tpu_name=args.tpu_name,
                        tpu_zone=args.tpu_zone,
                        tpu_project=args.tpu_project,
                        ssh_mode=args.ssh_mode,
                        tpu_host=tpu_host,
                        ssh_user=args.ssh_user,
                        ssh_key=args.ssh_key,
                        repo_path=args.tpu_repo_path,
                        port=args.tpu_port,
                        model=args.model,
                        items_per_step=ips,
                        max_running_requests=max_running_requests,
                        multi_item_extend_batch_size=multi_item_extend_batch_size,
                        multi_item_scoring_chunk_size=args.multi_item_scoring_chunk_size,
                        precompile_token_paddings=precompile_token_paddings,
                        precompile_bs_paddings=precompile_bs_paddings,
                        remote_log_path=remote_log,
                        server_env=server_env,
                        server_static_args=server_static_args,
                    )
                    precheck_tpu_endpoint(
                        base_url=args.tpu_url,
                        model=args.model,
                        health_timeout_sec=args.precheck_health_timeout_sec,
                        score_timeout_sec=args.precheck_score_timeout_sec,
                        attempts=args.precheck_attempts,
                        backoff_sec=args.precheck_backoff_sec,
                        tunnel_handle=tunnel_handle,
                        tunnel_autostart=args.tpu_tunnel_autostart,
                        ssh_mode=args.ssh_mode,
                        tpu_name=args.tpu_name,
                        tpu_zone=args.tpu_zone,
                        tpu_project=args.tpu_project,
                        tpu_host=tpu_host,
                        ssh_user=args.ssh_user,
                        ssh_key=args.ssh_key,
                    )
                if args.stabilize_with_workload_warmup:
                    run_workload_shape_warmup(
                        base_url=args.tpu_url,
                        model=args.model,
                        workloads=workloads,
                        warmup_all_shape_buckets=args.warmup_all_shape_buckets,
                        max_shape_bucket_requests=args.warmup_max_shape_bucket_requests,
                        timeout_sec=max(120, args.timeout_sec),
                        attempts=args.workload_warmup_attempts,
                        backoff_sec=args.workload_warmup_backoff_sec,
                        fail_on_error=args.fail_on_workload_warmup_error,
                    )
                if args.stabilization_sleep_sec > 0:
                    time.sleep(args.stabilization_sleep_sec)

                write_config(
                    path=cfg_path,
                    experiment_name=f"pr28-tpu-matrix-ips{ips}-r{repeat_idx}",
                    model=args.model,
                    tpu_url=profiler_tpu_url,
                    hardware=args.tpu_hardware,
                    cost_per_hour=args.tpu_cost_per_hour,
                    warmup_requests=args.warmup_requests,
                    timed_requests=args.timed_requests,
                    concurrency=args.concurrency,
                    timeout_sec=args.timeout_sec,
                    request_retry_attempts=args.request_retry_attempts,
                    request_retry_backoff_sec=args.request_retry_backoff_sec,
                    workloads=workloads,
                )
                run_dir, compile_signals = run_profiler_for_config(
                    profiler_dir,
                    cfg_path,
                    group_dir / "runs",
                    client_location=args.tpu_client_location,
                    tpu_name=args.tpu_name,
                    tpu_zone=args.tpu_zone,
                    tpu_project=args.tpu_project,
                    ssh_mode=args.ssh_mode,
                    tpu_host=tpu_host,
                    ssh_user=args.ssh_user,
                    ssh_key=args.ssh_key,
                    tpu_repo_path=args.tpu_repo_path,
                    tpu_client_runtime_dir=args.tpu_client_runtime_dir,
                    tpu_client_python=args.tpu_client_python,
                )
                raw_results = json.loads((run_dir / "raw_results.json").read_text())
                shape_signals = extract_shape_contract_signals(run_dir, workloads)
            except Exception as e:  # noqa: BLE001
                run_error = str(e)
                print(f"[warn] run failed for ips={ips}, repeat={repeat_idx}: {run_error}")
                run_dir = (group_dir / "failed_runs" / f"ips{ips}_r{repeat_idx}").resolve()
                run_dir.mkdir(parents=True, exist_ok=True)
                raw_results = build_failed_raw_results(args.timed_requests, workloads)
                (run_dir / "raw_results.json").write_text(json.dumps(raw_results, indent=2))
                shape_signals = {
                    "shape_contract_violation_count": 1,
                    "shape_contract_violations": [f"run_failed_before_shape_check: {run_error}"],
                    "shape_contract_checked_workloads": [str(w['name']) for w in workloads],
                    "shape_contract_passed_workloads": [],
                }

            assert run_dir is not None
            assert raw_results is not None
            try:
                tpu_log_path, total_remote_log_size = collect_tpu_log(
                    tpu_name=args.tpu_name,
                    tpu_zone=args.tpu_zone,
                    tpu_project=args.tpu_project,
                    ssh_mode=args.ssh_mode,
                    tpu_host=tpu_host,
                    ssh_user=args.ssh_user,
                    ssh_key=args.ssh_key,
                    remote_log_path=remote_log,
                    run_dir=run_dir,
                    start_offset_bytes=log_start_offset,
                )
                if args.reuse_server_per_candidate:
                    candidate_log_bytes_collected = total_remote_log_size
            except Exception as e:  # noqa: BLE001
                log_error_path = run_dir / "artifacts" / "tpu" / "tpu_server.log"
                log_error_path.parent.mkdir(parents=True, exist_ok=True)
                log_error_path.write_text(f"[log_collection_error] {e}\n")
                tpu_log_path = log_error_path
                if run_error is None:
                    run_error = f"log collection failed: {e}"
                print(f"[warn] log collection failed for ips={ips}, repeat={repeat_idx}: {e}")

            runs.append(
                MatrixRun(
                    items_per_step=ips,
                    repeat_idx=repeat_idx,
                    run_dir=run_dir,
                    raw_results=raw_results,
                    tpu_log_path=tpu_log_path,
                    compile_signals=compile_signals,
                    shape_signals=shape_signals,
                    run_error=run_error,
                )
            )

        if args.reuse_server_per_candidate:
            try:
                stop_tpu_server(
                    tpu_name=args.tpu_name,
                    tpu_zone=args.tpu_zone,
                    tpu_project=args.tpu_project,
                    ssh_mode=args.ssh_mode,
                    tpu_host=tpu_host,
                    ssh_user=args.ssh_user,
                    ssh_key=args.ssh_key,
                    port=args.tpu_port,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[warn] candidate cleanup failed for ips={ips}: {e}")

    if tunnel_handle is not None:
        try:
            stop_tpu_tunnel(tunnel_handle)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] TPU tunnel cleanup failed: {e}")

    report_path = build_ranked_report(
        group_dir,
        runs,
        repeats=args.repeats,
        timed_requests=args.timed_requests,
        gate_config=gate_config,
        workloads=workloads,
        workload_items=workload_items,
    )
    print(f"\nMatrix group directory: {group_dir}")
    print(f"Matrix report: {report_path}")
    print(f"Matrix summary JSON: {group_dir / 'matrix_summary.json'}")
    print(f"Matrix gates JSON: {group_dir / 'matrix_gates.json'}")


if __name__ == "__main__":
    main()
