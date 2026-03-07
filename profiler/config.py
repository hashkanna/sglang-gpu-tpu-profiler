"""Configuration loading and validation for the profiler framework."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ExperimentConfig:
    name: str
    model: str


@dataclass
class BackendConfig:
    key: str  # "gpu" or "tpu"
    name: str
    url: str
    hardware: str
    cost_per_hour: float


@dataclass
class ApiConfig:
    endpoint: str = "/v1/score"


@dataclass
class WorkloadConfig:
    name: str
    query_tokens: int
    num_items: int
    item_tokens: int
    label_token_ids: list[int]
    apply_softmax: bool = True
    # Optional strict shape contract fields for /v1/score serving-time stability.
    use_token_ids: bool = False
    enforce_shape_contract: bool = False
    query_token_buckets: list[int] = field(default_factory=list)
    item_token_buckets: list[int] = field(default_factory=list)
    num_items_buckets: list[int] = field(default_factory=list)
    pad_token_id: int = 0
    query_fill_token_id: int = 42
    item_fill_token_id: int = 84


@dataclass
class BenchmarkConfig:
    warmup_requests: int = 3
    timed_requests: int = 10
    concurrency: int = 1
    timeout_sec: int = 300
    request_retry_attempts: int = 3
    request_retry_backoff_sec: float = 0.25


@dataclass
class ProfilerConfig:
    experiment: ExperimentConfig
    backends: dict[str, BackendConfig]
    api: ApiConfig
    workloads: list[WorkloadConfig]
    benchmark: BenchmarkConfig

    def get_backend(self, key: str) -> BackendConfig:
        if key not in self.backends:
            raise KeyError(
                f"Backend '{key}' not found. Available: {list(self.backends.keys())}"
            )
        return self.backends[key]

    @property
    def backend_keys(self) -> list[str]:
        return list(self.backends.keys())


def load_config(path: str | Path) -> ProfilerConfig:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    _validate_raw(raw)

    experiment = ExperimentConfig(**raw["experiment"])

    backends = {}
    for key, bdata in raw["backends"].items():
        backends[key] = BackendConfig(key=key, **bdata)

    api = ApiConfig(**raw.get("api", {}))

    workload_field_names = {f.name for f in fields(WorkloadConfig)}
    workloads = [
        WorkloadConfig(**{k: v for k, v in w.items() if k in workload_field_names})
        for w in raw["workloads"]
    ]

    benchmark = BenchmarkConfig(**raw.get("benchmark", {}))

    return ProfilerConfig(
        experiment=experiment,
        backends=backends,
        api=api,
        workloads=workloads,
        benchmark=benchmark,
    )


def _validate_raw(raw: dict) -> None:
    """Validate required top-level keys and structure."""
    required = ["experiment", "backends", "workloads"]
    for key in required:
        if key not in raw:
            raise ValueError(f"Missing required config key: '{key}'")

    if not raw["backends"]:
        raise ValueError("At least one backend must be configured")

    if not raw["workloads"]:
        raise ValueError("At least one workload must be configured")

    exp = raw["experiment"]
    for field_name in ("name", "model"):
        if field_name not in exp:
            raise ValueError(f"experiment.{field_name} is required")

    for key, bdata in raw["backends"].items():
        for field_name in ("name", "url", "hardware", "cost_per_hour"):
            if field_name not in bdata:
                raise ValueError(f"backends.{key}.{field_name} is required")

    for i, w in enumerate(raw["workloads"]):
        for field_name in ("name", "query_tokens", "num_items", "item_tokens", "label_token_ids"):
            if field_name not in w:
                raise ValueError(f"workloads[{i}].{field_name} is required")
        _validate_workload_shape_contract(w, i)


def _validate_workload_shape_contract(workload_raw: dict, idx: int) -> None:
    """Validate optional shape-contract fields for one workload."""
    for key in ("query_tokens", "num_items", "item_tokens"):
        value = workload_raw.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"workloads[{idx}].{key} must be a positive integer")

    for bucket_key in ("query_token_buckets", "item_token_buckets", "num_items_buckets"):
        if bucket_key not in workload_raw:
            continue
        buckets = workload_raw[bucket_key]
        if not isinstance(buckets, list) or not buckets:
            raise ValueError(f"workloads[{idx}].{bucket_key} must be a non-empty list[int]")
        if any((not isinstance(v, int) or v <= 0) for v in buckets):
            raise ValueError(f"workloads[{idx}].{bucket_key} must contain positive integers")

    for token_key in ("pad_token_id", "query_fill_token_id", "item_fill_token_id"):
        if token_key not in workload_raw:
            continue
        value = workload_raw[token_key]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"workloads[{idx}].{token_key} must be a non-negative integer")

    use_token_ids = bool(workload_raw.get("use_token_ids", False))
    enforce_shape_contract = bool(workload_raw.get("enforce_shape_contract", False))
    if enforce_shape_contract and not use_token_ids:
        raise ValueError(
            f"workloads[{idx}] enforce_shape_contract=true requires use_token_ids=true"
        )
