#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Render a profiler YAML config from env vars.

Usage:
  scripts/render_profiler_config.sh [--template PATH] [--output PATH] [--baseline-config PATH]

Options:
  --template PATH   YAML template path (default: config/neocloud_gpu_vs_tpu.template.yaml)
  --output PATH     Output YAML path (default: config/generated_neocloud_gpu_vs_tpu.yaml)
  --baseline-config PATH
                   Canonical baseline YAML (default: config/pr28_scoring_baseline.yaml)
  -h, --help        Show this help
EOF
}

template_path="${ROOT_DIR}/config/neocloud_gpu_vs_tpu.template.yaml"
output_path="${ROOT_DIR}/config/generated_neocloud_gpu_vs_tpu.yaml"
baseline_path="${ROOT_DIR}/config/pr28_scoring_baseline.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --template)
      template_path="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
      shift 2
      ;;
    --baseline-config)
      baseline_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${template_path}" ]]; then
  echo "Template not found: ${template_path}" >&2
  exit 1
fi
if [[ ! -f "${baseline_path}" ]]; then
  echo "Baseline config not found: ${baseline_path}" >&2
  exit 1
fi

if ! baseline_defaults="$(python3 "${SCRIPT_DIR}/pr28_baseline.py" --baseline "${baseline_path}" --format shell-defaults)"; then
  echo "Failed to load baseline defaults from: ${baseline_path}" >&2
  exit 1
fi
eval "${baseline_defaults}"

# Fallbacks when baseline does not set a variable.
: "${EXPERIMENT_NAME:=neocloud-gpu-vs-tpu}"
: "${MODEL_NAME:=Qwen/Qwen3-0.6B}"
: "${API_ENDPOINT:=/v1/score}"

: "${GPU_BACKEND_NAME:=PyTorch/GPU (sglang main)}"
: "${TPU_BACKEND_NAME:=JAX/TPU scoring API}"
: "${GPU_HARDWARE:=NVIDIA L4}"
: "${TPU_HARDWARE:=TPU v6e-1}"
: "${GPU_COST_PER_HOUR:=0.95}"
: "${TPU_COST_PER_HOUR:=1.25}"

: "${PR28_HOTSHAPE_QUERY_TOKENS:=2000}"
: "${PR28_HOTSHAPE_NUM_ITEMS:=500}"
: "${PR28_HOTSHAPE_ITEM_TOKENS:=20}"
: "${SMALL_BATCH_QUERY_TOKENS:=120}"
: "${SMALL_BATCH_NUM_ITEMS:=10}"
: "${SMALL_BATCH_ITEM_TOKENS:=20}"
: "${MEDIUM_BATCH_QUERY_TOKENS:=500}"
: "${MEDIUM_BATCH_NUM_ITEMS:=100}"
: "${MEDIUM_BATCH_ITEM_TOKENS:=20}"
: "${LABEL_TOKEN_IDS:=198}"
: "${APPLY_SOFTMAX:=false}"

: "${WARMUP_REQUESTS:=3}"
: "${TIMED_REQUESTS:=10}"
: "${CONCURRENCY:=1}"
: "${TIMEOUT_SEC:=300}"

: "${PROVIDER_NAME:=generic}"
: "${PROVIDER_REGION:=unknown}"
: "${PROVIDER_ZONE:=unknown}"
: "${GPU_TYPE:=unknown}"
: "${GPU_SSH_HOST:=}"

if [[ -z "${GPU_URL:-}" ]]; then
  echo "GPU_URL is required (example: http://127.0.0.1:30000)" >&2
  exit 1
fi
if [[ -z "${TPU_URL:-}" ]]; then
  echo "TPU_URL is required (example: http://127.0.0.1:30001)" >&2
  exit 1
fi

export EXPERIMENT_NAME MODEL_NAME API_ENDPOINT
export GPU_BACKEND_NAME TPU_BACKEND_NAME GPU_HARDWARE TPU_HARDWARE
export GPU_COST_PER_HOUR TPU_COST_PER_HOUR
export PR28_HOTSHAPE_QUERY_TOKENS PR28_HOTSHAPE_NUM_ITEMS PR28_HOTSHAPE_ITEM_TOKENS
export SMALL_BATCH_QUERY_TOKENS SMALL_BATCH_NUM_ITEMS SMALL_BATCH_ITEM_TOKENS
export MEDIUM_BATCH_QUERY_TOKENS MEDIUM_BATCH_NUM_ITEMS MEDIUM_BATCH_ITEM_TOKENS
export LABEL_TOKEN_IDS APPLY_SOFTMAX
export WARMUP_REQUESTS TIMED_REQUESTS CONCURRENCY TIMEOUT_SEC
export PROVIDER_NAME PROVIDER_REGION PROVIDER_ZONE GPU_TYPE GPU_SSH_HOST
export GPU_URL TPU_URL

mkdir -p "$(dirname "${output_path}")"

if command -v envsubst >/dev/null 2>&1; then
  envsubst < "${template_path}" > "${output_path}"
else
  python3 - "${template_path}" "${output_path}" <<'PY'
import os
import re
import sys

template_path = sys.argv[1]
output_path = sys.argv[2]
template = open(template_path, "r", encoding="utf-8").read()
pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def replace(match: re.Match[str]) -> str:
    key = match.group(1)
    if key not in os.environ:
        raise KeyError(key)
    return os.environ[key]

try:
    rendered = pattern.sub(replace, template)
except KeyError as exc:
    print(f"Missing env var for template substitution: {exc}", file=sys.stderr)
    sys.exit(1)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(rendered)
PY
fi

PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python3 - "${output_path}" <<'PY'
import sys
from profiler.config import load_config

path = sys.argv[1]
load_config(path)
print(path)
PY
