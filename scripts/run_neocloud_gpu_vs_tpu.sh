#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
One-command benchmark workflow: neocloud GPU (PyTorch) vs TPU (JAX) scoring API.

Usage:
  scripts/run_neocloud_gpu_vs_tpu.sh [options]

Options:
  --config-template PATH          Config template path
                                  (default: config/neocloud_gpu_vs_tpu.template.yaml)
  --generated-config PATH         Rendered config path
                                  (default: config/generated_neocloud_gpu_vs_tpu.yaml)
  --baseline-config PATH          Canonical baseline defaults YAML
                                  (default: config/pr28_scoring_baseline.yaml)
  --output-root PATH              Results root directory (default: results)
  --deep-dive-reference-run PATH  Optional prior run dir for deep-dive ips32 lane
                                  (default: current run dir, so ips32/ips160 both point to the same run)
  --skip-gpu-start                Skip remote GPU server start command
  --dry-run                       Print all commands without executing them
  -h, --help                      Show this help

Important env vars:
  Connection:
    GPU_BASE_URL or (GPU_SSH_HOST + GPU_SSH_USER)
    TPU_BASE_URL or (TPU_SSH_HOST + TPU_SSH_USER, with tunnel enabled)
  Remote start/provision (optional):
    GPU_PROVIDER_START_CMD
    GPU_SERVER_START_CMD
  Artifacts (optional):
    GPU_REMOTE_LOG_PATH, TPU_LOG_PATH, TPU_REMOTE_LOG_PATH
    GPU_REMOTE_PROFILE_DIR, TPU_REMOTE_PROFILE_DIR
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

warn() {
  printf 'Warning: %s\n' "$*" >&2
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

print_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

run_shell_cmd() {
  local cmd="$1"
  printf '+ %s\n' "${cmd}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  bash -lc "${cmd}"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "Missing required command: $1"
  fi
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    die "Missing required env var: ${name}"
  fi
}

run_remote_shell() {
  local ssh_user="$1"
  local ssh_host="$2"
  local ssh_key="$3"
  local remote_cmd="$4"
  local -a ssh_cmd=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
  if [[ -n "${ssh_key}" ]]; then
    ssh_cmd+=(-i "${ssh_key}")
  fi
  ssh_cmd+=("${ssh_user}@${ssh_host}" "bash -lc $(printf '%q' "${remote_cmd}")")
  run_cmd "${ssh_cmd[@]}"
}

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

declare -a TUNNEL_PIDS=()

cleanup() {
  if [[ "${#TUNNEL_PIDS[@]}" -eq 0 ]]; then
    return 0
  fi
  for pid in "${TUNNEL_PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup EXIT

start_tunnel() {
  local name="$1"
  local ssh_user="$2"
  local ssh_host="$3"
  local ssh_key="$4"
  local local_port="$5"
  local remote_port="$6"

  if port_in_use "${local_port}"; then
    die "${name} local tunnel port ${local_port} is already in use. Change ${name^^}_LOCAL_PORT."
  fi

  local -a cmd=(
    ssh
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -N
    -L "${local_port}:127.0.0.1:${remote_port}"
  )
  if [[ -n "${ssh_key}" ]]; then
    cmd+=(-i "${ssh_key}")
  fi
  cmd+=("${ssh_user}@${ssh_host}")

  print_cmd "${cmd[@]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  "${cmd[@]}" &
  local pid=$!
  TUNNEL_PIDS+=("${pid}")
  sleep 1
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    die "Failed to start ${name} SSH tunnel."
  fi
  log "Started ${name} tunnel pid=${pid} localhost:${local_port} -> ${ssh_host}:${remote_port}"
}

probe_score_endpoint() {
  local name="$1"
  local base_url="$2"
  local payload="$3"
  local target="${base_url%/}${API_ENDPOINT}"
  local body_file
  body_file="$(mktemp)"

  local status
  status="$(curl -sS -m "${HEALTH_TIMEOUT_SEC}" -o "${body_file}" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -X POST \
    --data "${payload}" \
    "${target}" || true)"

  if [[ "${status}" != "200" ]]; then
    local preview
    preview="$(head -c 240 "${body_file}" | tr '\n' ' ')"
    rm -f "${body_file}"
    log "${name} probe failed (HTTP ${status}) at ${target}; response=${preview}"
    return 1
  fi

  if ! grep -Eq '"scores"|"logprobs"' "${body_file}"; then
    local preview
    preview="$(head -c 240 "${body_file}" | tr '\n' ' ')"
    rm -f "${body_file}"
    log "${name} probe failed: response missing scores/logprobs at ${target}; response=${preview}"
    return 1
  fi

  rm -f "${body_file}"
  return 0
}

wait_for_endpoint() {
  local name="$1"
  local base_url="$2"
  local payload="$3"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "Dry-run: skipping endpoint validation for ${name} (${base_url})"
    return 0
  fi

  local attempt
  for ((attempt = 1; attempt <= HEALTH_RETRIES; attempt++)); do
    if probe_score_endpoint "${name}" "${base_url}" "${payload}"; then
      log "${name} endpoint ready: ${base_url%/}${API_ENDPOINT}"
      return 0
    fi
    if (( attempt < HEALTH_RETRIES )); then
      sleep "${HEALTH_RETRY_SLEEP_SEC}"
    fi
  done

  die "${name} endpoint did not pass /v1/score probe after ${HEALTH_RETRIES} attempts. \
Check model loading, endpoint URL, and tunnel health."
}

copy_local_file_optional() {
  local label="$1"
  local src="$2"
  local dst="$3"
  if [[ -z "${src}" ]]; then
    return 0
  fi
  if [[ ! -f "${src}" ]]; then
    warn "${label} local file not found: ${src}"
    return 0
  fi
  run_cmd cp "${src}" "${dst}"
}

copy_local_dir_contents_optional() {
  local label="$1"
  local src="$2"
  local dst="$3"
  if [[ -z "${src}" ]]; then
    return 0
  fi
  if [[ ! -d "${src}" ]]; then
    warn "${label} local directory not found: ${src}"
    return 0
  fi
  run_cmd mkdir -p "${dst}"
  run_cmd cp -R "${src}/." "${dst}/"
}

copy_remote_file_optional() {
  local label="$1"
  local ssh_user="$2"
  local ssh_host="$3"
  local ssh_key="$4"
  local remote_path="$5"
  local dst="$6"
  if [[ -z "${ssh_user}" || -z "${ssh_host}" || -z "${remote_path}" ]]; then
    return 0
  fi

  local -a cmd=(scp -q)
  if [[ -n "${ssh_key}" ]]; then
    cmd+=(-i "${ssh_key}")
  fi
  cmd+=("${ssh_user}@${ssh_host}:${remote_path}" "${dst}")

  print_cmd "${cmd[@]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  if ! "${cmd[@]}"; then
    warn "Unable to copy ${label} from ${ssh_host}:${remote_path}"
  fi
}

copy_remote_dir_optional() {
  local label="$1"
  local ssh_user="$2"
  local ssh_host="$3"
  local ssh_key="$4"
  local remote_path="$5"
  local dst="$6"
  if [[ -z "${ssh_user}" || -z "${ssh_host}" || -z "${remote_path}" ]]; then
    return 0
  fi

  local -a cmd=(scp -rq)
  if [[ -n "${ssh_key}" ]]; then
    cmd+=(-i "${ssh_key}")
  fi
  cmd+=("${ssh_user}@${ssh_host}:${remote_path}" "${dst}")

  print_cmd "${cmd[@]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  if ! "${cmd[@]}"; then
    warn "Unable to copy ${label} directory from ${ssh_host}:${remote_path}"
  fi
}

DRY_RUN=0
SKIP_GPU_START=0
CONFIG_TEMPLATE="${ROOT_DIR}/config/neocloud_gpu_vs_tpu.template.yaml"
GENERATED_CONFIG="${ROOT_DIR}/config/generated_neocloud_gpu_vs_tpu.yaml"
BASELINE_CONFIG="${ROOT_DIR}/config/pr28_scoring_baseline.yaml"
OUTPUT_ROOT="${ROOT_DIR}/results"
DEEP_DIVE_REFERENCE_RUN_DIR="${DEEP_DIVE_REFERENCE_RUN_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-template)
      CONFIG_TEMPLATE="$2"
      shift 2
      ;;
    --generated-config)
      GENERATED_CONFIG="$2"
      shift 2
      ;;
    --baseline-config)
      BASELINE_CONFIG="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --deep-dive-reference-run)
      DEEP_DIVE_REFERENCE_RUN_DIR="$2"
      shift 2
      ;;
    --skip-gpu-start)
      SKIP_GPU_START=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ ! -f "${CONFIG_TEMPLATE}" ]]; then
  die "Config template not found: ${CONFIG_TEMPLATE}"
fi
if [[ ! -f "${BASELINE_CONFIG}" ]]; then
  die "Baseline config not found: ${BASELINE_CONFIG}"
fi

require_cmd python3
require_cmd curl

if ! baseline_defaults="$(python3 "${SCRIPT_DIR}/pr28_baseline.py" --baseline "${BASELINE_CONFIG}" --format shell-defaults)"; then
  die "Failed to load baseline defaults from ${BASELINE_CONFIG}"
fi
eval "${baseline_defaults}"

: "${PROVIDER_NAME:=generic}"
: "${PROVIDER_REGION:=unknown}"
: "${PROVIDER_ZONE:=unknown}"
: "${GPU_TYPE:=unknown}"

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

: "${HEALTH_TIMEOUT_SEC:=30}"
: "${HEALTH_RETRIES:=20}"
: "${HEALTH_RETRY_SLEEP_SEC:=3}"

: "${GPU_BASE_URL:=}"
: "${TPU_BASE_URL:=}"

: "${GPU_SSH_HOST:=}"
: "${GPU_SSH_USER:=}"
: "${GPU_SSH_KEY:=}"
: "${GPU_LOCAL_PORT:=30000}"
: "${GPU_REMOTE_PORT:=30000}"
: "${GPU_TUNNEL_ENABLED:=1}"

: "${TPU_SSH_HOST:=}"
: "${TPU_SSH_USER:=}"
: "${TPU_SSH_KEY:=}"
: "${TPU_LOCAL_PORT:=30001}"
: "${TPU_REMOTE_PORT:=30001}"
: "${TPU_TUNNEL_ENABLED:=1}"

: "${GPU_PROVIDER_START_CMD:=}"
: "${GPU_SERVER_START_CMD:=${GPU_BASELINE_SERVER_START_CMD:-}}"
: "${GPU_REMOTE_LOG_PATH:=/tmp/sglang_gpu_server.log}"
: "${GPU_REMOTE_PID_PATH:=/tmp/sglang_gpu_server.pid}"
: "${GPU_LOCAL_LOG_PATH:=}"
: "${GPU_REMOTE_PROFILE_DIR:=}"
: "${GPU_LOCAL_PROFILE_DIR:=}"

: "${TPU_LOG_PATH:=}"
: "${TPU_REMOTE_LOG_PATH:=}"
: "${TPU_REMOTE_PROFILE_DIR:=}"
: "${TPU_LOCAL_PROFILE_DIR:=}"

: "${GPU_COMMIT:=unknown}"
: "${TPU_COMMIT:=unknown}"

default_experiment_name="neocloud-gpu-vs-tpu-$(date -u +%Y%m%d)"
: "${EXPERIMENT_NAME:=${default_experiment_name}}"

if [[ -n "${GPU_SSH_HOST}" || -n "${TPU_SSH_HOST}" ]]; then
  require_cmd ssh
  require_cmd scp
fi

if [[ -n "${GPU_PROVIDER_START_CMD}" ]]; then
  log "Running GPU provider start command."
  run_shell_cmd "${GPU_PROVIDER_START_CMD}"
else
  log "No GPU_PROVIDER_START_CMD provided. Assuming provider VM is already running."
fi

if [[ "${SKIP_GPU_START}" -eq 0 ]]; then
  if [[ -n "${GPU_SERVER_START_CMD}" ]]; then
    require_env GPU_SSH_HOST
    require_env GPU_SSH_USER
    gpu_log_dir="$(dirname "${GPU_REMOTE_LOG_PATH}")"
    remote_start_cmd="$(cat <<EOF
set -euo pipefail
mkdir -p "${gpu_log_dir}"
if [[ -s "${GPU_REMOTE_PID_PATH}" ]] && kill -0 "\$(cat "${GPU_REMOTE_PID_PATH}")" >/dev/null 2>&1; then
  echo "GPU server already running pid=\$(cat "${GPU_REMOTE_PID_PATH}")"
else
  nohup ${GPU_SERVER_START_CMD} >"${GPU_REMOTE_LOG_PATH}" 2>&1 &
  echo \$! > "${GPU_REMOTE_PID_PATH}"
  echo "GPU server started pid=\$(cat "${GPU_REMOTE_PID_PATH}") log=${GPU_REMOTE_LOG_PATH}"
fi
EOF
)"
    log "Starting/validating remote GPU server process."
    run_remote_shell "${GPU_SSH_USER}" "${GPU_SSH_HOST}" "${GPU_SSH_KEY}" "${remote_start_cmd}"
  else
    log "No GPU_SERVER_START_CMD provided. Assuming GPU scoring server is already running."
  fi
else
  log "Skipping GPU server start per --skip-gpu-start."
fi

if [[ -z "${GPU_BASE_URL}" ]]; then
  if [[ "${GPU_TUNNEL_ENABLED}" != "1" ]]; then
    die "GPU_BASE_URL is empty and GPU_TUNNEL_ENABLED != 1. Provide GPU_BASE_URL or enable SSH tunnel."
  fi
  require_env GPU_SSH_HOST
  require_env GPU_SSH_USER
  start_tunnel "gpu" "${GPU_SSH_USER}" "${GPU_SSH_HOST}" "${GPU_SSH_KEY}" "${GPU_LOCAL_PORT}" "${GPU_REMOTE_PORT}"
  GPU_URL="http://127.0.0.1:${GPU_LOCAL_PORT}"
else
  GPU_URL="${GPU_BASE_URL}"
fi

if [[ -z "${TPU_BASE_URL}" ]]; then
  if [[ "${TPU_TUNNEL_ENABLED}" != "1" ]]; then
    die "TPU_BASE_URL is empty and TPU_TUNNEL_ENABLED != 1. Provide TPU_BASE_URL or enable SSH tunnel."
  fi
  require_env TPU_SSH_HOST
  require_env TPU_SSH_USER
  start_tunnel "tpu" "${TPU_SSH_USER}" "${TPU_SSH_HOST}" "${TPU_SSH_KEY}" "${TPU_LOCAL_PORT}" "${TPU_REMOTE_PORT}"
  TPU_URL="http://127.0.0.1:${TPU_LOCAL_PORT}"
else
  TPU_URL="${TPU_BASE_URL}"
fi

health_label_ids="${HEALTHCHECK_LABEL_TOKEN_IDS:-${LABEL_TOKEN_IDS}}"
health_apply_softmax="${HEALTHCHECK_APPLY_SOFTMAX:-${APPLY_SOFTMAX}}"
health_payload="$(cat <<EOF
{"model":"${MODEL_NAME}","query":"healthcheck","items":["healthcheck"],"label_token_ids":[${health_label_ids}],"apply_softmax":${health_apply_softmax}}
EOF
)"

wait_for_endpoint "GPU" "${GPU_URL}" "${health_payload}"
wait_for_endpoint "TPU" "${TPU_URL}" "${health_payload}"

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

log "Rendering profiler config from template."
run_cmd "${SCRIPT_DIR}/render_profiler_config.sh" --template "${CONFIG_TEMPLATE}" --output "${GENERATED_CONFIG}"

run_dir=""
profiler_cli_output_log=""
if [[ "${DRY_RUN}" -eq 1 ]]; then
  run_dir="${OUTPUT_ROOT}/DRY_RUN_${EXPERIMENT_NAME}"
  print_cmd python3 -m profiler.cli run --config "${GENERATED_CONFIG}" --output "${OUTPUT_ROOT}"
else
  mkdir -p "${OUTPUT_ROOT}"
  profiler_cli_output_log="$(mktemp)"
  print_cmd python3 -m profiler.cli run --config "${GENERATED_CONFIG}" --output "${OUTPUT_ROOT}"
  python3 -m profiler.cli run --config "${GENERATED_CONFIG}" --output "${OUTPUT_ROOT}" | tee "${profiler_cli_output_log}"
  run_dir="$(awk -F': ' '/^Results directory:/ {print $2}' "${profiler_cli_output_log}" | tail -n 1)"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    die "Could not resolve run directory from profiler output."
  fi
fi

gpu_artifacts_dir="${run_dir}/artifacts/gpu"
tpu_artifacts_dir="${run_dir}/artifacts/tpu"
run_cmd mkdir -p "${gpu_artifacts_dir}/profile" "${tpu_artifacts_dir}/profile"

if [[ -n "${profiler_cli_output_log}" ]]; then
  run_cmd cp "${profiler_cli_output_log}" "${run_dir}/artifacts/profiler_cli_run.log"
  rm -f "${profiler_cli_output_log}"
fi

gpu_log_dst="${gpu_artifacts_dir}/gpu_server.log"
tpu_log_dst="${tpu_artifacts_dir}/tpu_server.log"

# GPU logs/profiles
if [[ -n "${GPU_SSH_HOST}" ]]; then
  copy_remote_file_optional "GPU log" "${GPU_SSH_USER}" "${GPU_SSH_HOST}" "${GPU_SSH_KEY}" "${GPU_REMOTE_LOG_PATH}" "${gpu_log_dst}"
  copy_remote_dir_optional "GPU profile" "${GPU_SSH_USER}" "${GPU_SSH_HOST}" "${GPU_SSH_KEY}" "${GPU_REMOTE_PROFILE_DIR}" "${gpu_artifacts_dir}/profile/"
else
  copy_local_file_optional "GPU log" "${GPU_LOCAL_LOG_PATH}" "${gpu_log_dst}"
  copy_local_dir_contents_optional "GPU profile" "${GPU_LOCAL_PROFILE_DIR}" "${gpu_artifacts_dir}/profile"
fi

# TPU logs/profiles (TPU server already running; this workflow does not stop TPU)
copy_local_file_optional "TPU log" "${TPU_LOG_PATH}" "${tpu_log_dst}"
copy_local_dir_contents_optional "TPU profile" "${TPU_LOCAL_PROFILE_DIR}" "${tpu_artifacts_dir}/profile"
copy_remote_file_optional "TPU log" "${TPU_SSH_USER}" "${TPU_SSH_HOST}" "${TPU_SSH_KEY}" "${TPU_REMOTE_LOG_PATH}" "${tpu_log_dst}"
copy_remote_dir_optional "TPU profile" "${TPU_SSH_USER}" "${TPU_SSH_HOST}" "${TPU_SSH_KEY}" "${TPU_REMOTE_PROFILE_DIR}" "${tpu_artifacts_dir}/profile/"

run_cmd python3 -m profiler.cli report "${run_dir}" --config "${GENERATED_CONFIG}" --format html
run_cmd python3 scripts/summarize_deep_compare.py "${run_dir}" --output "${run_dir}/deep_summary.md"

ips32_run="${DEEP_DIVE_REFERENCE_RUN_DIR:-${run_dir}}"
if [[ "${DRY_RUN}" -eq 0 && ! -d "${ips32_run}" ]]; then
  warn "Deep-dive reference run not found: ${ips32_run}. Falling back to current run."
  ips32_run="${run_dir}"
fi
run_cmd python3 scripts/generate_pr28_scoring_report.py \
  --ips32-run "${ips32_run}" \
  --ips160-run "${run_dir}" \
  --gpu-commit "${GPU_COMMIT}" \
  --tpu-commit "${TPU_COMMIT}" \
  --output "${run_dir}/deep_dive_report.md"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  for required_file in raw_results.json dashboard_data.json dashboard.html deep_summary.md deep_dive_report.md; do
    if [[ ! -f "${run_dir}/${required_file}" ]]; then
      die "Expected output missing: ${run_dir}/${required_file}"
    fi
  done
fi

log "Workflow complete."
log "Run directory: ${run_dir}"
log "Primary artifacts:"
log "  - ${run_dir}/raw_results.json"
log "  - ${run_dir}/dashboard_data.json"
log "  - ${run_dir}/dashboard.html"
log "  - ${run_dir}/deep_summary.md"
log "  - ${run_dir}/deep_dive_report.md"
