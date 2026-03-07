#!/usr/bin/env bash
set -euo pipefail

# Run matrix benchmark, then enforce regression gates.
# Usage example:
#   scripts/run_tpu_regression_nightly.sh \
#     --baseline-config config/prod_scenario_scoring_baseline_t67_label_only_fused_candidate.yaml \
#     --tpu-name mi-bench-kannacl \
#     --tpu-zone us-east5-a \
#     --ssh-mode direct \
#     --tpu-repo-path /home/kanna/work/sglang-jax-pr27-prof \
#     --tpu-url http://127.0.0.1:30001 \
#     --tpu-connection-mode tunnel

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TMP_LOG="$(mktemp)"
cleanup() { rm -f "${TMP_LOG}"; }
trap cleanup EXIT

python3 scripts/run_tpu_scoring_matrix.py \
  --baseline-config "config/prod_scenario_scoring_baseline_t67_label_only_fused_candidate.yaml" \
  --concurrency 16 \
  --timed-requests 80 \
  --repeats 3 \
  --tpu-client-location tpu_vm \
  --output-base "results/tpu_tuning/nightly_regression" \
  --profiler-dir "." \
  "$@" | tee "${TMP_LOG}"

MATRIX_GROUP_DIR="$(grep -E 'Matrix group directory:' "${TMP_LOG}" | tail -1 | sed 's/.*Matrix group directory:[[:space:]]*//')"
if [[ -z "${MATRIX_GROUP_DIR}" ]]; then
  echo "Failed to parse matrix group directory from runner output." >&2
  exit 1
fi

MATRIX_SUMMARY="${MATRIX_GROUP_DIR}/matrix_summary.json"
GATE_RESULT="${MATRIX_GROUP_DIR}/regression_gate_result.json"

python3 scripts/check_tpu_regression_gates.py \
  --matrix-summary "${MATRIX_SUMMARY}" \
  --gates-config "config/tpu_regression_gates.yaml" \
  --output-json "${GATE_RESULT}"

echo "Regression gate result: ${GATE_RESULT}"
