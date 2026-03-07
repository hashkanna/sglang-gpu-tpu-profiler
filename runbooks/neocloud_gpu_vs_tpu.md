# Neocloud GPU vs TPU Scoring Runbook

This runbook benchmarks:
- GPU backend: sglang main scoring API (`/v1/score`, PyTorch) on a remote neocloud VM.
- TPU backend: existing JAX scoring API (`/v1/score`) that is already running.

The workflow reuses the existing profiler/report stack and produces the same artifact format:
- `raw_results.json`
- `dashboard_data.json`
- `dashboard.html`
- `deep_summary.md`
- `deep_dive_report.md`

It does **not** stop TPU.

## 1) Prerequisites

From `/Users/kanna/Sandbox/sglang-all/sglang-gpu-tpu-profiler`:

```bash
python3 -m profiler.cli --help
python3 scripts/summarize_deep_compare.py --help
python3 scripts/generate_pr28_scoring_report.py --help
```

System tools:
- `python3`
- `curl`
- `ssh` and `scp` (if using remote hosts/tunnels)

## 2) Files Added for This Workflow

- Canonical baseline defaults:
  - `config/pr28_scoring_baseline.yaml`
- Config template:
  - `config/neocloud_gpu_vs_tpu.template.yaml`
- Config renderer:
  - `scripts/render_profiler_config.sh`
- End-to-end workflow:
  - `scripts/run_neocloud_gpu_vs_tpu.sh`

## 3) Environment Setup (PR28 parity defaults)

Required minimum:

```bash
export MODEL_NAME="Qwen/Qwen3-0.6B"
export API_ENDPOINT="/v1/score"

# GPU: either direct URL or SSH tunnel inputs.
export GPU_SSH_HOST="your-gpu-host"
export GPU_SSH_USER="ubuntu"
export GPU_SSH_KEY="$HOME/.ssh/id_ed25519"
export GPU_REMOTE_PORT="30000"
export GPU_LOCAL_PORT="30000"

# TPU: use running TPU endpoint directly (recommended if already tunneled).
export TPU_BASE_URL="http://127.0.0.1:30001"
```

Optional provider metadata:

```bash
export PROVIDER_NAME="runpod"
export PROVIDER_REGION="us-ks"
export PROVIDER_ZONE="us-ks-1"
export GPU_TYPE="L4"
```

Optional remote start/provision hooks:

```bash
# Optional: provider VM startup command. If omitted, VM must already be running.
export GPU_PROVIDER_START_CMD='echo "start VM via provider CLI/API here"'

# Optional: remote command to start GPU scoring server if not already running.
export GPU_SERVER_START_CMD='python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --host 0.0.0.0 --port 30000'
```

Optional logs/profiles collection:

```bash
export GPU_REMOTE_LOG_PATH="/tmp/sglang_gpu_server.log"
export TPU_LOG_PATH="/path/to/local/tpu_server.log"
export TPU_REMOTE_LOG_PATH="/tmp/tpu_server.log"
export GPU_REMOTE_PROFILE_DIR="/tmp/gpu_profile"
export TPU_REMOTE_PROFILE_DIR="/tmp/tpu_profile"
```

## 4) One-Command Workflow

```bash
./scripts/run_neocloud_gpu_vs_tpu.sh \
  --baseline-config config/pr28_scoring_baseline.yaml
```

What it does:
1. Optionally runs `GPU_PROVIDER_START_CMD`.
2. Optionally runs `GPU_SERVER_START_CMD` on the GPU host over SSH.
3. Opens SSH tunnels as needed.
4. Probes both endpoints via `/v1/score`.
5. Renders config from `config/neocloud_gpu_vs_tpu.template.yaml`.
6. Runs `python3 -m profiler.cli run`.
7. Generates `dashboard.html` via `python3 -m profiler.cli report --format html`.
8. Generates `deep_summary.md` and `deep_dive_report.md` using existing scripts.
9. Collects GPU/TPU logs and optional profile directories into `artifacts/`.

Results are placed in:
- `results/<timestamp>_<experiment_name>/`

## 5) Dry-Run Mode

Print all commands without execution:

```bash
./scripts/run_neocloud_gpu_vs_tpu.sh --dry-run
```

Use this to validate provisioning/tunnel/profiler command wiring before touching remote infra.

## 6) Provider Adapter Examples

These are templates; adjust for your account tooling.

### RunPod Example

```bash
export PROVIDER_NAME="runpod"
export PROVIDER_REGION="us-ks"
export GPU_TYPE="L4"
export GPU_PROVIDER_START_CMD='runpodctl start pod "$RUNPOD_POD_ID"'
export GPU_SSH_HOST="ssh.runpod.io"
export GPU_SSH_USER="root"
export GPU_SSH_KEY="$HOME/.ssh/id_ed25519"
```

### Vast.ai Example

```bash
export PROVIDER_NAME="vast"
export PROVIDER_REGION="us"
export GPU_TYPE="L40S"
export GPU_PROVIDER_START_CMD='vastai start instance "$VAST_INSTANCE_ID"'
export GPU_SSH_HOST="your-vast-instance-host"
export GPU_SSH_USER="root"
export GPU_SSH_KEY="$HOME/.ssh/id_ed25519"
```

If provider API/CLI is unavailable, skip `GPU_PROVIDER_START_CMD` and do manual startup, then run the script with `--skip-gpu-start` (or set `GPU_SERVER_START_CMD` once SSH is available).

## 7) Workload/Config Parity Notes

Template defaults match PR28 scoring profile unless overridden:
- `pr28_hotshape`: query 2000 tokens, 500 items, item 20 tokens
- `small_batch`: query 120, 10 items, item 20 tokens
- `medium_batch`: query 500, 100 items, item 20 tokens
- benchmark: warmup 3, timed 10, concurrency 1, timeout 300s
- scoring label defaults: `label_token_ids: [198]`, `apply_softmax: false`

## 8) Troubleshooting

### SSH failures
- Symptom: `Permission denied`, timeout, or tunnel creation failure.
- Checks:
  - `ssh -i "$GPU_SSH_KEY" "$GPU_SSH_USER@$GPU_SSH_HOST" 'echo ok'`
  - Verify security groups/firewall and key path.

### Tunnel failures / local port conflicts
- Symptom: tunnel exits immediately or port already in use.
- Fix:
  - Change `GPU_LOCAL_PORT` / `TPU_LOCAL_PORT`.
  - Ensure no existing process is bound to that port.

### Endpoint health check fails on `/v1/score`
- Symptom: repeated probe failures before benchmark run.
- Fix:
  - Verify model is loaded on GPU server.
  - Confirm `GPU_BASE_URL`/`TPU_BASE_URL` and `API_ENDPOINT`.
  - Re-run with `--dry-run` to inspect effective commands.

### `/stop_profile` hangs
- This workflow does not require `/stop_profile` to complete benchmark/report artifacts.
- If you manually call profiling stop endpoints, use explicit client timeouts and continue log collection even if stop hangs.

### Compile warmup outliers
- Symptom: first timed request is much slower (especially TPU XLA warmup).
- Fix:
  - Increase `WARMUP_REQUESTS`.
  - Keep benchmark parity but capture separate warmup-only runs for analysis.
  - Compare `deep_summary.md` and `deep_dive_report.md` for p99 spikes.
