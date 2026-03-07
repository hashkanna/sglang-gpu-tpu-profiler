# Repeatable TPU JAX Scoring Soak vs GPU PyTorch Baseline

This runbook reproduces a deep, side-by-side comparison for `/v1/score`:

- TPU: `sglang-jax` PR28-style server args on TPU v6e-1
- GPU: prior `sglang` main run(s) on L4 PyTorch used as baseline

## 1) Run TPU soak matrix (single candidate, repeated)

```bash
cd /Users/kanna/Sandbox/sglang-all/sglang-gpu-tpu-profiler

python3 scripts/run_tpu_scoring_matrix.py \
  --baseline-config config/pr28_scoring_baseline.yaml \
  --tpu-name mi-bench-kannacl \
  --tpu-zone us-east5-b \
  --ssh-mode direct \
  --tpu-connection-mode tunnel \
  --tpu-tunnel-autostart \
  --tpu-host 34.162.50.81 \
  --ssh-user kanna \
  --ssh-key /Users/kanna/.ssh/google_compute_engine \
  --tpu-repo-path /home/kanna/work/sglang-jax-pr27-prof \
  --tpu-url http://127.0.0.1:30001 \
  --items-per-step 96 \
  --max-running-requests 96 \
  --multi-item-extend-batch-size 64 \
  --precompile-token-paddings 1024,2048,4096,16384 \
  --precompile-bs-paddings 24,96 \
  --repeats 10 \
  --warmup-requests 3 \
  --timed-requests 10 \
  --concurrency 1 \
  --timeout-sec 300 \
  --stabilize-with-workload-warmup \
  --workload-warmup-attempts 5 \
  --workload-warmup-backoff-sec 2 \
  --stabilization-sleep-sec 10 \
  --output-base results/tpu_tuning/soak_mr96_bs96_r10
```

`config/pr28_scoring_baseline.yaml` is the canonical PR28 scoring baseline for
workloads, benchmark defaults, and TPU server env/arg defaults.

Connection policy notes:
- `--tpu-connection-mode tunnel` + `--tpu-tunnel-autostart` enforces loopback
  URL access with managed SSH forwarding.
- Before every repeat, runner prechecks both `/v1/models` and `/v1/score`.

Matrix outputs:

- `matrix_report.md`
- `matrix_summary.json`
- `matrix_gates.json`
- `runs/*/raw_results.json`
- `runs/*/artifacts/tpu/tpu_server.log`

## 2) Generate TPU soak vs GPU baseline deep report

Use one or more prior GPU run directories as baseline.

```bash
python3 scripts/generate_soak_vs_gpu_report.py \
  --matrix-group-dir /Users/kanna/Sandbox/sglang-all/sglang-gpu-tpu-profiler/results/tpu_tuning/soak_mr96_bs96_r10/<matrix_group_dir> \
  --gpu-run /Users/kanna/Sandbox/sglang-all/sglang-gpu-tpu-profiler/results/20260227_084034_pr28-vs-main-l4-v6e1-ips32 \
  --gpu-run /Users/kanna/Sandbox/sglang-all/sglang-gpu-tpu-profiler/results/20260227_084718_pr28-vs-main-l4-v6e1-ips160
```

Outputs in `<matrix_group_dir>/soak_vs_gpu_report/`:

- `summary.json` (machine-readable source of truth)
- `deep_dive_soak_vs_gpu.md` (human-readable report)
- `dashboard_static.html` (no React/CDN dependency; safe local viewing)

## 3) View report

```bash
cd /Users/kanna/Sandbox/sglang-all/sglang-gpu-tpu-profiler
python3 -m http.server 8008
```

Open:

- `http://localhost:8008/results/tpu_tuning/soak_mr96_bs96_r10/<matrix_group_dir>/soak_vs_gpu_report/dashboard_static.html`
- `http://localhost:8008/results/tpu_tuning/soak_mr96_bs96_r10/<matrix_group_dir>/soak_vs_gpu_report/deep_dive_soak_vs_gpu.md`

## 4) Key interpretation fields

From `summary.json`:

- `tpu.workloads.*.success_rate`: repeat-level reliability under soak
- `tpu.workloads.*.throughput_success_median_items_per_sec`: steady performance when successful
- `tpu.workloads.*.throughput_all_mean_items_per_sec`: effective delivered throughput including failed repeats
- `tpu.score_phase.*.utilization_pct`: realized vs score-only theoretical throughput
- `tpu.signals.dominant_new_token`: scheduling chunk cadence signal
- `tpu.signals.missing_cache_handle_total`: cache-handle reliability regression guardrail
