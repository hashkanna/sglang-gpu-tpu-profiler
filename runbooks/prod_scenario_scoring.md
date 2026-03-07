# Production-Scenario Scoring Repro Runbook (TPU JAX vs RTX Pro 6000 GPU)

This runbook is the canonical apples-to-apples contract used for PR28 production-scenario scoring comparisons.

## Scope and Contract

- Model: `Qwen/Qwen3-0.6B`
- Endpoint: `/v1/score`
- Benchmark window (both TPU and GPU):
  - `warmup_requests=3`
  - `timed_requests=10`
  - `concurrency=1`
  - `timeout_sec=300`
- Canonical workload contract: `config/prod_scenario_scoring_baseline.yaml`
- Source scenario normalization: `config/prod_scenarios_contract.yaml`

## Canonical Workloads

| Workload | Scenario | query_tokens | num_items | item_tokens | total_tokens_expected |
|---|---|---:|---:|---:|---:|
| `track_low` | Track | 183 | 16 | 26 | 599 |
| `track_mean` | Track | 420 | 96 | 55 | 5700 |
| `track_high` | Track | 483 | 128 | 63 | 8547 |
| `home_low` | Home | 155 | 8 | 30 | 395 |
| `home_mean` | Home | 275 | 34 | 50 | 1975 |
| `home_high` | Home | 2375 | 256 | 70 | 20295 |

## TPU Runtime Contract (PR28 Baseline)

Required TPU env vars:

```bash
export SGLANG_RPA_KERNEL_V11=1
export SGLANG_SCORE_FROM_CACHE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE=1
export MULTI_ITEM_ENABLE_SCORE_FROM_CACHE_V2=1
export MULTI_ITEM_DISABLE_OVERLAP_SCHEDULE=1
export MULTI_ITEM_MASK_IMPL=dense
export MULTI_ITEM_SEGMENT_FALLBACK_THRESHOLD=0
export MULTI_ITEM_SCORE_FASTPATH_LOG_METRICS=1
export MULTI_ITEM_SCORE_LABEL_ONLY_LOGPROB=1
```

Canonical TPU matrix knobs for final comparison run:

```bash
export MULTI_ITEM_EXTEND_BATCH_SIZE=64
export MULTI_ITEM_EXTEND_MAX_RUNNING_REQUESTS=96
export MULTI_ITEM_SCORE_FROM_CACHE_V2_ITEMS_PER_STEP=128
export MULTI_ITEM_EXTEND_PRECOMPILE_BS_PADDINGS=24,64,96
```

Canonical TPU static args (from baseline):

```text
--trust-remote-code --host 0.0.0.0 --device tpu --dtype bfloat16 --tp-size 1
--mem-fraction-static 0.7 --max-prefill-tokens 32768 --chunked-prefill-size -1
--disable-overlap-schedule --attention-backend fa --page-size 64
--multi-item-scoring-delimiter 151643 --max-multi-item-seq-len 32768
--max-multi-item-count 1024 --multi-item-mask-impl dense
--multi-item-segment-fallback-threshold 0 --multi-item-enable-prefill-extend
--multi-item-enable-score-from-cache-v2 --multi-item-score-label-only-logprob
--multi-item-score-fastpath-log-metrics --enable-scoring-cache --skip-server-warmup
--log-level info
```

## Repro Commands

Run from:

```bash
cd /Users/kanna/Sandbox/sglang-all/sglang-gpu-tpu-profiler
```

### 1) TPU baseline matrix (production contract, final-comparison shape)

```bash
python3 scripts/run_tpu_scoring_matrix.py \
  --baseline-config config/prod_scenario_scoring_baseline.yaml \
  --tpu-name mi-bench-kannacl \
  --tpu-zone us-east5-a \
  --ssh-mode direct \
  --tpu-host 34.153.28.4 \
  --output-base results/tpu_tuning/t29_prod_tpu_baseline_c1 \
  --workload-filter track_low,track_mean,track_high,home_low,home_mean,home_high \
  --items-per-step 128 \
  --repeats 2 \
  --warmup-requests 3 \
  --timed-requests 10 \
  --concurrency 1 \
  --reuse-server-per-candidate \
  --stabilize-with-workload-warmup \
  --warmup-all-shape-buckets \
  --warmup-max-shape-bucket-requests 64
```

Expected hygiene gate in `matrix_gates.json`:

- `gate_pass=true`
- `timed_xla_compile_total=0`
- `shape_contract_violation_total=0`

### 2) GPU baseline refresh (same contract)

Launch GPU server (Runpod/RTX Pro 6000):

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 30000
```

Run contract benchmark against GPU server:

```bash
python3 scripts/run_gpu_prod_contract_benchmark.py \
  --baseline-config config/prod_scenario_scoring_baseline.yaml \
  --url http://127.0.0.1:30000 \
  --warmup-requests 3 \
  --timed-requests 10 \
  --timeout-sec 300 \
  --out results/20260301_prodshape_gpu_baseline_rtxpro6000/raw_results_run1.json
```

Repeat once more and aggregate as `raw_results.json` (same schema as T28 artifacts).

### 3) Generate final TPU vs GPU report

```bash
python3 scripts/generate_prod_final_report.py \
  --baseline-config config/prod_scenario_scoring_baseline.yaml \
  --tpu-summary results/tpu_tuning/t29_prod_tpu_baseline_c1/20260301_123909_tpu-items-per-step-matrix/matrix_summary.json \
  --tpu-gates results/tpu_tuning/t29_prod_tpu_baseline_c1/20260301_123909_tpu-items-per-step-matrix/matrix_gates.json \
  --gpu-results results/20260301_prodshape_gpu_baseline_rtxpro6000/raw_results.json \
  --items-per-step 128 \
  --json-out results/task_artifacts/20260301_t29_prod_final_report/final_report.json \
  --md-out results/task_artifacts/20260301_t29_prod_final_report/final_report.md
```

### 4) Optional capacity model

```bash
python3 scripts/generate_capacity_plan.py \
  --final-report results/task_artifacts/20260301_t29_prod_final_report/final_report.json \
  --json-out results/task_artifacts/20260301_t30_capacity_model/capacity_plan.json \
  --md-out results/task_artifacts/20260301_t30_capacity_model/capacity_plan.md
```

## Current Reference Results (from T29)

Per-workload actual numbers:

| Workload | TPU throughput (items/s) | GPU throughput (items/s) | TPU p50 (ms) | GPU p50 (ms) | TPU p99 (ms) | GPU p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `home_high` | 423.678 | 368.731 | 598.529 | 671.695 | 673.795 | 862.949 |
| `home_low` | 30.731 | 536.369 | 255.836 | 14.815 | 278.902 | 15.573 |
| `home_mean` | 117.606 | 1361.416 | 272.788 | 25.012 | 410.489 | 25.353 |
| `track_high` | 319.793 | 1450.209 | 395.845 | 87.796 | 412.594 | 91.565 |
| `track_low` | 56.174 | 1008.165 | 278.007 | 15.818 | 352.125 | 16.550 |
| `track_mean` | 260.570 | 1517.486 | 358.551 | 63.440 | 410.858 | 66.526 |

Aggregate means:

- TPU: throughput `201.425 items/s`, p50 `359.926 ms`, p99 `423.127 ms`
- GPU: throughput `1040.396 items/s`, p50 `146.429 ms`, p99 `179.753 ms`

## Apples-to-Apples Checklist

- Use exactly `config/prod_scenario_scoring_baseline.yaml` workloads.
- Keep benchmark window identical (`3/10/c1/300s`) for TPU and GPU.
- For TPU, run benchmark client co-located on TPU VM for latency/throughput truth.
- Do not mix PR27 hotshape contract results with this production contract.
- Require compile/shape hygiene gates on TPU timed window.
- Report actual numbers (items/s, p50, p99, failures), not percentage-only summaries.

## Client Location Policy (T32)

- Policy: `USE_COLOCATED_CLIENT_FOR_TPU_BENCHMARKS`
- Reason: off-VM client path materially inflates TPU latency and depresses throughput.
- Measured A/B (same server + workloads):
  - mean throughput: local client `101.087 items/s` vs co-located `469.631 items/s` (`4.646x`)
  - mean p50: local client `774.976 ms` vs co-located `155.680 ms` (`4.978x` reduction)
  - mean p99: local client `847.700 ms` vs co-located `157.131 ms` (`5.395x` reduction)
- Evidence: `results/task_artifacts/20260301_t32_client_colocation_ab/report.md`

## Copy/Paste PR Description Block (Actual Numbers)

```markdown
### Production-shaped scoring benchmark contract (Track + Home)
- Model: `Qwen/Qwen3-0.6B`
- Workloads: `track_low`, `track_mean`, `track_high`, `home_low`, `home_mean`, `home_high`
- Benchmark window (TPU and GPU): warmup=3, timed=10, concurrency=1, timeout=300s
- TPU hygiene: `timed_xla_compile_total=0`, `shape_contract_violation_total=0`, `missing_cache_handle_total=0`

### Actual per-workload results (items/s, p50 ms, p99 ms)
| Workload | TPU throughput | GPU throughput | TPU p50 | GPU p50 | TPU p99 | GPU p99 |
|---|---:|---:|---:|---:|---:|---:|
| home_high | 423.678 | 368.731 | 598.529 | 671.695 | 673.795 | 862.949 |
| home_low | 30.731 | 536.369 | 255.836 | 14.815 | 278.902 | 15.573 |
| home_mean | 117.606 | 1361.416 | 272.788 | 25.012 | 410.489 | 25.353 |
| track_high | 319.793 | 1450.209 | 395.845 | 87.796 | 412.594 | 91.565 |
| track_low | 56.174 | 1008.165 | 278.007 | 15.818 | 352.125 | 16.550 |
| track_mean | 260.570 | 1517.486 | 358.551 | 63.440 | 410.858 | 66.526 |

### Aggregate means
- TPU: throughput 201.425 items/s, p50 359.926 ms, p99 423.127 ms
- GPU: throughput 1040.396 items/s, p50 146.429 ms, p99 179.753 ms
```

## Evidence Paths

- Contract config: `config/prod_scenarios_contract.yaml`
- Baseline config: `config/prod_scenario_scoring_baseline.yaml`
- GPU benchmark utility: `scripts/run_gpu_prod_contract_benchmark.py`
- TPU source summary: `results/tpu_tuning/t29_prod_tpu_baseline_c1/20260301_123909_tpu-items-per-step-matrix/matrix_summary.json`
- GPU source summary: `results/20260301_prodshape_gpu_baseline_rtxpro6000/raw_results.json`
- Final report: `results/task_artifacts/20260301_t29_prod_final_report/final_report.md`
- Capacity model: `results/task_artifacts/20260301_t30_capacity_model/capacity_plan.md`
