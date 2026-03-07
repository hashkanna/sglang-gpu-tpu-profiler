# TPU Regression Gate (Nightly/CI)

Use the nightly wrapper to run a matrix sweep and fail the job on throughput, tail, failure-rate, or timed-compile regressions.
Default gate contract is now long-window/high-concurrency production scoring:
- `concurrency=16`
- `timed_requests=80`
- `repeats=3`
- co-located TPU VM client mode

## Command

```bash
scripts/run_tpu_regression_nightly.sh \
  --baseline-config config/prod_scenario_scoring_baseline_t67_label_only_fused_candidate.yaml \
  --tpu-name mi-bench-kannacl \
  --tpu-zone us-east5-a \
  --ssh-mode direct \
  --tpu-repo-path /home/kanna/work/sglang-jax-pr27-prof \
  --tpu-url http://127.0.0.1:30001 \
  --tpu-connection-mode tunnel \
  --repeats 3 \
  --timed-requests 80 \
  --concurrency 16
```

## Gate Config

Thresholds are defined in:

- `config/tpu_regression_gates.yaml`

The checker exits non-zero when any regression threshold is violated.
