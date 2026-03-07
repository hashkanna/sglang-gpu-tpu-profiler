# sglang-profiler: Design Document

**Automated Profiling Comparison Framework — PyTorch/GPU vs JAX/TPU**

| | |
|---|---|
| **Author** | Kannappan |
| **Date** | February 24, 2026 |
| **Version** | 1.1 — Expanded with diagnostic microbenchmarks, load regimes, TPU performance killers, full request lifecycle |
| **Status** | Draft |
| **Repository** | `sglang-jax/profiling/` |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Configuration System](#3-configuration-system)
4. [Layer 1: Runner](#4-layer-1-runner)
5. [Diagnostic Microbenchmarks](#5-diagnostic-microbenchmarks)
6. [Layer 2: Analyzers](#6-layer-2-analyzers)
7. [Layer 3: Reporter](#7-layer-3-reporter)
8. [Copy vs Compute: Deep Dive](#8-copy-vs-compute-deep-dive)
9. [Fairness and Normalization](#9-fairness-and-normalization)
10. [Load Regimes and Serving Profiles](#10-load-regimes-and-serving-profiles)
11. [CLI Interface](#11-cli-interface)
12. [sglang-jax Specific Considerations](#12-sglang-jax-specific-considerations)
13. [Common TPU Performance Killers](#13-common-tpu-performance-killers)
14. [Automation Assessment](#14-automation-assessment)
15. [Future Extensions](#15-future-extensions)
16. [Implementation Plan](#16-implementation-plan)
- [Appendix A: Hardware Reference](#appendix-a-hardware-reference)
- [Appendix B: Glossary](#appendix-b-glossary)

---

## 1. Executive Summary

This document specifies the design of **sglang-profiler**, a fully automated, config-driven framework for performing repeatable, side-by-side performance comparisons between sglang running on PyTorch/NVIDIA GPU and sglang-jax running on Google Cloud TPU. The framework produces comprehensive interactive reports covering throughput, latency, memory, copy-vs-compute utilization, per-layer analysis, and automated anomaly detection.

### 1.1 Problem Statement

Comparing ML inference performance across fundamentally different hardware architectures (CUDA/GPU vs XLA/TPU) and frameworks (PyTorch eager vs JAX compiled) is currently a manual, error-prone, and non-reproducible process. Existing profiling tools are platform-specific: NVIDIA Nsight for GPUs, TensorBoard/XProf for TPUs. No unified framework exists that can run identical workloads on both platforms, collect comparable metrics, and produce a single consolidated analysis.

### 1.2 Goals

- **Repeatability:** Any profiling run should be reproducible from a single config file and CLI command.
- **Fairness:** The framework must ensure apples-to-apples comparisons through identical prompts, controlled warmup, and statistically significant sampling (multiple runs per configuration).
- **Comprehensiveness:** Cover throughput, latency (per-stage and per-layer), memory (peak and timeline), hardware utilization, copy-vs-compute breakdown, and cost efficiency.
- **Automation:** Minimize manual intervention. After initial setup, producing a full comparison report should require a single CLI invocation.
- **Extensibility:** Support adding new backends (e.g., TensorRT-LLM, vLLM), new metrics, and new report sections without architectural changes.

### 1.3 Non-Goals

- Fully automated trace interpretation (timeline visual analysis remains human-in-the-loop, though heuristic flagging is in scope).
- HLO/CUDA assembly level optimization (the framework identifies bottlenecks; the engineer fixes them).
- Real-time monitoring or alerting (this is a batch profiling tool, not an observability platform).
- Multi-node / multi-host profiling (v1 targets single-host configurations; multi-host is a future extension).

---

## 2. Architecture Overview

The system follows a three-layer architecture with clean separation of concerns, allowing each layer to evolve independently.

### 2.1 System Layers

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| Layer 1 | Runner | Executes workloads on both backends, collects raw metrics into a common schema. |
| Layer 2 | Analyzers | Computes derived metrics, runs anomaly heuristics, performs statistical aggregation. |
| Layer 3 | Reporter | Generates interactive HTML dashboard and optional DOCX/PDF summary report. |

Additionally, a cross-cutting **Configuration** layer (`sweep.yaml`) parameterizes the entire pipeline, and a **Schema** layer defines the canonical data format exchanged between layers.

### 2.2 Data Flow

```
sweep.yaml → Microbenchmarks → microbench_results.json
                                        ↓
sweep.yaml → Runner → raw_metrics.parquet → Analyzers → analyzed_results.json → Reporter → report.html
```

All intermediate artifacts are persisted to disk, enabling re-analysis without re-running workloads, and enabling debugging of any stage independently.

### 2.3 Directory Structure

```
sglang-jax/profiling/
├── config/
│   ├── sweep.yaml              # Primary configuration
│   ├── prompts/                # Standard prompt sets
│   └── thresholds.yaml         # Anomaly detection thresholds
├── runner/
│   ├── __init__.py
│   ├── base.py                 # Abstract backend interface
│   ├── pytorch_gpu.py          # PyTorch/CUDA runner
│   ├── jax_tpu.py              # JAX/TPU runner
│   └── metrics_schema.py       # Common metrics dataclass (8-stage lifecycle)
├── microbench/
│   ├── __init__.py
│   ├── pure_model.py           # Microbench 1: pure model step (bypass server)
│   └── server_overhead.py      # Microbench 2: stub model (isolate server)
├── load_regimes/
│   ├── __init__.py
│   ├── single_request.py       # Single-request latency profiling
│   ├── steady_state.py         # Sustained QPS throughput profiling
│   ├── overload.py             # Stress test / degradation profiling
│   └── load_client.py          # Configurable HTTP load generator
├── analyzers/
│   ├── throughput.py
│   ├── latency.py              # Full 8-stage request lifecycle analysis
│   ├── memory.py
│   ├── copy_compute.py         # Copy vs compute breakdown
│   ├── server_overhead.py      # Server vs device attribution
│   ├── anomaly_detector.py     # 17 heuristic rules
│   └── statistical.py          # Confidence intervals, outlier removal
├── reporter/
│   ├── html_dashboard.py       # Interactive React-based report
│   └── docx_summary.py         # Optional Word doc export
├── cli.py                      # Click-based CLI entry point
└── results/                    # Output directory (gitignored)
    ├── run_047/
    │   ├── microbench_results.json
    │   ├── raw_metrics.parquet
    │   ├── analyzed_results.json
    │   ├── report.html
    │   └── traces/             # Raw profiler exports
    └── ...
```

---

## 3. Configuration System

All profiling runs are driven by a single YAML configuration file. This is the primary interface for the user and the single source of truth for reproducibility.

### 3.1 sweep.yaml Schema

```yaml
# sglang-profiler configuration
experiment:
  name: "llama2-7b-comparison"
  model: "meta-llama/Llama-2-7b-chat-hf"
  description: "Full profiling sweep for sglang-jax PR #142"

backends:
  pytorch_gpu:
    enabled: true
    device: "cuda:0"              # GPU device
    gpu_type: "a100-80gb"         # For cost calculation
    sglang_config:
      tp_size: 1
      mem_fraction: 0.90
      attention_backend: "flashinfer"

  jax_tpu:
    enabled: true
    tpu_type: "v5e-8"             # For cost calculation
    sglang_jax_config:
      mesh_shape: [1, 1, 8]
      attention_backend: "ragged_paged"
      page_size: 256
      use_ragged_attention: true

sweep:
  batch_sizes: [1, 4, 8, 16, 32, 64, 128]
  sequence_lengths: [128, 256, 512, 1024, 2048]
  decode_steps: 128
  num_runs: 3                      # Runs per config for statistics
  warmup_runs: 2                   # Discarded warmup iterations

prompts:
  source: "config/prompts/standard_eval.jsonl"
  # Or: source: "synthetic"
  # synthetic_config:
  #   distribution: "uniform"     # uniform | zipf | realistic

metrics:
  collect:
    - throughput          # Tokens/sec (prefill + decode)
    - time_to_first_token # TTFT in ms
    - latency_per_stage   # Embedding, Attn, FFN, etc.
    - latency_per_layer   # All transformer layers
    - memory_timeline     # HBM usage over decode steps
    - memory_peak         # High-water mark
    - compute_utilization # MXU% / SM occupancy%
    - memory_bandwidth    # HBM bandwidth utilization
    - copy_compute_ratio  # % time in copy vs compute vs idle
    - cost_efficiency     # $/million tokens

  profiler:
    pytorch:
      enable_trace: true           # torch.profiler trace export
      enable_nsys: false           # NVIDIA Nsight Systems
      enable_cuda_events: true     # torch.cuda.Event timing
    jax:
      enable_trace: true           # jax.profiler.trace() export
      enable_cost_analysis: true   # AOT compiled.cost_analysis()
      enable_memory_analysis: true # compiled.memory_analysis()
      enable_hlo_dump: false       # XLA HLO text dump

anomaly_detection:
  thresholds_file: "config/thresholds.yaml"
  enabled: true

microbenchmarks:
  enabled: true
  run_before_sweep: true          # Run automatically before full sweep

  pure_model:
    batch_sizes: [1, 16, 64]      # Subset of sweep batch sizes
    sequence_lengths: [512]        # Single representative length
    num_runs: 10                   # More runs for tight confidence intervals
    warmup_runs: 5
    pre_tokenize: true             # Bypass tokenizer entirely

  server_overhead:
    enabled: true                  # Requires server binary
    stub_model: true               # Replace model with no-op
    concurrency: [1, 4, 16, 64]
    duration_seconds: 30

batching:
  pytorch_gpu:
    max_batch_tokens: 4096
    batch_window_ms: 10
    max_waiting_requests: 128
    scheduling_policy: "fcfs"
  jax_tpu:
    max_batch_tokens: 4096        # Must match GPU config for fairness
    batch_window_ms: 10
    max_waiting_requests: 128
    scheduling_policy: "fcfs"

load_regimes:
  enabled: false                   # Enable for serving benchmarks
  single_request:
    concurrency: 1
    num_requests: 50
    think_time_ms: 100
  steady_state:
    concurrency: 16
    target_qps: 50
    duration_seconds: 120
    warmup_seconds: 30
  overload:
    concurrency: 128
    target_qps: 0                  # Open-loop
    duration_seconds: 60
  client:
    region: "us-central1"          # Must match server region
    request_distribution: "poisson"

report:
  format: ["html", "docx"]
  include_raw_traces: false        # Bundle trace files in output
  auto_open: true                  # Open HTML report after run
```

> **Design Decision: Why YAML over CLI flags?** A config file ensures every run is fully reproducible. The CLI accepts `--config sweep.yaml` plus optional overrides like `--override sweep.batch_sizes=[16,32]` for quick iteration. The config file is committed to the repo alongside results for full provenance.

---

## 4. Layer 1: Runner

The Runner layer is responsible for executing identical workloads on both backends and collecting raw, unprocessed metrics into a common schema. It is the most hardware-dependent layer and uses a plugin architecture to support multiple backends.

### 4.1 Backend Interface

All backends implement an abstract interface that guarantees metric compatibility:

```python
class BackendRunner(ABC):
    @abstractmethod
    def setup(self, config: BackendConfig) -> None:
        """Initialize model, allocate KV cache, warm up."""

    @abstractmethod
    def run_generation(
        self, prompts: list[str], max_tokens: int
    ) -> RawMetrics:
        """Run generation and return raw metrics."""

    @abstractmethod
    def collect_hardware_metrics(self) -> HardwareMetrics:
        """Collect GPU/TPU utilization, memory bandwidth."""

    @abstractmethod
    def export_trace(self, path: Path) -> None:
        """Export profiler trace for manual inspection."""

    @abstractmethod
    def teardown(self) -> None:
        """Release resources."""
```

### 4.2 PyTorch/GPU Runner

The PyTorch runner wraps the existing sglang inference engine and collects metrics through the following mechanisms:

| Metric | Collection Method | Unit |
|--------|-------------------|------|
| Throughput | Wall-clock timing with `torch.cuda.synchronize()` barriers | Tokens/sec |
| Per-kernel timing | `torch.cuda.Event` with start/end around each operation | Microseconds |
| Trace export | `torch.profiler.profile()` with `on_trace_ready` callback | Chrome JSON |
| GPU utilization | pynvml / nvidia-smi polling at 100ms intervals | Percentage |
| Memory | `torch.cuda.max_memory_allocated()` + `memory_stats()` | Bytes |
| Memory bandwidth | DCGM or nsys metrics (`dram__bytes_read/write`) | GB/s |
| Copy detection | `torch.profiler` categorizes ops as CUDA/CPU/Communication | Time breakdown |

### 4.3 JAX/TPU Runner

The JAX runner wraps the sglang-jax inference engine. Key differences from the PyTorch runner include the need for explicit `jax.block_until_ready()` calls (since JAX uses async dispatch) and the availability of AOT compilation metadata.

| Metric | Collection Method | Unit |
|--------|-------------------|------|
| Throughput | Wall-clock with `jax.block_until_ready()` synchronization | Tokens/sec |
| Per-op timing | `jax.profiler.trace()` timeline decomposition | Microseconds |
| Trace export | `jax.profiler.trace(output_path)` → TensorBoard logs | TensorBoard |
| TPU utilization | Cloud Monitoring API (`tpu.googleapis.com/utilization`) | Percentage |
| Memory | `jax.local_devices()[0].memory_stats()` + `compiled.memory_analysis()` | Bytes |
| FLOPs estimate | `compiled.cost_analysis()` on AOT-compiled functions | FLOPs |
| Copy detection | HLO op categorization (compute vs copy vs collective) | Time breakdown |
| Recompilation | Count of jit tracing events via JAX callback hooks | Count + time |

### 4.4 Common Metrics Schema

Both runners emit data conforming to a shared schema, stored as Parquet for efficient columnar access:

```python
@dataclass
class RunMetrics:
    # Identity
    run_id: str
    backend: Literal["pytorch_gpu", "jax_tpu"]
    timestamp: datetime
    config_hash: str          # SHA256 of sweep.yaml

    # Configuration
    batch_size: int
    sequence_length: int
    decode_steps: int
    model_name: str

    # Throughput
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    time_to_first_token_ms: float
    total_generation_time_ms: float

    # Per-stage latency (ms) — Full request lifecycle
    # Server/runtime overhead stages:
    latency_queue_wait: float             # Time in request queue before processing
    latency_tokenization: float           # Tokenization / preprocessing on CPU
    latency_batch_formation: float        # Dynamic batching window + batch assembly
    latency_host_to_device: float         # H2D transfer of input tensors
    # Device compute stages:
    latency_embedding: float
    latency_attention_prefill: float
    latency_attention_decode: float
    latency_ffn: float
    latency_kv_cache_mgmt: float
    latency_sampling: float
    # Post-device stages:
    latency_device_to_host: float         # D2H transfer of output logits/tokens
    latency_postprocess_serialize: float  # Detokenization + response serialization
    # Aggregate categories:
    latency_server_overhead_total: float  # Sum of non-device stages
    latency_device_compute_total: float   # Sum of device stages
    latency_copy_transfer: float          # All H2D + D2H + D2D combined
    latency_scheduling: float

    # Per-layer latency (ms) - list of 32/40/80 floats
    per_layer_attention_ms: list[float]
    per_layer_ffn_ms: list[float]

    # Memory
    memory_peak_gb: float
    memory_model_weights_gb: float
    memory_kv_cache_gb: float
    memory_activations_gb: float
    memory_timeline_gb: list[float]  # Per decode step

    # Hardware utilization
    compute_utilization_pct: float   # MXU% or SM%
    memory_bandwidth_utilization_pct: float

    # Copy vs Compute (% of wall time per phase)
    copy_compute_breakdown: dict[str, CopyComputeRatio]

    # Cost
    hardware_cost_per_hour: float
    cost_per_million_tokens: float
```

---

## 5. Diagnostic Microbenchmarks

Before running the full profiling sweep, the framework provides two targeted microbenchmarks that isolate whether a performance gap is caused by device compute or server/runtime overhead. This is the single most effective tactic for diagnosing "TPU is slower" issues — it typically provides a clear answer in under an hour and prevents days of chasing the wrong bottleneck.

### 5.1 Rationale

A very common outcome when TPU "lags behind" GPU is that TPU compute is fine, but the system is losing on runtime overhead — Python dispatcher latency, tokenization, host↔device copies, shape polymorphism, or XLA recompiles. The full profiling sweep can confirm this, but the microbenchmarks answer the question immediately with minimal setup.

### 5.2 Microbenchmark 1: Pure Model Step

This microbenchmark bypasses the entire HTTP server, scheduler, tokenizer, and batch formation pipeline. It feeds pre-tokenized, pre-batched tensors directly to the model function and measures only device compute time.

```python
class PureModelBenchmark:
    """Isolate device compute from all server overhead."""

    def run(self, config: BenchmarkConfig) -> PureModelMetrics:
        # Pre-tokenize and pre-batch on host
        input_ids = self._prepare_fixed_inputs(config)

        # Transfer to device once (not measured)
        device_inputs = self._to_device(input_ids)

        # Warmup (compilation + cache warming)
        for _ in range(config.warmup_runs):
            self._run_forward(device_inputs)
            self._sync()  # block_until_ready / cuda.synchronize

        # Timed runs
        times = []
        for _ in range(config.num_runs):
            start = time.perf_counter()
            output = self._run_forward(device_inputs)
            self._sync()
            times.append(time.perf_counter() - start)

        return PureModelMetrics(
            prefill_latency_ms=...,
            decode_step_latency_ms=...,
            device_throughput_tokens_per_sec=...,
        )
```

**If TPU is still slower here** → the problem is in the model/XLA/sharding/kernel fusion. Focus on Pallas kernels, sharding strategy, and HLO optimization.

**If TPU is comparable or faster here** → the problem is in the server/runtime stack. Focus on the server overhead microbenchmark below.

### 5.3 Microbenchmark 2: Server Overhead

This microbenchmark stubs out the model call (replaces it with a cheap no-op or minimal device operation) and stress-tests the full HTTP + batching + tokenization + serialization pipeline.

```python
class ServerOverheadBenchmark:
    """Isolate server/runtime overhead from device compute."""

    def setup(self, config: BenchmarkConfig):
        # Replace model forward with a trivial device op
        self.server = self._create_server_with_stub_model(config)

    def run(self, config: BenchmarkConfig) -> ServerOverheadMetrics:
        # Use real HTTP client, real tokenizer, real scheduler
        client = self._create_load_client(
            concurrency=config.concurrency,
            target_qps=config.target_qps,
        )

        results = client.run_load_test(
            prompts=config.prompts,
            duration_seconds=config.duration,
        )

        return ServerOverheadMetrics(
            p50_request_latency_ms=...,
            p99_request_latency_ms=...,
            queue_wait_ms=...,
            tokenization_ms=...,
            batch_formation_ms=...,
            serialize_ms=...,
            max_sustained_qps=...,
        )
```

**If TPU server overhead is higher** → focus on the scheduler, Python overhead, tokenization, or host↔device copy patterns. Common fixes include moving tokenization off-thread, reducing per-request synchronization, and pre-compiling for fixed shapes.

**If server overhead is comparable** → the gap is purely in device compute, and the full profiling sweep (Sections 6–8) will identify exactly where.

### 5.4 Microbenchmark Configuration

```yaml
microbenchmarks:
  enabled: true
  run_before_sweep: true          # Run automatically before full sweep

  pure_model:
    batch_sizes: [1, 16, 64]      # Subset of sweep batch sizes
    sequence_lengths: [512]        # Single representative length
    num_runs: 10                   # More runs for tight confidence intervals
    warmup_runs: 5
    measure_prefill: true
    measure_decode: true
    pre_tokenize: true             # Bypass tokenizer entirely

  server_overhead:
    enabled: true                  # Requires server binary
    stub_model: true               # Replace model with no-op
    concurrency: [1, 4, 16, 64]   # Client concurrency levels
    target_qps: [10, 50, 100]     # Target queries per second
    duration_seconds: 30
    measure_stages:
      - queue_wait
      - tokenization
      - batch_formation
      - host_to_device
      - device_to_host
      - postprocess_serialize
```

### 5.5 Microbenchmark Report Section

The microbenchmark results are displayed as the first diagnostic in the report — before the full sweep — as a "Where is the gap?" summary:

| Metric | PyTorch/GPU | JAX/TPU | Δ | Verdict |
|--------|-------------|---------|---|---------|
| Pure model prefill (bs=16, seq=512) | 8.2 ms | 7.1 ms | -13% TPU faster | ✅ Device compute is fine |
| Pure model decode step (bs=16) | 2.4 ms | 2.1 ms | -12% TPU faster | ✅ Device compute is fine |
| Server p50 latency (stub model) | 1.8 ms | 4.2 ms | +133% TPU slower | ❌ Server overhead is the bottleneck |
| Server tokenization | 0.3 ms | 0.3 ms | 0% | ✅ Not the issue |
| Server batch formation | 0.4 ms | 1.9 ms | +375% TPU slower | ❌ Batch scheduler overhead |
| Server queue wait (c=16) | 0.2 ms | 1.1 ms | +450% TPU slower | ❌ Queueing/dispatch overhead |

This table immediately tells you whether to focus on device optimization or server optimization, saving potentially days of misdirected profiling effort.

---

## 6. Layer 2: Analyzers

The Analyzer layer takes raw metrics from the Runner and produces derived insights. Each analyzer is a standalone module that reads from the common schema and emits structured analysis results.

### 6.1 Statistical Aggregation

Since each configuration is run multiple times (default: 3 runs), the statistical analyzer computes the median, mean, standard deviation, p5/p95 percentiles, and coefficient of variation (CV) for every numeric metric. A CV threshold (default: 15%) flags unstable measurements that may need more runs. Outlier removal uses the IQR method (1.5× interquartile range) before computing summary statistics.

### 6.2 Throughput Analyzer

Computes derived throughput metrics including tokens per second per dollar (cost-normalized throughput), scaling efficiency (throughput at batch N divided by N times throughput at batch 1), and the crossover batch size where TPU overtakes GPU. The crossover point is found by linear interpolation between adjacent batch sizes.

### 6.3 Copy vs Compute Analyzer

This is the core analyzer for the side-by-side copy-compute comparison. It categorizes all profiled operations into three buckets:

- **Compute:** Operations executing on the MXU/tensor cores (matmul, einsum, convolution) or vector/scalar units (elementwise ops, reductions).
- **Copy/Transfer:** Data movement between host and device (H2D, D2H), between HBM and on-chip memory (VMEM loads/stores on TPU, L1/L2 on GPU), and inter-device communication (all-reduce, all-gather for tensor parallelism).
- **Idle/Sync:** Time when the accelerator is waiting — synchronization barriers, kernel launch overhead, XLA dispatch latency, Python GIL contention.

For PyTorch/GPU, categorization uses `torch.profiler`'s built-in op categories (CUDA kernel, memcpy, CPU). For JAX/TPU, categorization is derived from HLO op types: fusion and dot ops are compute; copy, send, recv, all-reduce are transfer; the remainder is computed by subtracting compute and transfer from total wall time.

### 6.4 Anomaly Detector

The anomaly detector applies configurable heuristic rules to flag performance issues automatically. Each rule produces a severity (high/medium/low/info), a description, and an actionable suggestion.

| Rule | Condition | Severity |
|------|-----------|----------|
| XLA Recompilation | Recompilation count > 0 during steady-state decode | High |
| Copy Blocking Compute | Copy time > 20% of wall time in any phase | Medium |
| Memory Bandwidth Bottleneck | HBM BW utilization > 80% while compute util < 50% | High |
| Kernel Gap | Gap between consecutive kernels > 100µs | Medium |
| Shape Variance | More than 3 distinct input shapes in decode loop | High |
| KV Cache Overhead | KV cache management > 15% of per-step latency | Medium |
| Underutilized Sampling | Idle time > 40% during sampling phase | Low |
| Memory Regression | Peak memory > 110% of previous run with same config | Medium |
| Uneven Layer Latency | Any layer latency > 2× median layer latency | Medium |
| Scaling Anomaly | Throughput decreases when batch size increases | High |
| TPU Underfed | Compute utilization < 30% at batch sizes ≥ 16 (not enough work per step) | High |
| Server Overhead Dominant | Server overhead > 50% of total request latency | High |
| Excessive Sync | `block_until_ready()` or `cuda.synchronize()` called > 2× per decode step | Medium |
| Redundant Compute | Full logits computed when only target token logprobs needed (scoring path) | Medium |
| Collectives Overhead | All-reduce / all-gather time > 25% of per-step latency (multi-chip) | High |
| Input Pipeline Bottleneck | CPU utilization > 90% while device utilization < 50% | High |
| Batch Formation Stall | Batch formation latency > 2× median across all backends | Medium |

> **Extensibility: Custom Rules.** Rules are defined in `config/thresholds.yaml` and loaded at runtime. Users can add domain-specific rules (e.g., a rule specific to ragged paged attention page utilization) without modifying the analyzer code. Each rule is a Python function with a standard signature: `(metrics: RunMetrics, config: ThresholdConfig) → Optional[Anomaly]`.

---

## 7. Layer 3: Reporter

The Reporter generates the final deliverable: an interactive HTML dashboard and an optional DOCX summary. The HTML report is a self-contained single file (all JavaScript and CSS inlined) that can be opened in any browser without a server.

### 7.1 Report Sections

| Tab | Contents |
|-----|----------|
| Diagnostics | Microbenchmark results: pure model vs server overhead comparison. "Where is the gap?" summary table. Displayed first as the top-level triage. |
| Overview | Headline metrics (6 cards), radar chart, top anomalies summary. Provides an at-a-glance comparison. |
| Throughput | Bar chart of decode throughput vs batch size. Analysis note identifying crossover point and scaling pattern. |
| Latency | Horizontal bar chart of per-stage latency including full request lifecycle (queue → tokenize → batch → H2D → compute → D2H → serialize). Identifies dominant bottleneck per backend. |
| Copy vs Compute | Side-by-side stacked bar charts showing % compute, % copy, % idle per phase. This is the centerpiece visualization. |
| Memory | Area chart of HBM usage over decode steps. Highlights peak, steady-state, and page allocation pattern. |
| Per-Layer | Line chart of attention and FFN latency across all transformer layers. Flags outlier layers. |
| Server Overhead | Breakdown of non-device latency: queue wait, tokenization, batch formation, serialization. Compared across backends. |
| Load Regimes | Performance comparison across single-request, steady-state, and overload conditions. |
| Anomalies | Full list of heuristic-detected issues, with severity, description, and actionable suggestion per item. |
| Sweep Matrix | Full table of all (seq_len × batch_size) configurations with throughput, memory, and Δ columns. |

### 7.2 Interactive Dashboard Implementation

The HTML report is a self-contained single-file React application. No server, no CDN dependencies, no build step required to view it — just open `report.html` in any browser.

#### 7.2.1 Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| UI Framework | React (embedded via standalone UMD build) | Component-based, wide ecosystem, familiar to contributors |
| Charts | Recharts | Declarative, composable, built on D3 primitives. Supports all required chart types. |
| Styling | Inline styles / CSS-in-JS (no external stylesheets) | Required for single-file self-containment |
| Data | JSON embedded as `<script>` tag in HTML | No fetch calls, works offline, opens instantly |
| Build | Python `html_dashboard.py` generates the complete file | No Node.js/npm required at report generation time |

#### 7.2.2 Self-Contained HTML Generation

The reporter generates the dashboard by:

1. **Serializing analyzed results** to a JSON blob and embedding it as `window.__PROFILING_DATA__ = {...}` in a `<script>` tag.
2. **Inlining React + Recharts** from vendored UMD bundles (React 18 + Recharts 2.x, ~300KB gzipped total). These are stored in `reporter/vendor/` and concatenated into the HTML.
3. **Inlining the dashboard component** as a JSX→JS transpiled bundle (pre-transpiled at framework install time, not at report generation time).
4. **Writing a single HTML file** that contains everything. The file typically weighs 500KB–1MB depending on sweep size.

```python
def generate_report(analyzed_results: AnalyzedResults, output_path: Path):
    """Generate self-contained HTML dashboard."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>sglang-profiler — {analyzed_results.experiment_name}</title>
    <style>{EMBEDDED_CSS}</style>
</head>
<body>
    <div id="root"></div>
    <script>window.__PROFILING_DATA__ = {json.dumps(analyzed_results.to_dict())}</script>
    <script>{REACT_UMD}</script>
    <script>{RECHARTS_UMD}</script>
    <script>{DASHBOARD_BUNDLE}</script>
</body>
</html>"""
    output_path.write_text(html)
```

#### 7.2.3 Dashboard Layout and Navigation

The dashboard uses a tabbed layout with a persistent header:

```
┌──────────────────────────────────────────────────────────────┐
│  sglang Profiling Report          [Auto-Generated] Run #047  │
│  PyTorch/GPU (A100) vs JAX/TPU (v5e-8) · Llama-2-7B        │
├──────────────────────────────────────────────────────────────┤
│  Diagnostics │ Overview │ Throughput │ Latency │ Copy vs ... │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [Active tab content: charts + analysis notes]               │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  Config: sweep.yaml · Duration: 47m 23s · 9 configs × 3 runs│
└──────────────────────────────────────────────────────────────┘
```

- **Header:** Experiment name, backends, model, run ID, generation timestamp. Always visible.
- **Tab bar:** All 11 tabs (Diagnostics, Overview, Throughput, Latency, Copy vs Compute, Memory, Per-Layer, Server Overhead, Load Regimes, Anomalies, Sweep Matrix). Horizontally scrollable on mobile.
- **Content area:** Charts + analysis notes for the active tab. Each chart is a Recharts `<ResponsiveContainer>` for automatic sizing.
- **Footer:** Config file reference, total sweep duration, number of configurations and runs.

#### 7.2.4 Chart Specifications

Each tab has specific visualization requirements:

**Diagnostics Tab:**
- Summary table with color-coded verdict column (✅ / ❌) comparing microbenchmark results
- Two bar charts: pure model latency comparison and server overhead breakdown
- Automated "where is the gap?" narrative generated from microbenchmark data

**Overview Tab:**
- 6 metric cards in a responsive grid: TTFT, Decode Throughput, Prefill Throughput, Peak Memory, Compute Utilization, Cost Efficiency
- Each card shows both backends' values, the winner highlighted in green, and the percentage delta
- Radar chart (`<RadarChart>`) with 6 axes: Throughput, Latency, Memory Efficiency, Compute Utilization, Cost Efficiency, Scalability — normalized to 0–100 scale
- Top 4 anomalies from the anomaly detector as a compact list

**Throughput Tab:**
- Grouped bar chart (`<BarChart>`) — X: batch size, Y: tokens/sec, two bars per group (GPU red, TPU blue)
- Analysis note template identifies crossover batch size and explains the scaling pattern

**Latency Tab:**
- Horizontal bar chart (`<BarChart layout="vertical">`) — Y: stage name (all 8 lifecycle stages), X: latency in ms
- Stages ordered by total impact. Server overhead stages visually grouped separately from device compute stages
- Color-coded by category: device compute (green), data transfer (yellow), server overhead (orange), idle (gray)

**Copy vs Compute Tab (centerpiece):**
- Side-by-side 100% stacked horizontal bar charts — one for GPU, one for TPU
- Each bar represents a phase (Prefill, Decode Step 1, Decode Steady, KV Update, Sampling)
- Three segments per bar: Compute (green), Copy/Transfer (yellow), Idle/Sync (gray)
- Shared legend between both charts
- Analysis note highlights the biggest copy-compute discrepancy between backends

**Memory Tab:**
- Area chart (`<AreaChart>`) — X: decode step, Y: HBM usage in GB
- Two overlapping areas (GPU and TPU) with low opacity fill
- Horizontal reference lines for HBM capacity of each device

**Per-Layer Tab:**
- Multi-line chart (`<LineChart>`) — X: layer index (L0–L31), Y: latency in ms
- Four lines: GPU Attention, GPU FFN, TPU Attention, TPU FFN
- Dashed lines for FFN, solid for attention
- Outlier layers (> 2× median) highlighted with markers

**Server Overhead Tab:**
- Grouped bar chart comparing non-device latency stages across backends
- Pie charts showing device compute vs server overhead percentage per backend
- Timeline showing batch formation efficiency (tokens per batch over time)

**Load Regimes Tab:**
- Three-panel layout: single-request, steady-state, overload
- Each panel shows latency distribution (histogram), throughput time series, and device utilization time series
- Cross-regime comparison table highlighting how metrics change under load

**Anomalies Tab:**
- Card-based layout, one card per anomaly
- Each card: severity dot (color-coded), severity badge, backend tag, finding description, actionable suggestion
- Sorted by severity (high → medium → low → info)

**Sweep Matrix Tab:**
- Full HTML table with monospace values
- Columns: Seq Len, Batch Size, GPU tok/s, TPU tok/s, Δ%, GPU Memory, TPU Memory
- Δ column color-coded: green for TPU advantage, red for GPU advantage
- Sortable by clicking column headers (JavaScript sort, no server needed)

#### 7.2.5 Design Tokens

The dashboard uses a consistent dark theme optimized for long profiling sessions:

```javascript
const COLORS = {
  pytorch: "#EE4C2C",       // PyTorch orange-red
  jax: "#4285F4",           // Google blue
  bg: "#0f1117",            // Background
  card: "#1a1d27",          // Card background
  border: "#2a2e3d",        // Borders
  text: "#e2e8f0",          // Primary text
  textMuted: "#8892a8",     // Secondary text
  green: "#10b981",         // Positive / winner
  red: "#ef4444",           // Negative / alert
  yellow: "#f59e0b",        // Warning / copy
  purple: "#8b5cf6",        // Low severity
  compute: "#22c55e",       // Compute time (green)
  copy: "#f59e0b",          // Copy/transfer time (yellow)
  idle: "#64748b",          // Idle/sync time (gray)
};
```

PyTorch is always red/orange, JAX/TPU is always blue — consistent across every chart and metric card. The green/yellow/gray palette for compute/copy/idle is used consistently across the copy-vs-compute, latency, and server overhead tabs.

#### 7.2.6 Responsiveness and Interactivity

- All charts use Recharts `<ResponsiveContainer width="100%" height={...}>` for automatic resizing
- Tooltips on hover show exact values for all data points
- Legends are clickable to toggle series visibility (built into Recharts)
- Sweep Matrix table supports column sorting via vanilla JavaScript click handlers
- Tab state is maintained in React component state (no URL routing needed for a single-file report)
- The report renders correctly at viewport widths from 768px (tablet) to 2560px (ultrawide)

### 7.3 Analysis Notes

Each chart includes an auto-generated analysis note below it. These notes are templated from the analyzed data, not hardcoded. For example, the throughput analysis note template is:

```
GPU leads at bs={crossover_bs_low} ({gpu_advantage_pct}%) due to lower
kernel launch overhead. Crossover at bs≈{crossover_bs}. TPU advantage
widens to +{tpu_max_advantage_pct}% at bs={max_bs} — {explanation}.
```

This ensures the analysis is always consistent with the data and updates automatically when metrics change.

---

## 8. Copy vs Compute: Deep Dive

This section provides detailed specification for the copy-vs-compute analysis, as this was the original motivation for the profiling framework and represents the most technically nuanced aspect of the design.

### 8.1 GPU Copy-Compute Instrumentation

On NVIDIA GPUs, the CUDA execution model provides natural separation between compute kernels and memory operations. The following instrumentation strategy captures this:

- **Compute stream:** All matmul, attention, FFN, and elementwise kernels execute on the default CUDA stream. Their duration is captured by `torch.cuda.Event` pairs or `torch.profiler`.
- **Copy stream:** Host-to-device transfers (input tokens, position IDs), device-to-host transfers (sampled token IDs, logits), and device-to-device transfers (KV cache page copies) are captured by separate events on the copy stream.
- **Overlap detection:** When copy and compute overlap (non-blocking transfers with pinned memory), the profiler timeline shows both streams active simultaneously. The analyzer detects this by checking for temporal overlap in the event timeline and computes effective copy overhead as: `max(0, copy_time − overlap_with_compute)`.

### 8.2 TPU Copy-Compute Instrumentation

TPU architecture is fundamentally different. The XLA compiler fuses operations aggressively, and the hardware uses a DMA engine for HBM↔VMEM transfers that can overlap with MXU compute. Instrumentation requires a different approach:

- **HLO analysis:** After AOT compilation, the HLO graph contains explicit copy, fusion, and collective-permute operations. We categorize each HLO op and use `compiled.cost_analysis()` to estimate time per op.
- **Trace-based:** `jax.profiler.trace()` exports a timeline showing MXU busy time, DMA busy time, and scalar/vector unit busy time as separate lanes. These correspond directly to compute, copy, and scalar overhead.
- **XLA fusion complication:** XLA may fuse a copy into a compute kernel (e.g., a fused reduce-scatter-matmul). In these cases, the copy cost is embedded in the fusion and cannot be separated from the trace alone. The analyzer flags such fusions and reports them as "fused copy-compute" rather than attempting to split them artificially.

### 8.3 Phase Definitions

The copy-compute breakdown is reported per phase, where phases are defined as follows:

| Phase | Definition |
|-------|------------|
| Prefill | Processing of the entire input prompt in a single forward pass. Includes embedding lookup, all attention layers (no KV cache read), all FFN layers, and final logits computation. |
| Decode Step 1 | First autoregressive decode step. Distinct from steady-state because KV cache is being populated for the first time and memory allocation patterns differ. |
| Decode Steady | Subsequent decode steps (averaged). KV cache reads from previously populated pages. Most representative of serving throughput. |
| KV Update | Isolated measurement of KV cache page management: writing new KV entries, page table updates, and any page copy operations (eviction, defragmentation). |
| Sampling | Token sampling from logits: softmax, temperature scaling, top-k/top-p filtering, multinomial sampling, and token ID transfer back to host. |

---

## 9. Fairness and Normalization

Comparing GPU and TPU performance is inherently challenging due to architectural differences. This section documents the normalization decisions made to ensure fair comparison. **If any of these variables differ between backends, you can easily see 2–10× swings that have nothing to do with TPU vs GPU.**

### 9.1 Controlled Variables

| Variable | Approach |
|----------|----------|
| Prompts | Identical prompt text, tokenized by each backend's tokenizer. Token count may differ slightly; we normalize by output token count. |
| Workload type | Explicitly specified: scoring vs generation, same seq_len, same batch size, same number of items per request (for multi-item scoring), same output format (top-k vs full logprobs). |
| Model weights | Same checkpoint (Llama-2-7B), same dtype (bfloat16). Weight initialization is deterministic. |
| Math precision | FP16/BF16 matched. Same attention implementation (or the delta is known and documented). Same KV-cache behavior and same padding strategy. |
| Batch composition | Identical batch construction: same prompt order, same padding strategy (pad-to-max within batch). |
| Batching policy | Matched dynamic batching configuration: max batch tokens, batching window duration, queueing policy, and timeout settings. Documented in config. |
| Warmup | Configurable warmup runs (default: 2) discarded before measurement. Ensures JIT compilation and cache warming on both backends. JAX/TPU must be post-compile and warmed — compilation dominates if accidentally included. |
| Sampling | Greedy decoding (temperature=0) for deterministic output. Stochastic sampling adds variance. |
| Sequence length | Output generation capped at identical max_tokens. Input sequences padded or truncated to target length. |
| Network path | Same client machine, same cloud region, same load generator settings, same concurrency level. For serving benchmarks, the client must be co-located to eliminate network variance. |
| Concurrency | Matched client concurrency and QPS targets. Both backends see the same request arrival pattern. |

### 9.2 Batching Policy Configuration

Batching policy is one of the most commonly overlooked sources of unfair comparison. The framework explicitly controls and documents:

```yaml
batching:
  pytorch_gpu:
    max_batch_tokens: 4096        # Maximum tokens in a batch
    batch_window_ms: 10           # Dynamic batching window
    max_waiting_requests: 128     # Queue depth before backpressure
    scheduling_policy: "fcfs"     # First-come-first-served
  jax_tpu:
    max_batch_tokens: 4096        # Must match GPU config
    batch_window_ms: 10           # Must match GPU config
    max_waiting_requests: 128     # Must match GPU config
    scheduling_policy: "fcfs"     # Must match GPU config
```

If either backend has a more aggressive batching policy (longer window, more tokens per batch), it will appear faster at the cost of higher per-request latency — and the comparison becomes meaningless.

### 9.3 Load Generation Configuration

For serving benchmarks, the load generator is as important as the server:

```yaml
load_test:
  enabled: false                   # Enable for serving benchmarks
  client:
    tool: "vegeta"                 # or "locust", "ghz", custom
    region: "us-central1"          # Must match server region
    concurrency: [1, 4, 16, 64]   # Concurrent connections
    target_qps: [10, 50, 100, 0]  # 0 = open-loop (max throughput)
    duration_seconds: 60
    warmup_seconds: 10
    request_distribution: "poisson" # poisson | uniform | bursty
  measurement:
    measure_client_side_latency: true
    measure_server_side_latency: true  # Requires server instrumentation
    include_queue_wait: true
```

### 9.4 Normalization Decisions

Several comparison dimensions require explicit normalization choices. These are encoded in the config and documented in the report header:

- **Throughput normalization:** Reported as output tokens per second per device. For multi-device configurations, total throughput is also reported but per-device is the primary metric.
- **Cost normalization:** On-demand pricing from cloud providers. A100 80GB at $3.67/hr (GCP), TPU v5e-8 at $2.44/hr (GCP). Updated in config; not hardcoded. Reports cost per million output tokens.
- **Compute utilization:** Reported as percentage of theoretical peak. A100 peak is 312 TFLOPS (BF16 tensor core). TPU v5e peak is 197 TFLOPS (BF16 MXU) per chip. These peaks are documented in the report for transparency.
- **Memory bandwidth:** A100 HBM2e at 2.0 TB/s. TPU v5e HBM at 1.6 TB/s per chip. Utilization is percentage of these theoretical peaks.

---

## 10. Load Regimes and Serving Profiles

Good profiling captures not just one operating point but multiple load regimes that reveal different bottlenecks. A system that performs well at low concurrency may fall apart under load, and vice versa. The framework captures traces and metrics across three distinct regimes.

### 10.1 Single Request (Latency-Optimized)

A single request in isolation, with no batching or queueing effects. This measures the floor latency — the best possible response time the system can achieve. It is primarily compute-bound and reveals kernel launch overhead, compilation latency, and base copy costs.

```yaml
load_regimes:
  single_request:
    concurrency: 1
    num_requests: 50              # Enough for stable statistics
    think_time_ms: 100            # Gap between requests (no pipelining)
```

**What it reveals:** The minimum achievable latency. If TPU is slower here despite similar device throughput, the issue is dispatch overhead (XLA compilation, Python, host↔device sync). This is the regime where GPU's lower launch overhead typically wins.

### 10.2 Steady-State (Throughput-Optimized)

Sustained load at the target serving QPS, with continuous batching active and the system in steady state. This is the most representative of production serving and reveals the system's throughput ceiling.

```yaml
  steady_state:
    concurrency: 16               # Typical production concurrency
    target_qps: 50                # Target queries per second
    duration_seconds: 120         # Long enough for steady state
    warmup_seconds: 30            # Discard initial ramp
```

**What it reveals:** Sustained throughput, batch efficiency, device utilization under continuous load, and whether the system can keep the accelerator fed. This is where TPU typically wins — if it's not winning here, the batching or feeding pipeline is the problem.

### 10.3 Overload (Stress Test)

Load beyond the system's sustainable capacity. Reveals degradation behavior, queue buildup, and failure modes.

```yaml
  overload:
    concurrency: 128              # Intentionally exceed capacity
    target_qps: 0                 # Open-loop (as fast as possible)
    duration_seconds: 60
    measure_tail_latency: true    # p99, p99.9
    measure_error_rate: true
    measure_queue_depth: true
```

**What it reveals:** How gracefully each backend degrades. Does latency grow linearly or exponentially? Does the system OOM? Does batching break down? Do recompilations start happening under shape diversity? Overload profiling often reveals issues invisible in steady-state testing.

### 10.4 Trace Capture Per Regime

For each load regime, the framework captures:

| Artifact | Purpose |
|----------|---------|
| Full profiler trace (10-second window) | Timeline analysis of device busy/idle/copy patterns |
| Per-request latency histogram | Distribution shape reveals multimodal behavior (e.g., recompilation spikes) |
| Device utilization time series | Is the device consistently busy or bursty? |
| Queue depth time series | Is the system keeping up with arrivals? |
| Compile event log | Any recompilations triggered under load? |

Comparing traces across regimes answers a critical question: does the bottleneck shift? At single-request it might be dispatch overhead; at steady-state it might be memory bandwidth; at overload it might be scheduling. Understanding this shift is essential for prioritizing optimizations.

---

## 11. CLI Interface

The CLI is the primary user interface. It uses Click for argument parsing and supports the following commands.

### 11.1 Commands

| Command | Description |
|---------|-------------|
| `sglang-profiler run --config sweep.yaml` | Run the full pipeline: microbenchmarks → sweep → analyze → report. |
| `sglang-profiler run --config sweep.yaml --backend jax_tpu` | Run only one backend (useful during development). |
| `sglang-profiler microbench --config sweep.yaml` | Run only the diagnostic microbenchmarks (pure model + server overhead). Fast triage. |
| `sglang-profiler analyze results/run_047/` | Re-run analysis on existing raw metrics without re-executing workloads. |
| `sglang-profiler report results/run_047/` | Re-generate report from existing analysis results. |
| `sglang-profiler compare run_047 run_048` | Compare two runs side-by-side (e.g., before/after optimization). |
| `sglang-profiler serve-bench --config sweep.yaml` | Run load regime benchmarks (single-request, steady-state, overload). |
| `sglang-profiler validate --config sweep.yaml` | Validate config without running (check hardware availability, model access, etc.). |

### 11.2 Overrides

Any config value can be overridden from the CLI for quick iteration without editing the YAML:

```bash
# Run a quick test with single batch size
sglang-profiler run --config sweep.yaml \
  --override sweep.batch_sizes=[16] \
  --override sweep.num_runs=1

# Disable anomaly detection
sglang-profiler run --config sweep.yaml \
  --override anomaly_detection.enabled=false
```

---

## 12. sglang-jax Specific Considerations

Given that this framework is being built primarily to profile and optimize the sglang-jax port, several features are tailored to the specific challenges of the sglang-jax codebase.

### 12.1 Ragged Paged Attention Profiling

The ragged paged attention kernel is the single most performance-critical component in sglang-jax. The framework provides dedicated instrumentation for this path, including page utilization (percentage of allocated pages that contain active tokens vs padding), ragged vs padded fallback ratio (how often the ragged path is used vs the padded fallback for long sequences), Pallas kernel tiling efficiency (whether the chosen block sizes align well with the TPU v5e VMEM capacity), and per-attention-head latency breakdown to detect load imbalance across heads.

### 12.2 Score-from-Cache Fastpath

The score-from-cache fastpath (PR #142 in sglang-jax) avoids redundant forward passes for multi-item scoring. The profiler instruments this path separately to measure cache hit rate (what percentage of scoring requests use the fastpath), latency savings (fastpath latency vs full forward pass latency), and memory overhead of maintaining the score cache.

### 12.3 XLA Recompilation Tracking

XLA recompilation is the most common source of unexpected latency spikes in JAX workloads. The framework tracks every recompilation event by hooking into JAX's tracing callback, logging the trigger (which function, what input shape change caused it), the compilation time, and whether it occurred during warmup (expected) or steady-state (anomalous). The anomaly detector flags any steady-state recompilation as a high-severity issue.

### 12.4 Kernel Parameter Sensitivity

For Pallas kernels, performance is highly sensitive to block sizes and tiling parameters. The framework supports a special `kernel_sweep` mode that varies kernel parameters (`block_q`, `block_kv`, `block_d`) independently while holding the workload constant, producing a heatmap of throughput vs kernel configuration. This directly supports the kernel tuning work in the sglang-jax repository.

### 12.5 Label-Only Scoring Fastpath Detection

A common source of wasted compute in scoring APIs: the TPU path computes full logits across the entire vocabulary when only target token logprobs are needed. This is especially relevant for multi-item scoring workloads where each request scores a specific set of candidate tokens.

The profiler detects this by comparing:
- **FLOPs in the final projection layer:** If the model computes a full `[batch, seq_len, vocab_size]` logits tensor when the scoring API only needs `[batch, seq_len, num_candidates]`, the excess FLOPs are flagged as redundant compute.
- **Output tensor size:** The D2H transfer size reveals whether full logits or only target logprobs are being transferred back to host.
- **GPU baseline:** If the PyTorch path uses a label-only projection (or fused sampling) while the JAX path computes full logits, this asymmetry is flagged as a high-priority optimization target.

A "label-only" scoring fastpath that avoids the full vocabulary projection often yields 2–5× speedups on the scoring path alone, and the profiler highlights this opportunity explicitly when detected.

---

## 13. Common TPU Performance Killers

This section catalogs the most frequently encountered reasons why TPU scoring/serving APIs underperform GPU baselines. The profiling framework is specifically designed to detect and diagnose each of these patterns. Understanding these patterns guides both the profiling strategy and the optimization priority.

### 13.1 Not Enough Work Per Step (TPU Underfed)

TPUs use systolic arrays (MXU) that achieve peak efficiency on large matrix multiplications. If the scoring path processes tiny batches or short sequences, the MXU is underutilized — the 128×128 systolic array needs large tiles to stay busy.

**Symptoms:**
- Compute utilization < 30% at batch sizes that should be sufficient
- Throughput scales sublinearly with batch size
- Pure model microbenchmark shows TPU slower than GPU at small batch but faster at large batch

**Fixes:** Increase tokens per step via better batching, score multiple items together (batch multi-item scoring into single requests), microbatch accumulation to fill the MXU.

**Framework detection:** The "TPU Underfed" anomaly rule triggers when compute utilization is below 30% at batch sizes ≥ 16. The throughput-vs-batch-size chart in the report visually shows the scaling curve.

### 13.2 Shape Churn → Recompilations

Variable input shapes (seq_len, batch size, num_items per request) cause XLA to recompile the computation graph for each new shape. Each recompilation can take 5–60 seconds and completely stalls serving.

**Symptoms:**
- Periodic latency spikes of 5–60s in the latency histogram
- Recompilation count > 0 in steady-state (the anomaly detector flags this as high severity)
- Multimodal latency distribution

**Fixes:** Make shapes static via padding/bucketing. Use a small set of shape buckets (e.g., seq_len buckets at powers of 2). Ensure the hot path is inside a single `jit`/`pjit` with static input signatures. Avoid Python-level control flow that leaks into jit boundaries.

**Framework detection:** Recompilation tracking (Section 12.3) catches every instance. The shape variance rule flags > 3 distinct shapes.

### 13.3 Host↔Device Synchronization Overhead

Calling `jax.block_until_ready()` too frequently, or pulling logits/logprobs back to host each decode step unnecessarily, creates synchronization bottlenecks that stall the TPU pipeline.

**Symptoms:**
- High idle time in the copy-compute breakdown
- Device utilization shows bursty pattern (busy → idle → busy)
- Server overhead microbenchmark shows high D2H latency

**Fixes:** Keep results on device as long as possible. Batch postprocessing. Avoid per-request sync — use async dispatch and only sync at batch boundaries. For scoring, accumulate results on device and transfer once per batch.

**Framework detection:** The "Excessive Sync" anomaly rule counts sync calls per decode step. The copy-compute analyzer shows the idle percentage per phase.

### 13.4 Input Pipeline Bottleneck (CPU-Bound Feeding)

If tokenization, request parsing, or batch formation happens on the critical path, the CPU becomes the bottleneck feeding the TPU. The TPU sits idle waiting for the next batch.

**Symptoms:**
- High CPU utilization (> 90%) while device utilization is low
- Large gap between device compute end and next compute start in traces
- Server overhead microbenchmark shows tokenization or batch formation dominating

**Fixes:** Move tokenization to a separate thread/process. Pre-tokenize in load tests. Use a faster tokenizer backend (e.g., `tiktoken` or HuggingFace `fast` tokenizers). Pipeline batch preparation with device execution.

**Framework detection:** The "Input Pipeline Bottleneck" anomaly rule checks CPU vs device utilization ratio. The server overhead breakdown in the report shows exactly which CPU-side stage is slow.

### 13.5 Sharding / Collectives Overhead (Multi-Chip)

Bad partitioning choices across multiple TPU chips can drown the system in all-reduce and all-gather collectives, especially if the tensor parallelism strategy doesn't match the ICI topology.

**Symptoms:**
- Collective operations (all-reduce, all-gather) account for > 25% of per-step time
- Scaling from 1 chip to N chips yields much less than N× throughput
- HLO dump shows unexpected collective-permute operations

**Fixes:** Verify sharding spec matches physical topology. Prefer data parallelism for scoring if the model fits on a single chip. If tensor parallelism is needed, ensure partition axes align with ICI dimensions. Use `jax.debug.visualize_sharding()` to verify.

**Framework detection:** The "Collectives Overhead" anomaly rule monitors collective time fraction. The per-op trace categorization separates collective time from compute time.

### 13.6 Redundant Compute in Scoring Path

Computing full vocabulary logits (`[batch, vocab_size]`) when only a small set of target token logprobs are needed wastes both FLOPs and memory bandwidth.

**Symptoms:**
- Final projection layer accounts for a disproportionate fraction of compute time
- D2H transfer size is `batch × vocab_size × dtype_size` instead of `batch × num_targets × dtype_size`
- GPU path is faster because it uses a fused/selective projection

**Fixes:** Implement a label-only scoring fastpath that gathers only the required token logits. Fuse the gather with the final projection. See Section 12.5 for detailed detection strategy.

**Framework detection:** The "Redundant Compute" anomaly rule compares output tensor sizes against expected target-only sizes.

---

## 14. Automation Assessment

A central question motivating this design is: how much of the profiling workflow can be automated? The following table provides an honest assessment of each aspect.

| Aspect | Automation Level | Notes |
|--------|-----------------|-------|
| Running identical workloads | ✅ Fully automated | Config-driven, reproducible from CLI. |
| Diagnostic microbenchmarks | ✅ Fully automated | Pure model + server overhead split runs before sweep. |
| Latency/throughput collection | ✅ Fully automated | Built into both runners. |
| Full request lifecycle (8-stage) | ✅ Fully automated | Queue → tokenize → batch → H2D → compute → D2H → serialize instrumented. |
| Memory usage tracking | ✅ Fully automated | Per-step and peak, both backends. |
| Hardware utilization metrics | ✅ Fully automated | Polled via platform APIs. |
| Report generation | ✅ Fully automated | HTML + DOCX from analyzed data. |
| Recompilation detection | ✅ Fully automated | JAX callback hooks. |
| Copy-compute breakdown | ✅ Fully automated | Profiler categorization + HLO analysis. |
| Load regime profiling | ✅ Fully automated | Single-request, steady-state, overload captured per config. |
| TPU underfed detection | ✅ Fully automated | Compute utilization vs batch size heuristic. |
| Server vs device attribution | ✅ Fully automated | Microbenchmark comparison determines where gap is. |
| Anomaly detection (common) | ⚙️ Heuristic rules | 17 configurable rules catch ~85% of issues. |
| Redundant compute detection | ⚙️ Heuristic rules | Output tensor size comparison flags label-only opportunities. |
| Copy-compute overlap analysis | ⚠️ Semi-automated | Heuristics + occasional manual trace inspection. |
| Deep trace timeline analysis | ⚠️ Manual with flags | Framework exports traces; human reviews flagged sections. |
| HLO optimization analysis | ⚠️ Mostly manual | Grep for known antipatterns; deep analysis requires expertise. |
| Sharding topology validation | ⚠️ Semi-automated | Config validation + collective overhead detection; ICI analysis manual. |
| Fair comparison decisions | 🔧 One-time setup | Encoded in config. Revisited when hardware/pricing changes. |

> **Bottom Line:** First run requires significant setup and calibration (~2–3 days). Every subsequent run is a single CLI command producing a full comparison report in ~45 minutes (dominated by actual workload execution, not overhead). Manual intervention is only needed when actively debugging a specific bottleneck that the automated heuristics have flagged.

---

## 15. Future Extensions

### 15.1 Version 2 Scope

- **Multi-host profiling:** Extend to multi-host TPU pod slices and multi-GPU (NVLink) configurations. Requires distributed trace collection and ICI/NVLink bandwidth analysis.
- **Additional backends:** TensorRT-LLM, vLLM, and ONNX Runtime as additional comparison points. The plugin architecture supports this with new `BackendRunner` implementations.
- **Regression CI:** Integrate into CI/CD pipeline to run a reduced sweep on every PR and flag performance regressions before merge.
- **Historical trending:** Maintain a database of run results and add a trend view showing throughput and memory over time (across code commits).

### 15.2 Version 3 Scope

- **AI-assisted trace analysis:** Feed profiler trace summaries (not raw timelines) into an LLM to generate natural-language analysis. This partially addresses the "trace interpretation" gap.
- **Automated optimization suggestions:** Beyond anomaly detection, suggest specific code changes. For example: "Your block_kv=128 causes VMEM spills; try block_kv=64" based on the kernel parameter sensitivity data.
- **Live profiling mode:** Stream metrics in real-time during long generation workloads, showing a live updating dashboard.

---

## 16. Implementation Plan

| Phase | Deliverable | Duration | Dependencies |
|-------|-------------|----------|--------------|
| Phase 1 | Config system + CLI skeleton + metrics schema (including full 8-stage lifecycle) | 2–3 days | None |
| Phase 2 | Diagnostic microbenchmarks (pure model + server overhead) for both backends | 2–3 days | Phase 1 |
| Phase 3 | PyTorch/GPU runner with full metric collection | 3–4 days | Phase 1 |
| Phase 4 | JAX/TPU runner with full metric collection | 3–4 days | Phase 1 |
| Phase 5 | All analyzers including anomaly detection (17 rules) | 3–4 days | Phase 3 or 4 |
| Phase 6 | Load regime framework (single-request, steady-state, overload) | 2–3 days | Phase 3 and 4 |
| Phase 7 | HTML dashboard reporter (including diagnostics, server overhead, load regime tabs) | 3–4 days | Phase 5 |
| Phase 8 | Integration testing + calibration + fairness validation | 2–3 days | Phase 7 |
| Phase 9 | Documentation + open-source packaging | 1–2 days | Phase 8 |

**Total estimated effort:** 18–28 days of focused development. Phases 3 and 4 can be parallelized. Phase 2 (microbenchmarks) is designed to be useful immediately — even before the full sweep is built, the microbenchmarks answer the "where is the gap?" question. The framework is designed to be useful incrementally — even with only one backend runner completed, single-platform profiling reports can be generated.

---

## Appendix A: Hardware Reference

Reference specifications for the primary hardware targets.

| Specification | NVIDIA A100 80GB | Google TPU v5e (per chip) |
|---------------|------------------|--------------------------|
| Architecture | Ampere (GA100) | Custom ASIC |
| Compute Units | 108 SMs, 432 Tensor Cores | 1 MXU (128×128 systolic array) |
| Peak BF16 TFLOPS | 312 | 197 |
| HBM Capacity | 80 GB HBM2e | 16 GB HBM |
| HBM Bandwidth | 2.0 TB/s | 1.6 TB/s |
| On-chip Memory | 40 MB L2 Cache | ~32 MB VMEM + CMEM |
| Interconnect | NVLink 600 GB/s | ICI 1.6 Tbps bidirectional |
| GCP On-demand $/hr | $3.67 (a2-highgpu-1g) | $2.44 (v5e-8, 8 chips) |
| Power (TDP) | 400W | ~200W (estimated per chip) |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| TTFT | Time to First Token. Latency from request arrival to first output token generated. |
| MXU | Matrix Multiply Unit. TPU's systolic array for matrix operations. |
| HBM | High Bandwidth Memory. Off-chip DRAM used by both GPUs and TPUs. |
| VMEM | Vector Memory. On-chip SRAM on TPU, analogous to shared memory/L1 on GPU. |
| HLO | High Level Operations. XLA's intermediate representation of computation graphs. |
| Pallas | JAX's custom kernel language for writing low-level TPU/GPU kernels. |
| ICI | Inter-Chip Interconnect. TPU's high-speed chip-to-chip communication fabric. |
| KV Cache | Key-Value Cache. Stores attention key/value tensors from previous tokens to avoid recomputation. |
| Ragged Attention | Attention implementation that handles variable-length sequences without padding. |
| AOT | Ahead-of-Time compilation. JAX compiles functions before execution, enabling cost/memory analysis. |
| CV | Coefficient of Variation. Standard deviation divided by mean; measures measurement stability. |
| H2D / D2H | Host-to-Device / Device-to-Host. Data transfer between CPU memory and accelerator memory. |
| Shape Bucketing | Padding input tensors to a fixed set of predetermined sizes to avoid XLA recompilation. |
| QPS | Queries Per Second. Rate of incoming requests to a serving system. |
| Open-Loop Load | Load generation that sends requests at a fixed rate regardless of response time, revealing true queuing behavior. |
| Closed-Loop Load | Load generation where the next request waits for the previous response, hiding queuing effects. |
| Label-Only Scoring | Computing logprobs only for target tokens instead of the full vocabulary, avoiding redundant FLOPs. |
| Systolic Array | Hardware unit (MXU) that processes matrix multiplications by flowing data through a grid of processing elements. Efficient for large matrices, underutilized for small ones. |
| Duty Cycle | Fraction of time a hardware unit is actively computing vs idle. Key metric for diagnosing TPU underfeeding. |
