# QwQ-32B vLLM Ascend Roadmap

## Current Progress

- Project context and roadmap: complete.
- ModelScope download: complete for `Qwen/QwQ-32B` BF16 and
  `vllm-ascend/QwQ-32B-W8A8`.
- Reproducible `.venv` and four-rank HCCL validation: complete.
- BF16 and W8A8 serving smoke tests: complete.
- TP4 and PP4 high-concurrency and 4K-64K long-context matrices: complete.
- Four throughput figures and machine-readable summary: complete under
  `results/20260730_qwq32b_4npu_nocache_v2/`.
- Deterministic random QwQ-32B W4A8 dynamic checkpoint: complete under
  `models/QwQ-32B-W4A8-Random/`.
- W4A8 TP4/PP4 high-concurrency and 4K-64K matrices: complete, 24/24 points
  successful.
- Combined 72-row BF16/W8A8/W4A8 report and four figures: complete under
  `results/20260730_qwq32b_w4a8_random_noprefix/`.
- 910B4 W4A8 per-group microbenchmark, exact vLLM profiling, and source-path
  analysis: complete under the same result directory and
  `reports/W4A8_PERGROUP_ANALYSIS.md`.
- ModelScope metadata is retained and its downloads passed built-in validation;
  an additional explicit local SHA-256 manifest remains optional.
- Publication-grade repeats, a calibrated real W4A8 checkpoint, and PP4 W8A8
  lifecycle debugging remain follow-up work.

## 1. Objective and Definition of Done

Establish a reproducible QwQ-32B inference baseline on the local 8x Ascend
910B4 server, then compare BF16 with official Ascend W8A8.

The first machine-validation milestone is complete:

- BF16 and W8A8 generate valid output and serve the OpenAI-compatible API.
- Both modes complete matching online matrices on TP4 and PP4.
- The report contains commands, versions, throughput, latency, device samples,
  diagnostic failures, and W8A8/BF16 ratios.
- Repository documentation and scripts capture the reproduction path.

The next quality gate is three or more repeats for selected primary points,
with dispersion, plus a calibrated W4A8 checkpoint before task-quality claims.

## 2. Planned Repository Layout

```text
.
|-- CONTEXT.md
|-- ROADMAP.md
|-- README.md
|-- requirements/
|   `-- constraints.txt
|-- scripts/
|   |-- check_env.sh
|   |-- download_models.sh
|   |-- serve.sh
|   `-- smoke_offline.py
|-- benchmarks/
|   |-- configs/
|   |-- prompts/
|   |-- run_offline.sh
|   |-- run_online.sh
|   `-- summarize.py
|-- reports/
|-- results/
`-- models/
```

`.venv/`, `models/`, raw runtime caches, and large benchmark outputs will be
excluded from Git. Benchmark configurations, compact metrics, reports, and
scripts will be tracked.

## 3. Phase 0: Reproducible Environment

**Status: complete.**

### Work

1. Initialize this directory as an integration repository.
2. Add `.gitignore`, a short `README.md`, environment checker, and pinned
   dependency constraints.
3. Create `.venv/` with Python 3.10.
4. Install the CANN 8.5-compatible stack:
   - `torch==2.8.0`
   - `torch-npu==2.8.0.post2`
   - `vllm==0.13.0`
   - `vllm-ascend==0.13.0`
   - `transformers>=4.57.4`
5. Keep `/usr/local/Ascend/cann-8.5.0` as the host CANN installation.
6. Verify imports, NPU allocation, four-process HCCL startup, and a small
   all-reduce on logical devices `0,1,4,5`.

Use released wheels first. Do not clone or patch upstream vLLM repositories
unless a reproduced incompatibility requires source-level debugging.

### Exit Criteria

- Package versions and import paths are captured in a timestamped environment
  report.
- `torch.npu.device_count()` matches the selected visibility mask.
- A tensor allocation and four-rank collective complete without errors.
- The global Python environment and `../nitro` remain unchanged.

If the selected versions cannot be installed or initialized, stop model work,
record the exact failure, and choose a complete supported version tuple rather
than mixing individual package versions.

## 4. Phase 1: Model Acquisition

**Status: complete for usable, ModelScope-validated artifacts; optional
full-file SHA-256 manifest remains.**

The two model repositories were downloaded on 2026-07-29. Artifact manifest
generation remains before this phase is fully closed.

### Work

Download into the workspace:

```text
models/QwQ-32B/          <- Qwen/QwQ-32B
models/QwQ-32B-W8A8/    <- vllm-ascend/QwQ-32B-W8A8
```

For each model:

- Pin and record the source revision.
- Verify configuration, tokenizer, generation configuration, and all weight
  shards.
- Record file sizes and SHA-256 hashes in a manifest.
- For W8A8, require `quant_model_description.json` and verify its declared
  quantization method before launching vLLM.

Do not run ModelSlim conversion in this phase.

### Exit Criteria

- Both model directories are complete and loadable by Transformers config and
  tokenizer APIs without network fallback.
- A versioned manifest identifies every downloaded artifact.
- Available disk space remains sufficient for runtime caches and benchmark
  results.

## 5. Phase 2: Functional and Quality Smoke Tests

**Status: complete for the initial serving and generation checks.**

### Common Configuration

Use:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,4,5
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

Initial runtime settings:

- TP size: 4
- Executor: multiprocessing
- Maximum model length: 4096
- Deterministic prompt set and fixed sampling seed
- Eager mode first; graph mode remains disabled until the baseline passes

### BF16

Run the BF16 model with an explicit BF16 dtype. Validate:

- Offline `LLM.generate`.
- OpenAI-compatible server startup.
- `/v1/models` and `/v1/completions`.
- Chinese, English, reasoning, short-input, and near-limit input prompts.

### W8A8

Repeat the same tests with the official W8A8 model and
`--quantization ascend`. Use the same prompts and sampling configuration.

### Validation

Check:

- Output is non-empty and terminates normally.
- No NaN/Inf, device fault, worker crash, or leaked serving process occurs.
- Token counts and finish reasons are recorded.
- Greedy outputs are compared for obvious corruption; W8A8 is not required to
  be byte-identical to BF16.
- Peak HBM is recorded for each rank.

### Exit Criteria

Both precisions pass offline and online smoke tests in separate clean processes.
If W8A8 fails, preserve the BF16 baseline and diagnose model-format/runtime
compatibility before changing kernels or quantizing a new model.

## 6. Phase 3: Fair Performance Benchmark

**Status: machine-validation matrix complete with one timed repeat per point.
The repeat/dispersion quality gate remains.**

### Benchmark Rules

- Compare one TP4 BF16 instance with one TP4 W8A8 instance on the same logical
  cards `0,1,4,5`.
- Use identical model length, prompts, tokenization, scheduler configuration,
  memory utilization, request ordering, and output length caps.
- Run only one compared instance at a time.
- Warm up before measurement.
- Repeat each point at least three times and report median plus dispersion.
- Preserve stdout/stderr, server logs, client JSON, device snapshots, and exact
  commands in a timestamped directory.
- Report failures and OOMs as results; do not remove them silently.

### Original Offline Matrix

Use fixed tokenized inputs so tokenizer variability is outside the timed
section:

| Scenario | Input tokens | Output tokens | Batch/concurrency |
|---|---:|---:|---|
| Decode-heavy | 128 | 1024 | 1, 4, 8, 16 |
| Balanced | 1024 | 1024 | 1, 4, 8, 16 |
| Prefill-heavy | 3072 | 128 | 1, 4, 8, 16 |

Capture:

- Requests/s
- Input tokens/s
- Output tokens/s
- Total tokens/s
- End-to-end elapsed time
- Per-device peak HBM

### Implemented Online Matrix

The completed run starts the OpenAI-compatible server and drives deterministic
random-token prompts with prefix caching disabled:

```text
high concurrency = 1, 4, 8, 16, 32, 64 (1024 input / 256 output)
long context = 4K, 8K, 16K, 32K, 40K, 64K (128 output, concurrency 4)
```

The 64K point uses the QwQ README's static YaRN factor 4.0 and an enlarged
131072-position table. Points through 40K in the completed run use the native
model configuration.

Run balanced and decode-heavy workloads. Capture:

- Successful and failed request counts
- Request throughput
- Input, output, and total tokens/s
- TTFT p50/p90/p99
- TPOT p50/p90/p99
- End-to-end latency p50/p90/p99

### Reporting

For every matching point, compute:

```text
speedup = W8A8 metric / BF16 metric
memory_reduction = BF16 peak HBM / W8A8 peak HBM
```

Invert the ratio for latency metrics so the meaning of improvement remains
explicit. Do not collapse prefill-heavy and decode-heavy results into one
headline number.

### Exit Criteria

- Every successful W8A8 point has a directly comparable BF16 point.
- At least three valid repetitions exist for the primary balanced workload.
- The report explains whether each workload is compute-, memory-, scheduler-,
  or PCIe-collective-sensitive when evidence supports the conclusion.

## 7. Phase 4: Capacity and Optimization

**Status: W4A8 operator-path profiling complete; capacity sweeps and
publication-grade repeats remain.**

1. Test a second TP4 replica on logical devices `2,3,6,7` and measure aggregate
   whole-server serving throughput.
2. Evaluate graph mode with the same correctness checks and benchmark matrix.
3. Sweep KV-cache memory utilization, maximum batched tokens, and concurrency.
4. Extend the completed quantized GEMM profiling to attention, host scheduling,
   ACL Graph, and HCCL time.
5. Consider ModelSlim self-quantization only if the official W8A8 artifact is
   unsuitable or a calibration study becomes a project goal.
6. Consider reusing or adapting NITRO AscendC work only after profiling proves
   that a specific supported vllm-ascend operator is the bottleneck.
7. Replace the random W4A8 artifact with a calibrated QwQ-32B checkpoint before
   evaluating accuracy or task performance.

Each optimization must be measured against the frozen Phase 3 baseline and
must retain the Phase 2 functional checks.

## 8. Result Contract

Each benchmark run will create:

```text
results/<timestamp>_<precision>_<workload>/
|-- command.txt
|-- environment.txt
|-- model_manifest.txt
|-- npu_before.txt
|-- npu_after.txt
|-- server.log
|-- client.json
`-- metrics.json
```

The comparison report under `reports/` will state:

- Hardware, software, model revisions, and device mapping
- Exact workload definitions
- BF16, W8A8, and W4A8 raw medians and dispersion
- Speedup and memory ratios
- Correctness status
- Failed configurations
- Known limitations, especially the lack of HCCS

Execution success and performance conclusions will be reported separately. A
working W8A8 deployment is not itself evidence of a throughput improvement.
