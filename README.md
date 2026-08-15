# QwQ-32B on vLLM Ascend

This repository contains a reproducible QwQ-32B inference benchmark for one
Ascend 910B4 server. It compares:

- BF16, the official Ascend W8A8 checkpoint, and W4A8 dynamic
- four-card tensor parallelism (TP4)
- four-card pipeline parallelism (PP4)
- high-concurrency serving
- long-context serving up to 64K total tokens

The W4A8 checkpoint is deterministic random data with the real QwQ-32B tensor
shapes and vLLM Ascend metadata. It is valid for throughput, HBM, and operator
path measurements only, not model quality.

Read `CONTEXT.md` before changing the environment and `ROADMAP.md` for the
implementation phases.

## Environment

The runtime is installed in `.venv/` and uses the host CANN and ATB installs.
Every runtime script sources `scripts/common.sh`, which also disables proxy
variables and Hugging Face network fallback.

```bash
scripts/check_env.sh
source scripts/common.sh
.venv/bin/torchrun --standalone --nproc-per-node 4 scripts/check_hccl.py
```

The default four-card mask is `0,1,4,5`, selected to span both PCIe switch
groups on this host. `HCCL_OP_EXPANSION_MODE=AIV` is enabled because the
installed vLLM Ascend version recommends it for communication performance and
ACL Graph batch-shape coverage.

Serving scripts explicitly disable vLLM prefix caching. `vllm bench serve`
reuses its first generated prompt for endpoint validation and warmup, so leaving
the vLLM 0.13 default cache enabled would contaminate the measured workload.

## Serving

```bash
# precision: bf16, w8a8, or w4a8
# parallelism: tp4 or pp4
# max model length: prompt + output
scripts/serve.sh bf16 tp4 2048 8000

# QwQ's README-recommended static YaRN extension for requests above 40K.
HF_OVERRIDES_JSON='{"max_position_embeddings":131072,"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}' \
  scripts/serve.sh w8a8 pp4 65536 8000
```

Set `EXECUTION_MODE=eager` for a correctness-first run. The default is
`graph`, which leaves the vLLM Ascend performance path enabled.

Rebuild the random W4A8 checkpoint from the local BF16 model metadata with:

```bash
.venv/bin/python scripts/build_random_w4a8.py \
  --source models/QwQ-32B \
  --output models/QwQ-32B-W4A8-Random
```

## Benchmarks

Run a selected subset:

```bash
PRECISIONS="bf16" PARALLELISMS="tp4" SCENARIOS="concurrency" \
  REPEATS=1 benchmarks/run_matrix.sh
```

Run the default BF16/W8A8 matrix:

```bash
REPEATS=3 benchmarks/run_matrix.sh
```

Include the random W4A8 artifact explicitly:

```bash
PRECISIONS="bf16 w8a8 w4a8" REPEATS=3 benchmarks/run_matrix.sh
```

The matrix definition is:

| Scenario | Input / output | Sweep |
|---|---|---|
| High concurrency | 1024 / 256 tokens | concurrency 1, 4, 8, 16, 32, 64 |
| Long context | total 4K, 8K, 16K, 32K, 40K, 64K; output 128 | concurrency 4 |

QwQ-32B declares a native maximum of 40K tokens. The 64K point is an explicit
RoPE extrapolation experiment and is labeled as such in generated reports.
`run_matrix.sh` enables the static YaRN override for long-context runs by
default. Set `LONG_CONTEXT_HF_OVERRIDES=""` explicitly for native-only runs
whose points do not exceed 40K.

Each run is stored under `results/<run-id>/`. Generate or regenerate the
summary and exactly four comparison figures with:

```bash
.venv/bin/python benchmarks/summarize.py results/<run-id>
```

The completed four-NPU validation run is under
`results/20260730_qwq32b_4npu_nocache_v2/`.

The final three-precision report merges that baseline with the W4A8 run:

```bash
.venv/bin/python benchmarks/summarize.py \
  results/20260730_qwq32b_w4a8_random_noprefix \
  --include-run-dir results/20260730_qwq32b_4npu_nocache_v2
```

The combined report, memory table, and four figures are under
`results/20260730_qwq32b_w4a8_random_noprefix/`. The 910B4 per-group kernel
analysis is in `reports/W4A8_PERGROUP_ANALYSIS.md`; its microbenchmark and
exact vLLM profiler entry points are:

```bash
.venv/bin/python scripts/profile_w4a8_pergroup.py \
  --output-dir results/<run-id>/pergroup_profile
scripts/profile_w4a8_vllm.sh results/<run-id>/vllm_profile
.venv/bin/python scripts/summarize_w4a8_profile.py \
  --pergroup-dir results/<run-id>/pergroup_profile \
  --vllm-dir results/<run-id>/vllm_profile
```
