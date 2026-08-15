# QwQ-32B vLLM Ascend Project Context

> Work checkpoint: 2026-07-30 UTC
>
> Read this file first in a new session. Full measurements are in the result
> reports; historical plans are in `ROADMAP.md`.

## 0. Current Checkpoint

The first validation milestone is complete on four Ascend 910B4 NPUs:

- QwQ-32B BF16 baseline works with TP4 and PP4.
- The official QwQ-32B W8A8 checkpoint works with TP4 and PP4.
- A deterministic random-weight QwQ-32B W4A8 checkpoint works with TP4 and
  PP4. It is valid only for throughput, memory, and kernel-path analysis.
- High-concurrency and long-context measurements through 64K are complete.
- Prefix caching was disabled for all reported runs. Normal per-request KV
  cache was still enabled.
- ACL Graph was enabled for the reported runs.
- All four requested comparison plots were generated.
- The W4A8 group-128 execution path has been traced through vLLM-Ascend,
  CANN tiling, profiler data, and the local C220 Cube implementation.

Primary deliverables:

- Final three-precision report:
  `results/20260730_qwq32b_w4a8_random_noprefix/REPORT.md`
- Original BF16/W8A8 report:
  `results/20260730_qwq32b_4npu_nocache_v2/REPORT.md`
- W4A8 kernel analysis:
  `reports/W4A8_PERGROUP_ANALYSIS.md`
- Final plots:
  `results/20260730_qwq32b_w4a8_random_noprefix/*_throughput.png`

At the last check all NPUs were idle. This is not persistent state; run
`npu-smi info` before starting another server.

The workspace is not a Git repository. Do not assume `git status`, branches,
or commits are available here.

## 1. Goal and Measurement Contract

The project builds a QwQ-32B inference benchmark on vLLM-Ascend for a
910B4 system with weak inter-card connectivity.

Validated modes:

| Precision | Checkpoint | vLLM quantization |
| --- | --- | --- |
| BF16 | `models/QwQ-32B` | none |
| W8A8 | `models/QwQ-32B-W8A8` | `ascend` |
| W4A8 | `models/QwQ-32B-W4A8-Random` | `ascend` |

Parallel configurations:

- TP4: tensor parallel 4, pipeline parallel 1.
- PP4: tensor parallel 1, pipeline parallel 4.
- Selected logical NPU IDs: `0,1,4,5`.
- Corresponding `npu-smi` NPU IDs: `2,3,6,7`.

Benchmark scenarios:

- Concurrency: 1024 input tokens, 256 output tokens, concurrency
  `1,4,8,16,32,64`.
- Long context: concurrency 4, 128 output tokens, input lengths
  `4K,8K,16K,32K,40K,64K`.
- Native model context limit: 40960.
- The 64K point uses static YaRN with factor 4, original maximum 32768, and
  `max_position_embeddings=131072`.
- Only one timed repeat was collected per point. Results are suitable for
  machine validation and initial comparison, not variance-qualified claims.
- Long-context throughput does not establish model quality at 64K.

## 2. Validated Environment

Host:

- Architecture: AArch64.
- CPU: Kunpeng 920, 96 cores.
- RAM: approximately 502 GiB.
- CANN: 8.5.0 at `/usr/local/Ascend/cann-8.5.0`.
- Driver / `npu-smi`: 25.5.1.

Accelerators:

- 8 x Ascend `910B4-1`.
- 64 GiB HBM per NPU.
- No HCCS/P2P mesh is exposed.
- Cards use PCIe and are divided into switch groups 0-3 and 4-7.
- The four selected cards cross those two groups.

Project virtual environment:

| Package | Version |
| --- | --- |
| Python | 3.10 |
| torch | 2.8.0 |
| torch-npu | 2.8.0.post2 |
| torchvision | 0.23.0 |
| vllm | 0.13.0 |
| vllm-ascend | 0.13.0 |
| triton-ascend | 3.2.0 |
| transformers | 4.57.6 |
| numpy | 1.26.4 |
| matplotlib | 3.10.9 |

Always use `.venv`. The system Python environment has torch/torch-npu 2.7.1
for the successful INT8 pretraining work under `../nitro`; do not modify it.

Quick validation:

```bash
scripts/check_env.sh
source scripts/common.sh
.venv/bin/torchrun --standalone --nproc-per-node 4 scripts/check_hccl.py
```

## 3. Checkpoint Inventory

### BF16

- Path: `models/QwQ-32B`
- Source: ModelScope `Qwen/QwQ-32B`
- Size: approximately 62 GiB.
- 14 weight shards.
- Complete and load-tested.

### W8A8

- Path: `models/QwQ-32B-W8A8`
- Source: official vLLM-Ascend ModelScope checkpoint.
- Size: approximately 41 GiB.
- One safetensors file.
- Complete and load-tested.

### W4A8

- Path: `models/QwQ-32B-W4A8-Random`
- Deterministic random tensors with the real QwQ-32B shapes.
- Format: `W4A8_DYNAMIC`, version 1.0.0, group size 128.
- 10 shards, 3011 tensors, approximately 18.43 GiB of safetensors.
- Complete and load-tested.
- It has no language-model quality. Never use its outputs for accuracy,
  perplexity, or semantic evaluation.

BF16 and W8A8 were downloaded directly from ModelScope with proxy variables
unset and were hash-validated. No trustworthy QwQ-32B W4A8 checkpoint was
found: the apparent W4 candidate identified itself as W8A8S in its metadata,
which is why the random W4A8 checkpoint was built.

Rebuild the random checkpoint only if required:

```bash
.venv/bin/python scripts/build_random_w4a8.py \
  --source models/QwQ-32B \
  --output models/QwQ-32B-W4A8-Random
```

## 4. Throughput Checkpoint

Values below are output tokens/second.

### High Concurrency

| Mode | C1 BF16/W8/W4 | C16 BF16/W8/W4 | C64 BF16/W8/W4 |
| --- | ---: | ---: | ---: |
| TP4 | 21.00 / 21.07 / 21.56 | 239.59 / 241.99 / 237.73 | 468.58 / 418.75 / 417.61 |
| PP4 | 10.72 / 13.49 / 17.46 | 131.97 / 159.72 / 153.96 | 362.45 / 479.42 / 344.10 |

### Long Context

| Mode | 4K BF16/W8/W4 | 32K BF16/W8/W4 | 64K BF16/W8/W4 |
| --- | ---: | ---: | ---: |
| TP4 | 48.64 / 50.52 / 48.20 | 10.99 / 11.81 / 10.92 | 5.16 / 5.49 / 5.13 |
| PP4 | 29.06 / 36.99 / 40.14 | 10.58 / 12.17 / 11.75 | 4.92 / 5.45 / 5.17 |

Do not infer a stable precision ranking from these single-repeat numbers.
BF16, W8A8, and W4A8 convergence is plausible because the tested workload is
not purely weight-bandwidth-bound. Attention, communication, scheduler cost,
kernel launch cost, and KV-cache traffic remain.

### Memory

Approximate TP4 weight memory per rank:

- BF16: 15.41 GiB.
- W8A8: 10.22 GiB.
- W4A8: 4.83 GiB.

Approximate PP4 weight memory per rank:

- BF16: 15.29 GiB.
- W8A8: 10.18 GiB.
- W4A8: 4.78 GiB.

Peak HBM remains near 59 GiB because `--gpu-memory-utilization 0.90` allows
vLLM to assign memory freed by weight quantization to the KV cache. This is
expected and does not mean the compressed checkpoints consume BF16-sized
weight memory.

## 5. W4A8 Physical Execution Path

The evidence does not support either of these blanket claims:

- "910B4 executes the whole W4A8 model as BF16."
- "910B4 provides a direct S8 x S4 GEMM for this model."

The observed group-128 path depends on GEMM `M`.

### Small M, usually decode

For the ordinary projection case at `M <= 16`, CANN selects its MSD group
kernel:

1. BF16 activation is decomposed into three scaled signed-INT4 components.
2. Packed INT4 weights remain INT4.
3. Cube performs three S4 x S4 matrix multiplies with INT32 accumulation.
4. The partial results are reconstructed in FP32 and written in BF16.

This is real integer Cube work. It is not direct INT8 x INT4.

Local physical evidence:

- Ascend 910B4 maps to AIC-C-220 / `dav-c220-cube` in
  `/usr/local/Ascend/cann-8.5.0/aarch64-linux/data/platform_config/Ascend910B4.ini`.
- The C220 matmul implementation at
  `/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h`
  contains the `int4b_t, int4b_t -> int32_t` specialization and calls
  `mad_s4`.
- Exact tiling evidence is saved in
  `results/20260730_qwq32b_w4a8_random_noprefix/pergroup_profile/tiling_evidence.txt`.

The local platform registry has no `Intrinsic_mmad s8s4` entry. Absence of
S8 x S4 and absence of FP8 support do not imply absence of the independent
S4 x S4 integer instruction.

### Larger M, usually prefill

For `M > 16` in the traced ordinary projection:

1. The custom kernel loads packed INT4 weight tiles.
2. It applies the per-group scale and materializes a BF16 weight tile.
3. Cube executes BF16 x BF16 GEMM.

Thus W4 storage and HBM traffic are retained, but the large-M Cube operation
is BF16. Some very wide projections can select a general MSD split-K variant;
they remain in the same low-bit MSD family rather than becoming S8 x S4.

### Exact vLLM Profile

Each profiled rank recorded:

- 2048 `WeightQuantBatchMatmulV2` calls.
- Input dtypes: BF16 activation, INT4 weight, BF16 antiquant scale.
- Formats: ND activation, FRACTAL_NZ weight, ND scale.
- Approximately 35-37 microseconds average per call.
- 69.36-70.18% of NPU time in this operator.

Profiler summary:

`results/20260730_qwq32b_w4a8_random_noprefix/vllm_profile/weight_quant_summary.csv`

Full analysis and source references:

`reports/W4A8_PERGROUP_ANALYSIS.md`

The packaged operator object did not expose a usable textual disassembly, so
no binary opcode-dump claim is made. The conclusion rests on the selected
tiling path, runtime profile, CANN kernel source, 910B4 architecture mapping,
and C220 basic API.

## 6. Known Behaviors and Interpretations

### PP4

PP4 is supported and all requested PP4 points completed. A W8A8 PP4 server
occasionally exits after one `vllm bench serve` invocation. The benchmark
runner handles this by cold-starting clean server configurations. The
lifecycle issue is still unexplained.

### Weak interconnect versus TP4

ACL Graph can reduce host launch and dispatch overhead, but it does not remove
TP all-reduce traffic. TP4 working well in this matrix is consistent with
compute/communication overlap, relatively large GEMMs, batching, and graph
replay. It is not evidence that the weak interconnect is irrelevant.

### Triton messages

The observed Triton failure is a non-fatal import probe for an optional
GPT-OSS MoE kernel. QwQ-32B is dense and the measured path uses Ascend
operators, HCCL, and ACL Graph. `triton-ascend` remains installed as a
dependency; seeing its name does not prove that the QwQ linear path ran a
Triton kernel.

### `nocache`

`nocache` in result names means prefix caching was disabled. The normal
attention KV cache was enabled. Disabling prefix caching prevents repeated
prompts from receiving cross-request prefix-reuse benefits.

### 64K

A direct 64K attempt beyond the model's native 40960 limit failed in
`GatherV3`. Static YaRN plus the larger advertised position limit fixed
execution. Treat that point as a systems throughput measurement only.

## 7. Artifact Map

Final result directory:

```text
results/20260730_qwq32b_w4a8_random_noprefix/
├── REPORT.md
├── tp4_concurrency_throughput.png
├── pp4_concurrency_throughput.png
├── tp4_long_context_throughput.png
├── pp4_long_context_throughput.png
├── pergroup_profile/
├── vllm_profile/
└── ...
```

Important scripts:

| File | Purpose |
| --- | --- |
| `scripts/serve.sh` | Start BF16, W8A8, or W4A8 in TP4/PP4 |
| `benchmarks/run_matrix.sh` | Run concurrency and long-context matrices |
| `benchmarks/summarize.py` | Merge JSON results, report, and plots |
| `scripts/build_random_w4a8.py` | Build deterministic random W4A8 checkpoint |
| `scripts/profile_w4a8_pergroup.py` | Isolated group-quant profiling |
| `scripts/profile_w4a8_vllm.sh` | End-to-end W4A8 vLLM profiling |
| `scripts/summarize_w4a8_profile.py` | Summarize profiler output |
| `scripts/monitor_npu.py` | Record NPU memory/utilization |

## 8. Resume Commands

Inspect the machine before use:

```bash
cd /home/gongcheng/PureInt8LLMPretraining/vllm
npu-smi info
scripts/check_env.sh
```

Start representative servers:

```bash
scripts/serve.sh bf16 tp4 2048 8000
scripts/serve.sh w8a8 pp4 40960 8000
scripts/serve.sh w4a8 tp4 2048 8000
```

Start a 64K server:

```bash
HF_OVERRIDES_JSON='{"max_position_embeddings":131072,"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}' \
  scripts/serve.sh w8a8 pp4 65536 8000
```

Use eager mode only as an ACL Graph diagnostic:

```bash
EXECUTION_MODE=eager scripts/serve.sh bf16 tp4 2048 8000
```

Run a selected matrix:

```bash
PRECISIONS="bf16 w8a8 w4a8" \
PARALLELISMS="tp4 pp4" \
SCENARIOS="concurrency long_context" \
REPEATS=3 \
benchmarks/run_matrix.sh
```

Summarize one run:

```bash
.venv/bin/python benchmarks/summarize.py results/<run-id>
```

Recreate the current merged report:

```bash
.venv/bin/python benchmarks/summarize.py \
  results/20260730_qwq32b_w4a8_random_noprefix \
  --include-run-dir results/20260730_qwq32b_4npu_nocache_v2
```

Profile W4A8:

```bash
.venv/bin/python scripts/profile_w4a8_pergroup.py \
  --output-dir results/<run-id>/pergroup_profile

scripts/profile_w4a8_vllm.sh results/<run-id>/vllm_profile

.venv/bin/python scripts/summarize_w4a8_profile.py \
  --pergroup-dir results/<run-id>/pergroup_profile \
  --vllm-dir results/<run-id>/vllm_profile
```

## 9. Recommended Next Work

1. Rerun the primary comparison points at least three times and report median,
   minimum, maximum, and dispersion.
2. Isolate the PP4 W8A8 post-benchmark exit with server and worker logs.
3. Profile HCCL time, attention time, host scheduling, and ACL Graph replay to
   explain why TP4 tolerates the weak topology.
4. Sweep `max_num_batched_tokens`, concurrency, and KV-cache allocation rather
   than treating the current vLLM defaults as optimal.
5. Test a second TP4 replica on logical IDs `2,3,6,7` and measure aggregate
   whole-machine serving throughput.
6. Obtain a calibrated, real QwQ-32B W4A8 checkpoint before any quality claim.
7. Generate a SHA256 checkpoint manifest if the model directories will be
   moved or shared.

The next session should start from the final report and the W4A8 analysis,
then choose either statistical reruns or root-cause profiling. The functional
bring-up phase is already complete.
