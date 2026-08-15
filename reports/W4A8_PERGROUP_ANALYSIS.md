# QwQ-32B W4A8 Dynamic on Ascend 910B4

## Scope

No trustworthy pre-quantized QwQ-32B W4A8 checkpoint was found on ModelScope.
The only search hit labeled QwQ W4A8 declared `W8A8S` in its quantization
metadata. The benchmark therefore uses
`models/QwQ-32B-W4A8-Random`, a deterministic random checkpoint with the real
QwQ-32B tensor shapes and the vLLM Ascend `W4A8_DYNAMIC` version 1.0.0 format.

The artifact is valid only for runtime compatibility, dense-kernel throughput,
HBM use, and operator profiling. Its output has no model-quality meaning.

## Measured Evidence

The complete matching matrix has 72 aggregate rows: BF16, W8A8, and W4A8
across TP4/PP4, six concurrency points, and six context lengths. All 24 W4A8
points completed without a failed request.

| Configuration | BF16 weights/rank | W8A8 weights/rank | W4A8 weights/rank | BF16 KV tokens | W4A8 KV tokens |
|---|---:|---:|---:|---:|---:|
| TP4 concurrency | 15.41 GB | 10.22 GB | 4.83 GB | 617,088 | 792,064 |
| PP4 concurrency | 15.29 GB | 10.18 GB | 4.78 GB | 597,120 | 770,944 |

At TP4 concurrency 1/4/8/16, W4A8 is within 1% to 3% of BF16. It falls to
0.87x BF16 at concurrency 32 and 0.89x at concurrency 64. PP4 exposes a
larger small-M benefit: W4A8 is 1.63x, 1.56x, and 1.46x BF16 at concurrency
1, 4, and 8, then falls to 0.95x at concurrency 64.

The direct group-128 microbenchmark measured the same BF16/INT4/BF16 operator
at M=1,8,16,32,128,1024. CANN debug logs report `Check msd group succ` through
M=16. At M=32 and above they report the explicit `M <= 128/8` rejection,
followed by `Check custom succ`.

The exact TP4 vLLM trace confirms this is not an ND-only microbenchmark
artifact. Each of the four ranks recorded 2,048 `WeightQuantBatchMatmulV2`
calls with BF16 activations, INT4 `FRACTAL_NZ` weights, and BF16 scales.
The operator accounted for 69.36% to 70.18% of measured NPU kernel time, with
an average invocation time of 35.35 to 36.94 microseconds.

## Runtime Entry Point

vLLM Ascend 0.13.0 implements this format in
`.venv/lib/python3.10/site-packages/vllm_ascend/quantization/w4a8_dynamic.py`.
The relevant behavior is:

1. The loaded packed weight is transposed and converted to `FRACTAL_NZ`.
2. Four packed INT8 bytes are viewed as one INT32 container. The logical
   elements remain INT4.
3. Inference calls `torch_npu.npu_weight_quant_batchmatmul` with a BF16
   activation, packed INT4 weight, BF16 per-group scale, and group size 128.
4. vLLM does not pass `inner_precise`, so the torch_npu default is 0.

The Python layer does not explicitly materialize a full BF16 weight tensor and
does not explicitly quantize the activation to INT8. The physical calculation
is selected inside CANN `WeightQuantBatchMatmulV2`.

## 910B4 Kernel Selection

CANN 8.5 registers MSD per-group ahead of the general custom template. On
Ascend 910B the tiler reports that native `mmad s8s4` and BF16 L1-to-BT
capabilities are unavailable, so the A3-specific one-pass S8xS4 route is not
selected.

For W4 per-group with group size 128, the MSD templates require:

- `M <= group_size / 8`, therefore `M <= 16`
- no transpose of A or B
- K divisible by 128
- N divisible by 64
- BF16/FP16 output rather than INT8 output

The optimized MSD-group template also requires `K <= 13952`; the general MSD
split-K template handles larger aligned K. All QwQ TP4 and PP4 Linear shapes
meet the alignment requirements. The PP4 MLP down projection has K=27648 and
therefore uses the general MSD split-K template at small M.

## What the Cube and Vector Cores Execute

### Physical INT4 instruction evidence

The `int4b_t` types in the MSD template are not merely checkpoint or API
metadata. This host identifies itself as `910B4-1`; its installed
`Ascend910B4.ini` maps the device to `AIC-C-220`, `dav-c220-cube`, and
`NpuArch=2201`. The CANN 8.5 C220 implementation of `MmadCal` explicitly
accepts the tuple `int32_t, int4b_t, int4b_t` and dispatches that tuple to the
`mad_s4` intrinsic. The official CANN 8.5 `Mmad` support table likewise lists
`int4b_t x int4b_t -> int32_t` for Atlas A2 products.

This is independent of FP8 support. Integer S4 dot products and FP8 floating
point matrix multiplication are separate Cube capabilities. The runtime
message `Platform not support Intrinsic_mmad s8s4` only rejects the mixed
S8 x S4 instruction; it does not reject S4 x S4. The legacy
`AICoreintrinsicDtypeMap` entry in the local `Ascend910B4.ini` does not
enumerate S4 x S4, but the W4A8 tiler does not use that entry to gate the MSD
S4 x S4 template. The compile-time C220 basic API, selected runtime tiling key,
and successful kernel execution provide the relevant evidence for this path.

### Decode with M at most 16

This is a real low-bit Cube path, but it is not a one-pass INT8 x INT4 GEMM:

1. Vector cores load BF16 activations and cast them to FP32.
2. They calculate an activation maximum per 128-element group.
3. For an INT4 weight, they unfold each normalized activation into three
   rounded INT4 components.
4. Cube cores execute INT4 x INT4 Matmul operations with INT32 accumulation.
5. Vector cores combine the three INT32 results in FP32, apply activation and
   weight scales, and cast the output back to BF16.

This multi-scale decomposition avoids persistent BF16 weights, but it performs
multiple low-bit GEMMs plus Vector pre/post-processing.

### Prefill or decode with M greater than 16

The MSD templates reject these shapes. The 910B custom MIX template then:

1. Vector cores load packed INT4 weight tiles.
2. They cast the tile through FP16/FP32, apply the per-group scale, and cast
   the dequantized tile to BF16.
3. The BF16 tile is written to a rotating workspace cache.
4. Cube cores consume the cached BF16 tile in a BF16 Matmul.

This does not expand the entire checkpoint to BF16 in HBM, but the active
weight tile is genuinely dequantized to BF16 before the Cube GEMM.

## Conclusion

`W4A8_DYNAMIC` on 910B4 is shape-dependent:

- Small-M decode: multi-scale INT4 x INT4 Cube GEMMs, INT32 accumulation, FP32
  reconstruction.
- Large-M prefill and decode: Vector tile dequantization followed by BF16 Cube
  GEMM.
- It is not uniformly an INT8 activation GEMM and it is not uniformly a
  dequantize-everything-to-BF16 implementation.

The final measured throughput, memory table, four comparison figures, and
profiler evidence are stored under
`results/20260730_qwq32b_w4a8_random_noprefix/`.

## Evidence Paths

- vLLM entry:
  `.venv/lib/python3.10/site-packages/vllm_ascend/quantization/w4a8_dynamic.py`
- NZ conversion:
  `.venv/lib/python3.10/site-packages/vllm_ascend/utils.py`
- CANN tiling registry:
  `/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_registry.cpp`
- MSD group eligibility:
  `/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_msd_group.cpp`
- MSD group kernel:
  `/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_msd_group.h`
- C220 physical `Mmad` implementation:
  `/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h`
- 910B4 architecture mapping and capability metadata:
  `/usr/local/Ascend/cann-8.5.0/aarch64-linux/data/platform_config/Ascend910B4.ini`
- General MSD split-K kernel:
  `/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_msd_split_k.h`
- Custom tile antiquantization:
  `/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_common.h`
- Benchmark runner:
  `benchmarks/run_matrix.sh`
- Random checkpoint builder:
  `scripts/build_random_w4a8.py`
- Direct profiler summary:
  `results/20260730_qwq32b_w4a8_random_noprefix/pergroup_profile/kernel_summary.csv`
- Preserved CANN tiling decisions:
  `results/20260730_qwq32b_w4a8_random_noprefix/pergroup_profile/tiling_evidence.txt`
- Exact vLLM profiler summary:
  `results/20260730_qwq32b_w4a8_random_noprefix/vllm_profile/weight_quant_summary.csv`
- Final three-precision report:
  `results/20260730_qwq32b_w4a8_random_noprefix/REPORT.md`
