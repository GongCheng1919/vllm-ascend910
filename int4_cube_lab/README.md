# int4_cube_lab

Hand-written AscendC per-channel low-bit GEMM for Ascend 910B4, built to answer
one question: **does 910B4 have a native INT4 cube pipe, and is it 2x INT8?**

Answer: yes, and yes (1.98–2.02x, byte-matched). See [REPORT.md](REPORT.md).

A second kernel, **W4A8** (int8 activation, int4 weight), was then built on the
same pipeline to answer: *can a native W4A8 match W8A8, especially at small M?*
Answer: it beats it — 1.65–1.88x at M <= 16 on Qwen3-32B MLP shapes.
See [REPORT_W4A8.md](REPORT_W4A8.md).

```
kernels/
  perchannel_lowbit_gemm.inc   W4A4/W8A8 implementation (byte-parameterised)
  perchannel_int4_gemm.cpp     QBITS=4  -> mad_s4
  perchannel_int8_gemm.cpp     QBITS=8  -> mad      (controlled baseline)
  perchannel_w4a8_gemm.inc     W4A8 implementation (MSD split, stacked-A)
  perchannel_w4a8_gemm{,_m64,_m32,_m16}.cpp
host/
  gemm_harness.h               shared 8-step Kernel Launch harness (W4A4/W8A8)
  w4a8_harness.h               same, plus the two A planes and w_ksum
  gemm_ref.{h,cpp}             int4 packer, MSD splitter, OpenMP CPU reference
  main_perchannel_{int4,int8,w4a8}_gemm*.cpp
scripts/  build.sh run.sh profile.sh sweep.sh sweep_qwen3.sh sweep_w4a8.sh
bench/    bench_official.py    torch.mm bf16 + npu_quant_matmul int8
results/  *.csv
```

## Use

```bash
bash scripts/build.sh
export ASCEND_RT_VISIBLE_DEVICES=1     # device 0 is wedged — see REPORT.md §6

./build/perchannel_int4_gemm_test --rows 1024 --cols 1024 --k 2048 --check
./build/perchannel_int4_gemm_test --rows 512 --cols 512 --k 1024 --stress 30
./build/perchannel_w4a8_gemm_test --rows 1024 --cols 1024 --k 2048 --check
bash scripts/sweep.sh
bash scripts/sweep_w4a8.sh all           # W4A8 vs W8A8 vs W4A4
python3 bench/bench_official.py --out results/official.csv
```

Flags: `--check` (vs CPU reference), `--stress N` (re-launch + re-check N times),
`--profile` (+`--nosync` to batch launches and time kernels rather than host
round-trips), `--cold` (rotate weight copies to defeat L2 reuse), `--blocks N`
(override blockDim), `--warmup/--repeat`.

Shape constraints: `N % 128 == 0`; M is padded up to the build's `TILE_M`;
`K / elemsPerByte < 65536`.  K must be a multiple of the L0 slice depth:

| build | K multiple of |
|---|---|
| int4 | 512 |
| int8 | 256 |
| w4a8 `_m16/_m32/_m64` | 512 |
| w4a8 (TILE_M=128) | 256 |

## The one thing to know before editing the kernel

Every buffer, DMA descriptor and LoadData descriptor is in **bytes**, not
elements — an int4 fractal (16x64) and an int8 fractal (16x32) are both 512 B and
C0 is 32 B either way. That is what makes the two builds issue identical
instruction streams, which is what makes the byte-matched experiment valid. Only
`mm.k` is in elements. If you add an element-based quantity, divide by
`ELEMS_PER_BYTE` or you will reproduce the exact class of bug documented in
`../gemm_precision_lab/CHECKPOINT_2026-08-09.md`.
