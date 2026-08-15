#!/usr/bin/env python3
"""Official-path GEMM baselines on Ascend 910B4.

  BF16   torch.mm
  INT8   torch_npu.npu_quant_matmul          (per-channel + per-token, bf16 out)
  W4A16  torch_npu.npu_weight_quant_batchmatmul  (bf16 act x int4 weight —
         no int4 cube; dequantises to bf16 and runs the bf16 pipe. This is what
         a W4 vLLM path does today, so it is the number our int4-cube kernel
         has to beat to be worth building.)

Emits csv: kernel,M,N,K,avg_us,tflops

  ASCEND_RT_VISIBLE_DEVICES=1 python3 bench/bench_official.py --square
  ASCEND_RT_VISIBLE_DEVICES=1 python3 bench/bench_official.py \
      --shapes 1,51200,5120 1024,5120,25600 --out results/qwen3_official.csv
"""
import argparse
import os
import time

import torch
import torch_npu  # noqa: F401  (registers the 'npu' device)


# 910B4 L2 is 168 MiB.  Looping over one weight tensor serves it from L2, which
# real inference never does — 64 distinct layers stream from HBM.  COLD makes
# each path rotate over enough weight copies to blow past L2 between reuses.
COLD = False
COLD_FOOTPRINT = 1 << 30
MAX_ROTATE = 16


def rotate_count(weight_bytes):
    if not COLD:
        return 1
    return max(2, min(MAX_ROTATE, -(-COLD_FOOTPRINT // weight_bytes)))


def bench(fn, warmup=5, repeat=20):
    """fn takes the iteration index so it can pick a rotating weight copy."""
    for i in range(warmup):
        fn(i)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for i in range(repeat):
        fn(i)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e6  # us


def bench_bf16(M, N, K, w, r):
    a = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    rot = rotate_count(K * N * 2)
    bs = [torch.randn(K, N, dtype=torch.bfloat16, device="npu") for _ in range(rot)]
    return bench(lambda i: torch.mm(a, bs[i % rot]), w, r)


def bench_int8(M, N, K, w, r):
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="npu")
    rot = rotate_count(K * N)
    bs = [torch.randint(-8, 8, (K, N), dtype=torch.int8, device="npu") for _ in range(rot)]
    w_scale = torch.rand(N, dtype=torch.float32, device="npu") * 0.001 + 0.0005
    a_scale = torch.rand(M, dtype=torch.float32, device="npu") * 0.001 + 0.0005

    def run(i=0):
        return torch_npu.npu_quant_matmul(
            a, bs[i % rot], w_scale, pertoken_scale=a_scale,
            output_dtype=torch.bfloat16
        )

    run()
    return bench(run, w, r)


def bench_w4a16(M, N, K, w, r):
    a = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    rot = rotate_count(K * N // 2)
    wqs = []
    for _ in range(rot):
        wi = torch.randint(-8, 8, (K, N), dtype=torch.int32, device="npu")
        wqs.append(torch_npu.npu_convert_weight_to_int4pack(wi.contiguous()))
    scale = torch.rand(N, dtype=torch.bfloat16, device="npu") * 0.001 + 0.0005
    zero = torch.zeros(N, dtype=torch.bfloat16, device="npu")

    def run(i=0):
        return torch_npu.npu_weight_quant_batchmatmul(a, wqs[i % rot], scale, zero)

    run()
    return bench(run, w, r)


PATHS = (("torch.mm_bf16", bench_bf16),
         ("npu_quant_matmul_int8", bench_int8),
         ("npu_weight_quant_bmm_w4a16", bench_w4a16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=[],
                    help="M,N,K triples")
    ap.add_argument("--square", action="store_true",
                    help="also run 512..8192 cubes")
    ap.add_argument("--out", default="results/official.csv")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--cold", action="store_true",
                    help="rotate over many weight copies to defeat L2 reuse")
    args = ap.parse_args()

    global COLD
    COLD = args.cold

    shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes]
    if args.square or not shapes:
        shapes += [(s, s, s) for s in (512, 1024, 2048, 4096, 8192)]

    print(f"device: ASCEND_RT_VISIBLE_DEVICES="
          f"{os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '(unset)')}")
    torch.npu.set_device(0)

    rows = []
    for name, fn in PATHS:
        for (M, N, K) in shapes:
            try:
                us = fn(M, N, K, args.warmup, args.repeat)
                tflops = 2.0 * M * N * K / us / 1e6
                rows.append((name, M, N, K, f"{us:.2f}", f"{tflops:.2f}"))
                print(f"{name:28s} M={M:<5d} N={N:<6d} K={K:<6d} "
                      f"{us:10.2f} us  {tflops:8.2f} TFLOPS")
            except Exception as e:  # noqa: BLE001 - report and keep sweeping
                rows.append((name, M, N, K, "FAILED", "FAILED"))
                print(f"{name:28s} M={M:<5d} N={N:<6d} K={K:<6d} "
                      f"FAILED: {type(e).__name__}: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("kernel,M,N,K,avg_us,tflops\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
