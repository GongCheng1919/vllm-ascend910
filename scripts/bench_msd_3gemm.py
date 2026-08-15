#!/usr/bin/env python3
"""独立对比 MSD 三次 GEMM 的成本（与 vLLM 无关）。

固定形状 (M, K=5120, N=6912, group_size=128)，同一种子、同一权重，
对比五种配置在同一个 CANN 算子/参考 GEMM 上的耗时：

  1) ref_bf16   : torch.mm BF16xBF16             —— 1 次 GEMM 参考
  2) w8          : int8 权重 WQBMM (MSD 2 分量, 127/254)
  3) w4_d0       : int4 权重 + |x|<=8    (数学上只需 d0)
  4) w4_d0d1     : int4 权重 + |x|<=128  (数学上 d0,d1)
  5) w4_d0d1d2   : int4 权重 + |x|<=2048 (d0,d1,d2 全非零)

若 3==4==5，证明内核无条件执行 3 次 GEMM（mmad 不因数值为零而跳过），
三次 GEMM 的成本就是总耗时里减去固定向量/流水开销后的全部。
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch_npu

K = 5120
N = 6912
GROUP = 128


def wq(x, weight, scale):
    return torch_npu.npu_weight_quant_batchmatmul(
        x,
        weight,
        antiquant_scale=scale,
        antiquant_group_size=GROUP,
    )


def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch_npu.npu.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    return times[len(times) // 2], times[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m-values", default="1,8,16,32")
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260809)
    torch_npu.npu.set_device(0)

    raw_w4 = torch.randint(-8, 8, (K, N), dtype=torch.int32).npu()
    w4 = torch_npu.npu_convert_weight_to_int4pack(raw_w4)
    w8 = torch.randint(-128, 127, (K, N), dtype=torch.int8).npu()
    wbf16 = torch.randn((K, N), dtype=torch.bfloat16, device="npu").npu()
    scale = torch.ones((K // GROUP, N), dtype=torch.bfloat16, device="npu")
    torch_npu.npu.synchronize()

    rows = []
    RNG = 2048.0  # w4_d0d1d2 幅度≈2048(3分量)；w4_d0≈8(1分量)；w4_d0d1≈128(2分量)
    for m in [int(v) for v in args.m_values.split(",")]:
        x_bf16 = torch.randn((m, K), dtype=torch.bfloat16, device="npu")
        x_d0 = torch.randn((m, K), dtype=torch.bfloat16, device="npu") * (RNG / 256)
        x_d0d1 = torch.randn((m, K), dtype=torch.bfloat16, device="npu") * (RNG / 16)
        x_d0d1d2 = torch.randn((m, K), dtype=torch.bfloat16, device="npu") * RNG
        torch_npu.npu.synchronize()

        runs = [
            ("ref_bf16", lambda: torch.mm(x_bf16, wbf16)),
            ("w8", lambda: wq(x_bf16, w8, scale)),
            ("w4_d0", lambda: wq(x_d0, w4, scale)),
            ("w4_d0d1", lambda: wq(x_d0d1, w4, scale)),
            ("w4_d0d1d2", lambda: wq(x_d0d1d2, w4, scale)),
        ]
        for name, fn in runs:
            median, minimum = bench(fn, iters=args.iters)
            row = {"m": m, "config": name, "median_us": round(median, 2), "min_us": round(minimum, 2)}
            rows.append(row)
            print(json.dumps(row), flush=True)

    with (out / "msd_3gemm.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["m", "config", "median_us", "min_us"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
