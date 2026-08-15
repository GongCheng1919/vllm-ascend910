#!/usr/bin/env python3
"""对比三条 cube GEMM 路径在不同计算规模下的吞吐（TFLOPS）。

路径（与 vLLM 实际算子一一对应）:
  1) bf16 : torch.mm(BF16, BF16)                          —— BF16 cube
  2) int8 : torch_npu.npu_quant_matmul(INT8, INT8)        —— W8A8 实际算子, s8s8 cube + 向量反量化
  3) int4 : torch_npu.npu_weight_quant_batchmatmul(BF16, INT4pack, group=128)
                                                          —— W4A8 实际算子:
                                                             M<=16 走 MSD 3次 s4s4; M>16 走反量化成 BF16 的 BF16 cube

每个 M 值输出 median 耗时与有效 TFLOPS = 2*M*K*N / t。
用法:
  .venv/bin/python scripts/bench_3gemm.py --output-dir results/<run-id> \
      --m-values "1,8,16,32,64,128,256,1024,4096"
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch_npu

GROUP = 128


def bench(fn, iters=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch_npu.npu.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m-values", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096")
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--n", type=int, default=6912)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    k, n = args.k, args.n
    torch.manual_seed(20260809)
    torch_npu.npu.set_device(0)

    # 权重与 scale（与 vLLM 相同 dtype/布局）
    wbf16 = torch.randn((k, n), dtype=torch.bfloat16, device="npu")
    w8 = torch.randint(-128, 127, (k, n), dtype=torch.int8).npu()
    w4 = torch_npu.npu_convert_weight_to_int4pack(
        torch.randint(-8, 8, (k, n), dtype=torch.int32).npu())
    scale8 = torch.randn((n,), dtype=torch.bfloat16, device="npu")  # per-channel, W8A8 反量化
    scale4 = torch.ones((k // GROUP, n), dtype=torch.bfloat16, device="npu")  # per-group 128
    torch_npu.npu.synchronize()

    rows = []
    for m in [int(v) for v in args.m_values.split(",")]:
        x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device="npu")
        x_int8 = torch.randint(-128, 127, (m, k), dtype=torch.int8).npu()
        torch_npu.npu.synchronize()

        runs = [
            ("bf16", lambda: torch.mm(x_bf16, wbf16)),
            ("int8", lambda: torch_npu.npu_quant_matmul(
                x_int8, w8, scale8, output_dtype=torch.bfloat16)),
            ("int4", lambda: torch_npu.npu_weight_quant_batchmatmul(
                x_bf16, w4, antiquant_scale=scale4, antiquant_group_size=GROUP)),
        ]
        for name, fn in runs:
            us = bench(fn, iters=args.iters)
            tflops = 2.0 * m * k * n / (us * 1e-6) / 1e12
            rows.append({"m": m, "gemm": name, "median_us": round(us, 2),
                         "tflops": round(tflops, 2)})
            print(json.dumps(rows[-1]), flush=True)

    with (out / "gemm3.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["m", "gemm", "median_us", "tflops"])
        writer.writeheader()
        writer.writerows(rows)
    with (out / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"k": k, "n": n, "group": GROUP, "iters": args.iters}, handle, indent=2)


if __name__ == "__main__":
    main()
