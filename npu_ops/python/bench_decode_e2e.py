#!/usr/bin/env python3
"""Decode-shaped layer benchmark: vLLM's current W4A8 path vs the P3 kernels.

Compares, on QwQ-32B's four Linear shapes with RANDOM weights:

  bf16     torch.matmul (the Phase 3 BF16 baseline's per-layer work)
  stock    torch_npu.npu_weight_quant_batchmatmul  <- what vllm_ascend does today
  midgroup torch.ops.npu.midgroup_{quant_a,w4a8_gemm}  <- P3

Random weights are fine here: this measures latency, and P2's accuracy gate has
not passed anyway, so no quality claim is being made.

usage: ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_decode_e2e.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4a8_ops as W  # noqa: E402
from vllm_midgroup_linear import MidGroupW4A8Linear, should_use_midgroup  # noqa: E402

# QwQ-32B, TP=1: (name, N, K)
SHAPES = [("qkv", 7168, 5120), ("o", 5120, 5120),
          ("gate_up", 55296, 5120), ("down", 5120, 27648)]
DEV = "npu:0"


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256])
    ap.add_argument("--group", type=int, default=1024)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch_npu
    W.load()
    torch.manual_seed(0)

    rows = []
    print(f"{'proj':9}{'M':>6}{'bf16':>10}{'stock':>10}{'midgroup':>10}"
          f"{'mg/stock':>10}{'mg/bf16':>10}{'dispatch':>10}")
    for name, N, K in SHAPES:
        w_bf16 = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
        w_dev = w_bf16.to(DEV)
        mg = MidGroupW4A8Linear(w_bf16, name=name, device=DEV)

        # Stock path: int4 weight packed into int32, [K, N], with a per-group
        # antiquant scale -- exactly what process_weights_after_loading builds.
        w_i8 = torch.randint(-8, 8, (K, N), dtype=torch.int32, device=DEV)
        w_pack = torch_npu.npu_convert_weight_to_int4pack(w_i8)
        aq_scale = torch.randn(K // args.group, N, device=DEV).to(torch.bfloat16)

        for M in args.ms:
            x = (torch.randn(M, K) * 0.5).to(torch.bfloat16).to(DEV)
            t_bf16 = timeit(lambda: torch.matmul(x, w_dev.t()))
            t_stock = timeit(lambda: torch_npu.npu_weight_quant_batchmatmul(
                x, w_pack, antiquant_scale=aq_scale, antiquant_group_size=args.group))
            t_mg = timeit(lambda: mg(x))
            use = "midgroup" if should_use_midgroup(name, M) else "fallback"
            print(f"{name:9}{M:6d}{t_bf16:10.1f}{t_stock:10.1f}{t_mg:10.1f}"
                  f"{t_stock/t_mg:9.2f}x{t_bf16/t_mg:9.2f}x{use:>10}")
            rows.append((name, M, N, K, t_bf16, t_stock, t_mg))

        del mg, w_dev, w_pack, w_i8, aq_scale
        torch.npu.empty_cache()

    print()
    print(f"{'M':>6}{'bf16':>10}{'stock':>10}{'midgroup':>10}{'mg/stock':>10}{'mg/bf16':>10}")
    for M in args.ms:
        sel = [r for r in rows if r[1] == M]
        if len(sel) != len(SHAPES):
            continue
        b, s, g = (sum(r[i] for r in sel) for i in (4, 5, 6))
        print(f"{M:6d}{b:10.1f}{s:10.1f}{g:10.1f}{s/g:9.2f}x{b/g:9.2f}x")

    if args.out:
        with open(args.out, "w") as f:
            f.write("proj,M,N,K,bf16_us,stock_us,midgroup_us\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.3f},{r[5]:.3f},{r[6]:.3f}\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
