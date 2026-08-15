#!/usr/bin/env python3
"""Per-projection device time under NPU graph replay, with effective bandwidth.

Decode projections are weight-streaming: the honest yardstick is bytes of
weight moved / time.  Quoting that next to each arm shows immediately whether a
gap is "int4 does not help" or "our kernel is leaving bandwidth on the table".

usage: ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_proj_graph.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4a8_ops as W  # noqa: E402
from vllm_midgroup_linear import MidGroupW4A8Linear  # noqa: E402

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER = 5120, 40, 8, 128, 27648
QKV_OUT = HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM
DEV = "npu:0"


def graph_wall(fn, iters=50, warmup=5):
    s = torch.npu.Stream()
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        for _ in range(warmup):
            fn()
    torch.npu.current_stream().wait_stream(s)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        fn()
    torch.npu.synchronize()
    for _ in range(5):
        g.replay()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 128])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch_npu
    W.load()
    torch.manual_seed(0)

    shapes = {"qkv": (QKV_OUT, HIDDEN), "o": (HIDDEN, HIDDEN),
              "gate_up": (2 * INTER, HIDDEN), "down": (HIDDEN, INTER)}
    rows = []
    print(f"{'proj':>8}{'M':>5}{'bf16':>9}{'w8a8':>9}{'w4a8':>9}{'w4a8_gemm':>11}"
          f"{'quant_a':>9}   {'GB/s: bf16':>11}{'w8a8':>7}{'w4a8':>7}")
    for name, (N, K) in shapes.items():
        wb = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
        wd = wb.to(DEV)
        w8 = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=DEV)
        w8t = w8.t()
        w8s = torch.rand(N, device=DEV, dtype=torch.float32) * 1e-3
        mg = MidGroupW4A8Linear(wb, name=name, device=DEV)
        # bytes of weight-side state each arm streams from HBM
        by_bf16 = N * K * 2
        by_w8 = N * K
        by_w4 = (mg.w_q.numel() * mg.w_q.element_size()
                 + mg.w_scale.numel() * mg.w_scale.element_size()
                 + mg.w_zero.numel() * mg.w_zero.element_size()
                 + mg.w_ksum.numel() * mg.w_ksum.element_size())

        for m in args.batches:
            x = (torch.randn(m, K) * 0.5).to(torch.bfloat16).to(DEV)
            t_bf = graph_wall(lambda: torch.matmul(x, wd.t()))

            def f8():
                qx, pts = torch_npu.npu_dynamic_quant(x)
                return torch_npu.npu_quant_matmul(qx, w8t, w8s, pertoken_scale=pts,
                                                  output_dtype=torch.bfloat16)
            t_8 = graph_wall(f8)
            t_4 = graph_wall(lambda: mg(x))

            tile = 16 if m <= 16 else (64 if m <= 64 else 128)
            mp = (m + tile - 1) // tile * tile
            xp = torch.randn(mp, K, dtype=torch.bfloat16, device=DEV)
            a_hi, a_lo, a_s, a_k = torch.ops.npu.midgroup_quant_a(xp)
            t_g = graph_wall(lambda: torch.ops.npu.midgroup_w4a8_gemm(
                a_hi, a_lo, a_s, a_k, mg.w_q, mg.w_scale, mg.w_ksum, mg.w_zero, K))
            t_q = graph_wall(lambda: torch.ops.npu.midgroup_quant_a(xp))

            print(f"{name:>8}{m:5d}{t_bf:9.1f}{t_8:9.1f}{t_4:9.1f}{t_g:11.1f}{t_q:9.1f}   "
                  f"{by_bf16 / t_bf / 1e3:11.0f}{by_w8 / t_8 / 1e3:7.0f}"
                  f"{by_w4 / t_4 / 1e3:7.0f}")
            rows.append((name, m, N, K, t_bf, t_8, t_4, t_g, t_q,
                         by_bf16, by_w8, by_w4))
        del mg, w8, w8t, wd
        torch.npu.empty_cache()

    if args.out:
        with open(args.out, "w") as f:
            f.write("proj,M,N,K,bf16_us,w8a8_us,w4a8_us,w4a8_gemm_us,quant_a_us,"
                    "bytes_bf16,bytes_w8a8,bytes_w4a8\n")
            for r in rows:
                f.write(",".join(f"{v:.3f}" if isinstance(v, float) else str(v)
                                 for v in r) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
