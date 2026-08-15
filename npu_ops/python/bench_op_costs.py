#!/usr/bin/env python3
"""Itemised per-op device cost, W4A8 vs W8A8, at decode M.

A single-op graph cannot resolve a decode-sized op: `graph_wall(lambda: None)`
is already ~67 us of replay floor.  Capturing R copies of the op in ONE graph
and subtracting the floor gives (T_R - T_0)/R, which is valid far below it.

The point of the table is to separate BANDWIDTH from FIXED LAUNCH COST.  Fit
`t = bytes/B + c` across the four projections and both terms fall out; that is
how P4_E2E.md §8.5 concluded our GEMM already streams at ~1165 GB/s (W8A8's
level) and its remaining deficit is a ~12 us/launch fixed term.

usage: ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_op_costs.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4a8_ops as W  # noqa: E402
from vllm_midgroup_linear import MidGroupW4A8Linear  # noqa: E402

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER = 5120, 40, 8, 128, 27648
QKV_OUT = HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM
DEV = "npu:0"
R = 20


def graph_wall(fn, iters=50, warmup=3):
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


def per_op(fn, floor):
    def rep():
        for _ in range(R):
            fn()
    return (graph_wall(rep) - floor) / R


def fit(points):
    """Least squares t = bytes/B + c over (bytes_MB, us); returns (GB/s, c_us)."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den          # us per MB
    c = (sy - slope * sx) / n
    return 1e3 / slope, c                       # GB/s, us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch_npu
    W.load()
    torch.manual_seed(0)
    floor = graph_wall(lambda: None)
    m = args.m
    tile = 16 if m <= 16 else (64 if m <= 64 else 128)
    mp = (m + tile - 1) // tile * tile
    print(f"M={m} (padded to {mp})   replay floor {floor:.1f} us   R={R}\n")
    print(f"{'proj':>8}{'MB(w4)':>8}{'MB(w8)':>8}{'pad':>7}{'copy_':>7}{'quantA':>8}"
          f"{'dynq':>7}{'gemm4':>8}{'gemm8':>8}{'w4 tot':>8}{'w8 tot':>8}")
    rows, pts4, pts8 = [], [], []
    for name, N, K in (("qkv", QKV_OUT, HIDDEN), ("o", HIDDEN, HIDDEN),
                       ("gate_up", 2 * INTER, HIDDEN), ("down", HIDDEN, INTER)):
        x = torch.randn(m, K, dtype=torch.bfloat16, device=DEV)
        xp = torch.randn(mp, K, dtype=torch.bfloat16, device=DEV)
        buf = torch.zeros(mp, K, dtype=torch.bfloat16, device=DEV)
        mg = MidGroupW4A8Linear((torch.randn(N, K) * 0.02).to(torch.bfloat16),
                                name=name, device=DEV)
        a_hi, a_lo, a_s, a_k = torch.ops.npu.midgroup_quant_a(xp)
        w8 = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=DEV)
        w8t, w8s = w8.t(), torch.rand(N, device=DEV, dtype=torch.float32) * 1e-3
        qx = torch.randint(-127, 127, (m, K), dtype=torch.int8, device=DEV)
        pts = torch.rand(m, device=DEV, dtype=torch.float32) * 1e-3

        mb4 = (mg.w_q.numel() + mg.w_scale.numel() * 2 + mg.w_zero.numel() * 2
               + mg.w_ksum.numel() * 4) / 1e6
        mb8 = N * K / 1e6

        t_pad = per_op(lambda: F.pad(x, (0, 0, 0, mp - m)), floor) if mp != m else 0.0
        t_cp = per_op(lambda: buf[:m].copy_(x), floor) if mp != m else 0.0
        t_qa = per_op(lambda: torch.ops.npu.midgroup_quant_a(xp), floor)
        t_dq = per_op(lambda: torch_npu.npu_dynamic_quant(x), floor)
        t_g4 = per_op(lambda: torch.ops.npu.midgroup_w4a8_gemm(
            a_hi, a_lo, a_s, a_k, mg.w_q, mg.w_scale, mg.w_ksum, mg.w_zero, K), floor)
        t_g8 = per_op(lambda: torch_npu.npu_quant_matmul(
            qx, w8t, w8s, pertoken_scale=pts, output_dtype=torch.bfloat16), floor)
        t_4 = per_op(lambda: mg(x), floor)

        def f8():
            q, p = torch_npu.npu_dynamic_quant(x)
            return torch_npu.npu_quant_matmul(q, w8t, w8s, pertoken_scale=p,
                                              output_dtype=torch.bfloat16)
        t_8 = per_op(f8, floor)

        print(f"{name:>8}{mb4:8.1f}{mb8:8.1f}{t_pad:7.1f}{t_cp:7.1f}{t_qa:8.1f}"
              f"{t_dq:7.1f}{t_g4:8.1f}{t_g8:8.1f}{t_4:8.1f}{t_8:8.1f}")
        rows.append((name, mb4, mb8, t_pad, t_cp, t_qa, t_dq, t_g4, t_g8, t_4, t_8))
        pts4.append((mb4, t_g4))
        pts8.append((mb8, t_g8))
        del mg, w8, w8t
        torch.npu.empty_cache()

    b4, c4 = fit(pts4)
    b8, c8 = fit(pts8)
    print(f"\nGEMM fit  t = bytes/B + c")
    print(f"  w4a8 : B = {b4:6.0f} GB/s   c = {c4:5.1f} us/launch")
    print(f"  w8a8 : B = {b8:6.0f} GB/s   c = {c8:5.1f} us/launch")
    print(f"  layer totals: w4a8 {sum(r[9] for r in rows):.1f} us   "
          f"w8a8 {sum(r[10] for r in rows):.1f} us")

    if args.out:
        with open(args.out, "w") as f:
            f.write("proj,mb_w4a8,mb_w8a8,pad_us,copy_us,quant_a_us,dynquant_us,"
                    "gemm_w4a8_us,gemm_w8a8_us,total_w4a8_us,total_w8a8_us\n")
            for r in rows:
                f.write(r[0] + "," + ",".join(f"{v:.3f}" for v in r[1:]) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
