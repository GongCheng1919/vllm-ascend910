#!/usr/bin/env python3
"""Decode layer under NPU GRAPH REPLAY -- the number the real engine sees.

`bench_decode_arms.py` runs the layer in eager python and concluded the
whole-layer win is small because "non-projection work is 49-75% of the layer".
That conclusion does not survive measurement.  `host_vs_device.py` shows every
eager arm at batch<=16 is HOST-bound: the host needs ~900 us just to *enqueue*
a layer with all four projections replaced by no-ops, so the device is idle and
the projections' device time is invisible.  Two artefacts inflate the eager
denominator:

  1. python dispatch: ~900 us/layer of host time at batch=1, independent of
     arm.  vllm-ascend runs decode under ACL graph replay, where this is gone.
  2. `F.scaled_dot_product_attention(..., enable_gqa=True)` expands 8 KV heads
     to 40, costing 3x what `npu_fused_infer_attention_score` -- the op
     vllm-ascend actually calls -- costs (805 vs 272 us at batch=64).

This bench removes both: the layer uses the fused ops vllm-ascend uses
(npu_rms_norm, npu_fused_infer_attention_score, npu_swiglu) and is captured
into an NPUGraph, then replayed.  Custom W4A8 ops capture fine (verified).

usage: ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_decode_graph.py
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

# QwQ-32B, TP=1
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER = 5120, 40, 8, 128, 27648
KV_HIDDEN = KV_HEADS * HEAD_DIM
QKV_OUT = HEADS * HEAD_DIM + 2 * KV_HIDDEN
DEV = "npu:0"
EPS = 1e-6


def make_layer(lin, kv, torch_npu, batch, kv_len):
    """One decoder layer built from the ops vllm-ascend's decode path uses."""
    k_cache, v_cache, n1, n2 = kv

    def fwd(x):
        res = x
        h = torch_npu.npu_rms_norm(x, n1, EPS)[0]
        qkv = lin["qkv"](h)
        q, k, v = qkv.split([HEADS * HEAD_DIM, KV_HIDDEN, KV_HIDDEN], dim=-1)
        # KV append at a fixed slot: shapes are static, which graph capture needs.
        k_cache[:, kv_len:kv_len + 1] = k.view(batch, 1, KV_HEADS, HEAD_DIM)
        v_cache[:, kv_len:kv_len + 1] = v.view(batch, 1, KV_HEADS, HEAD_DIM)
        a = torch_npu.npu_fused_infer_attention_score(
            q.view(batch, 1, HEADS, HEAD_DIM), k_cache, v_cache,
            num_heads=HEADS, num_key_value_heads=KV_HEADS,
            input_layout="BSND", scale=HEAD_DIM ** -0.5)[0]
        x = res + lin["o"](a.reshape(batch, HIDDEN))
        res = x
        h = torch_npu.npu_rms_norm(x, n2, EPS)[0]
        h = torch_npu.npu_swiglu(lin["gate_up"](h), dim=-1)
        return res + lin["down"](h)
    return fwd


def eager_wall(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def graph_wall(fn, x, iters=50, warmup=5):
    """Capture `fn(x)` into an NPUGraph and time replays.

    Warmup runs on a side stream first (the same requirement as CUDA graphs):
    allocator and any lazy op init must be done before capture.
    """
    s = torch.npu.Stream()
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        for _ in range(warmup):
            fn(x)
    torch.npu.current_stream().wait_stream(s)
    torch.npu.synchronize()

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        fn(x)
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
    ap.add_argument("--kv-len", type=int, default=1024)
    ap.add_argument("--eager", action="store_true", help="also time the eager path")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch_npu
    W.load()
    torch.manual_seed(0)

    shapes = {"qkv": (QKV_OUT, HIDDEN), "o": (HIDDEN, HIDDEN),
              "gate_up": (2 * INTER, HIDDEN), "down": (HIDDEN, INTER)}
    wb = {n: (torch.randn(N, K) * 0.02).to(torch.bfloat16) for n, (N, K) in shapes.items()}
    wd = {n: t.to(DEV) for n, t in wb.items()}
    n1 = torch.ones(HIDDEN, dtype=torch.bfloat16, device=DEV)
    n2 = torch.ones(HIDDEN, dtype=torch.bfloat16, device=DEV)

    w8 = {n: torch.randint(-127, 127, (N, K), dtype=torch.int8, device=DEV)
          for n, (N, K) in shapes.items()}
    w8s = {n: torch.rand(shapes[n][0], device=DEV, dtype=torch.float32) * 1e-3 for n in shapes}

    def vllm_w8a8(name):
        wq, ws, wt = w8[name], w8s[name], w8[name].t()
        def f(x):
            qx, pts = torch_npu.npu_dynamic_quant(x)
            return torch_npu.npu_quant_matmul(qx, wt, ws, pertoken_scale=pts,
                                              output_dtype=torch.bfloat16)
        return f

    mg = {n: MidGroupW4A8Linear(wb[n], name=n, device=DEV) for n in shapes}

    def prequant_w4a8(name, b):
        """Fusion CEILING: the activation is quantised outside the graph, so only
        the GEMM is timed.  A fused rmsnorm(+quant) cannot beat a free quant.
        Numerically stale -- shapes and op sequence only."""
        m = mg[name]
        tile = 16 if b <= 16 else (64 if b <= 64 else 128)
        mp = (b + tile - 1) // tile * tile
        probe = torch.randn(mp, m.K, dtype=torch.bfloat16, device=DEV)
        a_hi, a_lo, a_s, a_k = torch.ops.npu.midgroup_quant_a(probe)
        def f(z):
            y = torch.ops.npu.midgroup_w4a8_gemm(a_hi, a_lo, a_s, a_k, m.w_q,
                                                 m.w_scale, m.w_ksum, m.w_zero, m.K)
            return y[:b]
        return f

    def prequant_w8a8(name, b):
        """Same ceiling for the W8A8 arm, so the comparison stays fair: vllm's
        path also pays a separate npu_dynamic_quant per projection."""
        wt, ws, K = w8[name].t(), w8s[name], shapes[name][1]
        qx = torch.randint(-127, 127, (b, K), dtype=torch.int8, device=DEV)
        pts = torch.rand(b, device=DEV, dtype=torch.float32) * 1e-3
        def f(z):
            return torch_npu.npu_quant_matmul(qx, wt, ws, pertoken_scale=pts,
                                              output_dtype=torch.bfloat16)
        return f

    rows = []
    print(f"kv_len={args.kv_len}   NPU graph replay, us/layer")
    hdr = (f"{'batch':>6}{'nonproj':>10}{'bf16':>10}{'w8a8':>10}{'w8a8-pq':>10}"
           f"{'w4a8-mlp':>10}{'w4a8-all4':>11}{'w4a8-pq':>10}"
           f"{'vs bf16':>9}{'vs w8a8':>9}{'fuse↑':>8}")
    print(hdr)
    for b in args.batches:
        x = (torch.randn(b, HIDDEN) * 0.5).to(torch.bfloat16).to(DEV)
        kc = torch.randn(b, args.kv_len + 1, KV_HEADS, HEAD_DIM,
                         dtype=torch.bfloat16, device=DEV)
        vc = torch.randn(b, args.kv_len + 1, KV_HEADS, HEAD_DIM,
                         dtype=torch.bfloat16, device=DEV)
        kv = (kc, vc, n1, n2)

        bf16_lin = {n: (lambda t: (lambda z: torch.matmul(z, t.t())))(wd[n]) for n in shapes}
        zeros = {n: torch.zeros(b, shapes[n][0], dtype=torch.bfloat16, device=DEV)
                 for n in shapes}
        arms = {
            "nonproj": {n: (lambda z0: (lambda z: z0))(zeros[n]) for n in shapes},
            "bf16": bf16_lin,
            "w8a8": {n: vllm_w8a8(n) for n in shapes},
            "w4a8-mlp": {**bf16_lin, "gate_up": mg["gate_up"], "down": mg["down"]},
            "w4a8-all4": {n: mg[n] for n in shapes},
            "w8a8-pq": {n: prequant_w8a8(n, b) for n in shapes},
            "w4a8-pq": {n: prequant_w4a8(n, b) for n in shapes},
        }
        res = {}
        for name, lin in arms.items():
            f = make_layer(lin, kv, torch_npu, b, args.kv_len)
            try:
                res[name] = graph_wall(f, x)
            except Exception as e:  # noqa: BLE001
                print(f"  [{b}/{name}] graph capture failed: {type(e).__name__}: {e}")
                res[name] = float("nan")
            if args.eager:
                res[name + "_eager"] = eager_wall(lambda: f(x))

        best = min(res["w4a8-mlp"], res["w4a8-all4"])
        print(f"{b:6d}{res['nonproj']:10.1f}{res['bf16']:10.1f}{res['w8a8']:10.1f}"
              f"{res['w8a8-pq']:10.1f}{res['w4a8-mlp']:10.1f}{res['w4a8-all4']:11.1f}"
              f"{res['w4a8-pq']:10.1f}"
              f"{res['bf16'] / best:8.2f}x{res['w8a8'] / best:8.2f}x"
              f"{res['w4a8-all4'] / res['w4a8-pq']:7.2f}x")
        if args.eager:
            print(f"{'':6}{'  (eager)':<4}{res['nonproj_eager']:6.1f}"
                  f"{res['bf16_eager']:10.1f}{res['w8a8_eager']:10.1f}"
                  f"{res['w8a8-pq_eager']:10.1f}{res['w4a8-mlp_eager']:10.1f}"
                  f"{res['w4a8-all4_eager']:11.1f}{res['w4a8-pq_eager']:10.1f}")
        rows.append((b, res))

    if args.out:
        keys = ["nonproj", "bf16", "w8a8", "w8a8-pq", "w4a8-mlp", "w4a8-all4", "w4a8-pq"]
        if args.eager:
            keys += [k + "_eager" for k in keys]
        with open(args.out, "w") as f:
            f.write("batch," + ",".join(k.replace("-", "_") + "_us" for k in keys) + "\n")
            for b, r in rows:
                f.write(f"{b}," + ",".join(f"{r[k]:.3f}" for k in keys) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
