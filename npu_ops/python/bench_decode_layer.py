#!/usr/bin/env python3
"""WHOLE decode-layer benchmark: BF16 vs mid-group W4A8, QwQ-32B geometry.

The earlier `bench_decode_e2e.py` timed only the four GEMM projections.  A real
decode layer also runs two RMSNorms, RoPE, attention over the KV cache, and two
residual adds -- none of which the quantised GEMM speeds up, and attention at
decode is KV-cache bandwidth bound.  Quoting the projections-only ratio
overstates the layer speedup, so this measures the layer.

Attention is the SAME code in both arms (bf16 SDPA over a materialised KV
cache), so the only difference is the four Linears.  Random weights: this is a
latency measurement and P2's accuracy gate has not passed.

usage: ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_decode_layer.py
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

# QwQ-32B, TP=1
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER = 5120, 40, 8, 128, 27648
QKV_OUT = HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM   # 5120 + 1024 + 1024
DEV = "npu:0"


def rms_norm(x, w, eps=1e-6):
    v = x.float()
    return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


class Layer:
    """One decoder layer.  `linears` maps name -> callable(x) -> y."""

    def __init__(self, linears, kv_len: int, batch: int):
        self.lin = linears
        self.n1 = torch.ones(HIDDEN, dtype=torch.bfloat16, device=DEV)
        self.n2 = torch.ones(HIDDEN, dtype=torch.bfloat16, device=DEV)
        # KV cache preallocated with room for this step's token, so the forward
        # writes in place instead of torch.cat-ing a fresh copy every call.
        self.k = torch.randn(batch, KV_HEADS, kv_len + 1, HEAD_DIM,
                             dtype=torch.bfloat16, device=DEV)
        self.v = torch.randn(batch, KV_HEADS, kv_len + 1, HEAD_DIM,
                             dtype=torch.bfloat16, device=DEV)
        self.batch = batch
        self.pos = kv_len

    def __call__(self, x):
        b = self.batch
        res = x
        h = rms_norm(x, self.n1)
        qkv = self.lin["qkv"](h)
        q, k, v = qkv.split([HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM,
                             KV_HEADS * HEAD_DIM], dim=-1)
        q = q.view(b, 1, HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(b, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(b, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        # In-place KV append + native GQA.  repeat_interleave-ing 8 KV heads up
        # to 40 would allocate 5x the cache per call and dominate the layer,
        # making the projections look far cheaper than they are -- real paged
        # attention never does that.
        self.k[:, :, self.pos:self.pos + 1] = k
        self.v[:, :, self.pos:self.pos + 1] = v
        a = F.scaled_dot_product_attention(q, self.k, self.v, enable_gqa=True)
        a = a.transpose(1, 2).reshape(b, HIDDEN)
        x = res + self.lin["o"](a)

        res = x
        h = rms_norm(x, self.n2)
        gu = self.lin["gate_up"](h)
        g, u = gu.split([INTER, INTER], dim=-1)
        h = F.silu(g) * u
        return res + self.lin["down"](h)


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 128])
    ap.add_argument("--kv-len", type=int, default=1024)
    ap.add_argument("--mg-projs", nargs="+", default=["gate_up", "down"],
                    help="which projections use the mid-group kernel; rest stay bf16")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch_npu  # noqa: F401
    W.load()
    torch.manual_seed(0)

    shapes = {"qkv": (QKV_OUT, HIDDEN), "o": (HIDDEN, HIDDEN),
              "gate_up": (2 * INTER, HIDDEN), "down": (HIDDEN, INTER)}
    wb = {n: (torch.randn(N, K) * 0.02).to(torch.bfloat16) for n, (N, K) in shapes.items()}
    wd = {n: t.to(DEV) for n, t in wb.items()}
    mg = {n: MidGroupW4A8Linear(wb[n], name=n, device=DEV) for n in args.mg_projs}

    bf16_lin = {n: (lambda t: (lambda x: torch.matmul(x, t.t())))(wd[n]) for n in shapes}
    mix_lin = dict(bf16_lin)
    for n in args.mg_projs:
        mix_lin[n] = (lambda m: (lambda x: m(x)))(mg[n])

    rows = []
    print(f"mid-group on: {', '.join(args.mg_projs)}   kv_len={args.kv_len}")
    print(f"{'batch':>6}{'bf16 layer':>12}{'mixed layer':>13}{'speedup':>9}"
          f"{'attn+norm':>11}{'share':>8}")
    for b in args.batches:
        x = (torch.randn(b, HIDDEN) * 0.5).to(torch.bfloat16).to(DEV)
        lb = Layer(bf16_lin, args.kv_len, b)
        lm = Layer(mix_lin, args.kv_len, b)
        t_b = timeit(lambda: lb(x))
        t_m = timeit(lambda: lm(x))
        # Everything that is NOT a projection.  The layer must be built ONCE:
        # constructing it inside the timed closure re-allocates the KV cache every
        # iteration, which is what made this column exceed the full layer.
        zeros = {n: torch.zeros(b, shapes[n][0], dtype=torch.bfloat16, device=DEV)
                 for n in shapes}
        ident = {n: (lambda z0: (lambda z: z0))(zeros[n]) for n in shapes}
        lo = Layer(ident, args.kv_len, b)
        t_o = timeit(lambda: lo(x))
        print(f"{b:6d}{t_b:12.1f}{t_m:13.1f}{t_b/t_m:8.2f}x{t_o:11.1f}{t_o/t_b*100:7.1f}%")
        rows.append((b, t_b, t_m, t_o))

    if args.out:
        with open(args.out, "w") as f:
            f.write("batch,bf16_layer_us,mixed_layer_us,nonproj_us\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]:.3f}\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
