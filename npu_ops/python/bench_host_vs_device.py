#!/usr/bin/env python3
"""Is the decode-layer benchmark host-bound or device-bound?

`P4_E2E.md` §7.2 concluded the whole-layer speedup is small because the
non-projection part (attention + norms + residuals) is 49-75% of the layer.
But the per-projection CSV and the whole-layer CSV do not add up:

  batch=1 bf16 projections (e2e_decode_layer.csv): 37+26+454+333 = 850 us
  batch=1 non-projection (e2e_decode_layer_full.csv):            854 us
  batch=1 whole bf16 layer:                                     1159 us   <-- < 850+854

Components that do not sum mean the wall clock is measuring something that
OVERLAPS -- i.e. host dispatch racing device execution.  This script separates
them: `issue` = time for the host to enqueue N iterations (no sync), `wall` =
same loop with a trailing sync.  issue ~= wall => HOST-bound.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

NPU_OPS = "/home/gongcheng/PureInt8LLMPretraining/vllm/npu_ops/python"
sys.path.insert(0, NPU_OPS)

import w4a8_ops as W  # noqa: E402
from vllm_midgroup_linear import MidGroupW4A8Linear  # noqa: E402

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER = 5120, 40, 8, 128, 27648
KV_HIDDEN = KV_HEADS * HEAD_DIM
QKV_OUT = HEADS * HEAD_DIM + 2 * KV_HIDDEN
DEV = "npu:0"
EPS = 1e-6


def rms_norm(x, w):
    v = x.float()
    return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w


class Attn:
    def __init__(self, batch, kv_len):
        self.k = torch.randn(batch, KV_HEADS, kv_len + 1, HEAD_DIM,
                             dtype=torch.bfloat16, device=DEV)
        self.v = torch.randn(batch, KV_HEADS, kv_len + 1, HEAD_DIM,
                             dtype=torch.bfloat16, device=DEV)
        self.pos, self.b = kv_len, batch

    def __call__(self, q, k, v):
        b = self.b
        q = q.view(b, 1, HEADS, HEAD_DIM).transpose(1, 2)
        self.k[:, :, self.pos:self.pos + 1] = k.view(b, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        self.v[:, :, self.pos:self.pos + 1] = v.view(b, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, self.k, self.v, enable_gqa=True)
        return a.transpose(1, 2).reshape(b, HIDDEN)


def make_layer(lin, attn, n1, n2):
    def fwd(x):
        res = x
        qkv = lin["qkv"](rms_norm(x, n1))
        q, k, v = qkv.split([HEADS * HEAD_DIM, KV_HIDDEN, KV_HIDDEN], dim=-1)
        x = res + lin["o"](attn(q, k, v))
        res = x
        gu = lin["gate_up"](rms_norm(x, n2))
        g, u = gu.split([INTER, INTER], dim=-1)
        return res + lin["down"](F.silu(g) * u)
    return fwd


def measure(fn, iters=30, warmup=5):
    """(issue_us, wall_us) per iteration."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()          # host has enqueued everything
    torch.npu.synchronize()
    t2 = time.perf_counter()          # device has drained
    return (t1 - t0) / iters * 1e6, (t2 - t0) / iters * 1e6


def main() -> int:
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
    w8s = {n: torch.rand(shapes[n][0], device=DEV, dtype=torch.float32) * 0.001
           for n in shapes}

    def vllm_w8a8(name):
        wq, ws = w8[name], w8s[name]
        def f(x):
            qx, pts = torch_npu.npu_dynamic_quant(x)
            return torch_npu.npu_quant_matmul(qx, wq.t(), ws, pertoken_scale=pts,
                                              output_dtype=x.dtype)
        return f

    mg = {n: MidGroupW4A8Linear(wb[n], name=n, device=DEV) for n in shapes}

    batches = [int(v) for v in (sys.argv[1:] or ["1", "16", "64"])]
    print(f"{'batch':>5} {'arm':<16}{'issue_us':>10}{'wall_us':>10}{'bound':>8}")
    for b in batches:
        x = (torch.randn(b, HIDDEN) * 0.5).to(torch.bfloat16).to(DEV)
        attn = Attn(b, 1024)
        bf16_lin = {n: (lambda t: (lambda z: torch.matmul(z, t.t())))(wd[n]) for n in shapes}
        v8_lin = {n: vllm_w8a8(n) for n in shapes}
        zeros = {n: torch.zeros(b, shapes[n][0], dtype=torch.bfloat16, device=DEV) for n in shapes}
        ident = {n: (lambda z0: (lambda z: z0))(zeros[n]) for n in shapes}

        arms = {
            "nonproj": ident,
            "bf16": bf16_lin,
            "vllm-w8a8": v8_lin,
            "ours-mlp": {**bf16_lin, "gate_up": mg["gate_up"], "down": mg["down"]},
            "ours-all4": {n: mg[n] for n in shapes},
        }
        for name, lin in arms.items():
            f = make_layer(lin, attn, n1, n2)
            iss, wall = measure(lambda: f(x))
            tag = "HOST" if iss > 0.9 * wall else ("dev" if iss < 0.5 * wall else "mixed")
            print(f"{b:5d} {name:<16}{iss:10.1f}{wall:10.1f}{tag:>8}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
