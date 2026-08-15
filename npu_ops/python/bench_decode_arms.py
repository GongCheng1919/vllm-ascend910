#!/usr/bin/env python3
"""Whole decode-layer, four arms — isolates FUSION from BIT-WIDTH.

  bf16          torch.matmul projections
  vllm-w8a8     npu_dynamic_quant + npu_quant_matmul   (vllm_ascend's real path)
  nitro-w8a8    nitro's FUSED blocks: RMSNorm(+quant) -> GEMM, producer-side quant
  ours-w4a8     P3 mid-group W4A8, consumer-side quant (current state)

`vllm-w8a8` and `nitro-w8a8` are the SAME bit width and essentially the same
algorithm; they differ in that nitro quantises inside the kernel that produced
the activation, so the GEMM costs one launch instead of two.  Their ratio is
therefore a direct read on what fusing our W4A8 would buy.

Two caveats on the nitro arm, both making it PESSIMISTIC (see MGCKPT/P4_E2E.md):
  * its mid_group_gemm host pads M up to 128, so decode batches below 128 do
    extra rows;
  * it dual-quantises (row + col.T) because it is a training path; the col.T
    transpose is pure overhead for inference.
So nitro-w8a8 is a LOWER BOUND on fused inference performance.

Attention is identical in every arm.  Random weights; no quality claim.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NITRO = "/home/gongcheng/PureInt8LLMPretraining/nitro-workspace/nitro"
if NITRO not in sys.path:
    sys.path.insert(0, NITRO)

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
    """Shared attention state: preallocated KV cache, native GQA SDPA."""

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


def make_generic_layer(lin, attn, n1, n2):
    """bf16 / vllm-w8a8 / ours-w4a8 share this shape: norm -> proj, unfused."""
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


def make_nitro_layer(wts, attn, n1, n2):
    """nitro's fused blocks: B1 norm+quant+QKV, B3 O+res+norm+quant, B4 MLP+res."""
    from nitro.nn.transformer.int8_functions_jit_ascend import (
        after_attention_ascend, before_attention_ascend, mlp_residual_ascend)

    def fwd(x):
        q, k, v = before_attention_ascend(x, n1, EPS, wts["qkv"], KV_HIDDEN, None)
        attn_out = attn(q, k, v)
        new_res, normed, n_rq, n_rs, n_cqT, n_csT = after_attention_ascend(
            x, attn_out, n2, EPS, wts["o"], None)
        return mlp_residual_ascend(new_res, normed, n_rq, n_rs, n_cqT, n_csT,
                                   wts["gate_up"], wts["down"])
    return fwd


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
    ap.add_argument("--mg-projs", nargs="+", default=["gate_up", "down"])
    ap.add_argument("--rotate", type=int, default=4,
                    help="copies of the L2-resident weights (qkv/o) to cycle over")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch_npu
    W.load()
    # Prefer the copy built against THIS interpreter's torch.  nitro's shipped
    # .so is built for the system torch 2.7.1; loading it here registers the ops
    # under the wrong dispatch key (`MAIA:` instead of `PrivateUse1:`) and every
    # npu tensor is rejected as CPU.
    nitro_so = os.environ.get(
        "NITRO_SO",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "build-nitro-venv", "lib", "libnitro_npu_ops.so"))
    nitro_so = os.path.normpath(nitro_so)
    if not hasattr(torch.ops.npu, "mid_group_gemm"):
        if os.path.exists(nitro_so):
            torch.ops.load_library(nitro_so)
        else:
            print(f"[warn] {nitro_so} missing -- run npu_ops/build_nitro_for_venv.sh; "
                  "nitro arm will be skipped")
    torch.manual_seed(0)

    shapes = {"qkv": (QKV_OUT, HIDDEN), "o": (HIDDEN, HIDDEN),
              "gate_up": (2 * INTER, HIDDEN), "down": (HIDDEN, INTER)}
    wb = {n: (torch.randn(N, K) * 0.02).to(torch.bfloat16) for n, (N, K) in shapes.items()}
    wd = {n: t.to(DEV) for n, t in wb.items()}

    # COLD weights.  Timing the same layer 20x leaves its weights in the 168 MB
    # L2, but a real model walks 64 DIFFERENT layers, so every projection reads
    # from HBM.  qkv (73 MB bf16) and o (52 MB) FIT in L2 and were being measured
    # at ~2 TB/s -- L2 bandwidth -- which flatters bf16 exactly on the two
    # projections where the quantised path was losing.  gate_up (566 MB) and down
    # (283 MB) already exceed L2, so only qkv/o need rotating.
    ROT = {"qkv": args.rotate, "o": args.rotate, "gate_up": 1, "down": 1}
    wd_rot = {n: [wd[n]] + [wd[n].clone() for _ in range(ROT[n] - 1)] for n in shapes}
    n1 = torch.ones(HIDDEN, dtype=torch.bfloat16, device=DEV)
    n2 = torch.ones(HIDDEN, dtype=torch.bfloat16, device=DEV)

    # vllm native w8a8: int8 weight [N,K] + per-channel scale, activation quantised
    # per call by npu_dynamic_quant -- the real vllm_ascend path.
    w8 = {n: torch.randint(-127, 127, (N, K), dtype=torch.int8, device=DEV)
          for n, (N, K) in shapes.items()}
    w8s = {n: torch.rand(N, device=DEV, dtype=torch.float32) * 0.001 for n, (N, _) in shapes.items()}

    def vllm_w8a8(name):
        wq, ws = w8[name], w8s[name]
        def f(x):
            qx, pts = torch_npu.npu_dynamic_quant(x)
            return torch_npu.npu_quant_matmul(qx, wq.t(), ws, pertoken_scale=pts,
                                              output_dtype=x.dtype)
        return f

    mg = {n: MidGroupW4A8Linear(wb[n], name=n, device=DEV) for n in args.mg_projs}

    def prequant_linear(name, b):
        """Upper bound on what fusing the quant into the producer could buy.

        The activation is quantised ONCE, outside the timed region, and only the
        GEMM is timed.  A fused rmsnorm(+quant) kernel cannot beat a free quant,
        so `ours-prequant` brackets the fusion benefit from above.  It is NOT a
        correct layer -- the quantised activation is stale -- it only has the
        right shapes and the right op sequence.
        """
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

    rows = []
    print(f"kv_len={args.kv_len}   ours-w4a8 on: {', '.join(args.mg_projs)}")
    print(f"{'batch':>6}{'bf16':>10}{'vllm-w8a8':>11}{'nitro-w8a8':>12}{'ours-w4a8':>11}"
          f"{'ours-prequant':>13}{'fuse-gain':>10}{'pq/bf16':>9}")
    for b in args.batches:
        x = (torch.randn(b, HIDDEN) * 0.5).to(torch.bfloat16).to(DEV)
        attn = Attn(b, args.kv_len)

        def bf16_rot(name):
            copies = wd_rot[name]
            ctr = [0]
            def f(z):
                t = copies[ctr[0]]
                ctr[0] = (ctr[0] + 1) % len(copies)
                return torch.matmul(z, t.t())
            return f
        bf16_lin = {n: bf16_rot(n) for n in shapes}
        fb = make_generic_layer(bf16_lin, attn, n1, n2)
        t_bf = timeit(lambda: fb(x), iters=20)

        v8_lin = {n: vllm_w8a8(n) for n in shapes}
        fv = make_generic_layer(v8_lin, attn, n1, n2)
        t_v8 = timeit(lambda: fv(x), iters=20)

        ours_lin = dict(bf16_lin)
        for n in args.mg_projs:
            ours_lin[n] = (lambda m: (lambda z: m(z)))(mg[n])
        fo = make_generic_layer(ours_lin, attn, n1, n2)
        t_ow = timeit(lambda: fo(x), iters=20)

        pq_lin = dict(bf16_lin)
        for n in args.mg_projs:
            pq_lin[n] = prequant_linear(n, b)
        fp = make_generic_layer(pq_lin, attn, n1, n2)
        t_pq = timeit(lambda: fp(x), iters=20)

        try:
            with torch.no_grad():
                fn_ = make_nitro_layer(wd, attn, n1, n2)
                t_n8 = timeit(lambda: fn_(x), iters=20)
        except Exception as e:  # noqa: BLE001
            print(f"{b:6d}  nitro arm failed: {type(e).__name__}: {e}")
            t_n8 = float("nan")

        print(f"{b:6d}{t_bf:10.1f}{t_v8:11.1f}{t_n8:12.1f}{t_ow:11.1f}{t_pq:12.1f}"
              f"{t_ow/t_pq:9.2f}x{t_bf/t_pq:9.2f}x")
        rows.append((b, t_bf, t_v8, t_n8, t_ow, t_pq))

    if args.out:
        with open(args.out, "w") as f:
            f.write("batch,bf16_us,vllm_w8a8_us,nitro_w8a8_us,ours_w4a8_us,ours_prequant_us\n")
            for r in rows:
                f.write(",".join(str(v) if i == 0 else f"{v:.3f}"
                                 for i, v in enumerate(r)) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
