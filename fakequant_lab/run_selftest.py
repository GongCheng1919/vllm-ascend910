"""P1 self-tests for the fake-quant primitives.

These check the claims the harness rests on, rather than assuming them:

  T1  scales really are bf16, and quantization used the bf16 value
  T2  int4/int8 codes stay in range; per-channel == one group spanning K
  T3  MSD split is exact for all 256 int8 values
  T4  the fp32 group GEMM equals an exact integer reference (the load-bearing one)
  T5  group-major export layout matches the kernel's [G, N] / [G, M]
  T6  how far `dequant` (the default) is from the kernel's arithmetic

Run:  ASCEND_RT_VISIBLE_DEVICES=1 python -m fakequant_lab.run_selftest
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    HAVE_NPU = True
except Exception:
    HAVE_NPU = False

from .fake_linear import Config, FakeQuantLinear
from .quant import (QMAX, QMIN, QuantSpec, fake_quant, msd_recombine, msd_split,
                    quantize, snr_db, to_group_major, w_ksum)

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def t1_scale_is_bf16(dev):
    print("\nT1  scale dtype / bf16-consistency")
    x = torch.randn(64, 1024, device=dev, dtype=torch.bfloat16)
    for spec in (QuantSpec(8, 256), QuantSpec(4, 256), QuantSpec(8, None),
                 QuantSpec(4, 256, sym=False)):
        qt = quantize(x, spec)
        check(f"{spec.label}: scale dtype is bf16", qt.scale.dtype == torch.bfloat16)
        check(f"{spec.label}: zero point present iff asymmetric",
              (qt.zero is None) == spec.sym)
        # Re-quantizing with the bf16 scale must reproduce the same codes; if
        # quantize() had used an fp32 scale internally, some codes would differ.
        G = spec.groups(1024)
        gl = spec.group_len(1024)
        xg = x.float().reshape(64, G, gl)
        re = torch.round(xg / qt.scale.float().unsqueeze(-1))
        if qt.zero is not None:
            re = re + qt.zero.float().unsqueeze(-1)
        re = re.clamp(QMIN[spec.bits], QMAX[spec.bits])
        check(f"{spec.label}: codes derive from the bf16 scale",
              torch.equal(re.reshape(64, 1024).to(torch.int8), qt.q))


def t2_ranges_and_pc(dev):
    print("\nT2  code range / per-channel == single group")
    x = torch.randn(32, 2048, device=dev, dtype=torch.bfloat16) * 3.0
    for bits in (4, 8):
        for sym in (True, False):
            q = quantize(x, QuantSpec(bits, 512, sym=sym)).q
            check(f"int{bits}{'' if sym else '-asym'}: codes in [{QMIN[bits]}, {QMAX[bits]}]",
                  int(q.min()) >= QMIN[bits] and int(q.max()) <= QMAX[bits],
                  f"min={int(q.min())} max={int(q.max())}")
    pc, gK = quantize(x, QuantSpec(8, None)), quantize(x, QuantSpec(8, 2048))
    check("per-channel == gk=K", torch.equal(pc.q, gK.q) and torch.equal(pc.scale, gK.scale))
    check("asymmetric is never worse than symmetric at the same GK",
          snr_db(x, fake_quant(x, QuantSpec(4, 512, sym=False)))
          > snr_db(x, fake_quant(x, QuantSpec(4, 512))))
    # Finer groups must not be worse than coarser ones.
    snrs = [snr_db(x, fake_quant(x, QuantSpec(4, g))) for g in (2048, 1024, 512, 256)]
    check("int4 SNR improves monotonically as GK shrinks",
          all(b >= a - 0.01 for a, b in zip(snrs, snrs[1:])),
          " ".join(f"gk{g}={s:.1f}dB" for g, s in zip((2048, 1024, 512, 256), snrs)))


def t3_msd(dev):
    print("\nT3  MSD split exactness")
    # NPU aclnnArange has no int8 kernel; build the sweep on CPU and move it.
    a = torch.arange(-128, 128, dtype=torch.int32).to(torch.int8).to(dev)
    hi, lo = msd_split(a)
    check("h, l both in [-8, 7]",
          int(hi.min()) >= -8 and int(hi.max()) <= 7 and int(lo.min()) >= -8 and int(lo.max()) <= 7)
    check("a == 16*h + l + 8 for all 256 int8 values", torch.equal(msd_recombine(hi, lo), a))
    # The nibble P3 will actually store is (a & 15) ^ 8; the cube reads it back
    # as a signed int4.  Check that round-trip so the packing convention is pinned.
    nibble = (a.to(torch.int32) & 15) ^ 8
    decoded = ((nibble + 8) & 15) - 8
    check("stored nibble (a&15)^8 decodes to l", torch.equal(decoded.to(torch.int8), lo))
    check("hi nibble is the top 4 bits", torch.equal(((a.to(torch.int32) >> 4) & 15),
                                                     hi.to(torch.int32) & 15))


def _exact_reference(x2, w, cfg) -> torch.Tensor:
    """CPU int64 reference: integer sums are done exactly, rescale in float64."""
    qa, qw = quantize(x2, cfg.a), quantize(w, cfg.w)
    a_q, a_s = qa.q.cpu(), qa.scale.cpu()
    w_q, w_s = qw.q.cpu(), qw.scale.cpu()
    w_z = None if qw.zero is None else qw.zero.cpu()
    K = x2.shape[-1]
    G, gl = cfg.a.groups(K), cfg.a.group_len(K)
    acc = torch.zeros(x2.shape[0], w.shape[0], dtype=torch.float64)
    for g in range(G):
        sl = slice(g * gl, (g + 1) * gl)
        a_g = a_q[:, sl].to(torch.int64)
        part = (a_g @ w_q[:, sl].to(torch.int64).T).to(torch.float64)      # exact
        if w_z is not None:
            part -= a_g.sum(1, keepdim=True).double() * w_z[:, g].double().unsqueeze(0)
        acc += part * (a_s[:, g : g + 1].double() * w_s[:, g].double())
    return acc


def t4_gemm_exactness(dev):
    print("\nT4  group GEMM == exact integer reference")
    torch.manual_seed(0)
    for M, N, K, gk, bits, sym in ((64, 512, 1024, 256, 4, True),
                                   (128, 256, 5120, 1024, 4, True),
                                   (32, 512, 5120, 1024, 8, True),
                                   (17, 640, 5120, None, 4, True),
                                   (64, 512, 5120, 1024, 4, False)):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        base = nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)
        cfg = Config("t", QuantSpec(bits, gk, "W", sym=sym), QuantSpec(8, gk, "A"))
        fq = FakeQuantLinear(base, cfg, mode="kernel")
        got = fq._kernel_gemm(x).double().cpu()
        ref = _exact_reference(x, base.weight.data, cfg)
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        # fp32 rescale/accumulate over groups is the only slack; the integer
        # sums themselves must be bit-exact.
        tag = f"M{M} N{N} K{K} gk={gk} int{bits}{'' if sym else '-asym'}"
        check(f"{tag}: max rel err < 1e-6", rel < 1e-6, f"rel={rel:.2e}")


def t5_group_major(dev):
    print("\nT5  group-major export layout (P0_COST.md D4)")
    w = torch.randn(640, 5120, device=dev, dtype=torch.bfloat16)
    spec = QuantSpec(4, 1024, "W")
    qt = quantize(w, spec)
    ks = w_ksum(qt.q, spec)
    gm_s, gm_k = to_group_major(qt.scale), to_group_major(ks)
    check("ws [N,G] -> [G,N]", tuple(gm_s.shape) == (5, 640) and gm_s.is_contiguous())
    check("w_ksum [N,G] -> [G,N] int32",
          tuple(gm_k.shape) == (5, 640) and gm_k.dtype == torch.int32)
    check("group-major is a faithful transpose", torch.equal(gm_s[2], qt.scale[:, 2]))
    check("w_ksum matches a direct sum",
          torch.equal(ks[:, 3], qt.q[:, 3072:4096].to(torch.int32).sum(-1)))


def t6_dequant_vs_kernel(dev):
    print("\nT6  cost of `dequant` (bf16 matmul) vs `kernel` arithmetic")
    torch.manual_seed(0)
    for gk in (1024, None):
        x = torch.randn(256, 5120, device=dev, dtype=torch.bfloat16)
        base = nn.Linear(5120, 1536, bias=False, device=dev, dtype=torch.bfloat16)
        cfg = Config("t", QuantSpec(4, gk, "W"), QuantSpec(8, gk, "A"))
        y_exact = FakeQuantLinear(base, cfg, "kernel")(x).float()
        y_fast = FakeQuantLinear(base, cfg, "dequant")(x).float()
        y_bf16 = base(x).float()
        d = snr_db(y_exact, y_fast)
        print(f"       gk={str(gk):>5s}  quant SNR(kernel vs bf16) = {snr_db(y_bf16, y_exact):6.2f} dB"
              f"   dequant-vs-kernel = {d:6.2f} dB")
        check(f"gk={gk}: `dequant` noise is well below quantization noise",
              d > snr_db(y_bf16, y_exact) + 15.0,
              "otherwise `dequant` would contaminate the curve")


def t7_gptq(dev):
    print("\nT7  GPTQ")
    from .gptq import HessianAccumulator, gptq_quantize, inverse_hessian
    torch.manual_seed(0)
    N, K, gk = 512, 1024, 256
    spec = QuantSpec(4, gk, "W")
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)

    # Correlated activations, so the Hessian actually carries information.
    A = torch.randn(K, K, device=dev) / (K ** 0.5)
    X = (torch.randn(4096, K, device=dev) @ A).to(torch.bfloat16)

    acc = HessianAccumulator(K, dev)
    acc.add(X)
    Hinv, dead = inverse_hessian(acc.H.clone())
    qt = gptq_quantize(W, Hinv, spec, dead)

    check("codes stay in the int4 range",
          int(qt.q.min()) >= QMIN[4] and int(qt.q.max()) <= QMAX[4])
    check("scale is bf16 and per group", qt.scale.dtype == torch.bfloat16
          and tuple(qt.scale.shape) == (N, K // gk))

    # GPTQ minimizes ||(W - Ŵ)X||, so compare *output* error against RTN.
    ref = F.linear(X, W)
    rtn = snr_db(ref, F.linear(X, fake_quant(W, spec).to(torch.bfloat16)))
    gptq = snr_db(ref, F.linear(X, qt.dequantize().to(torch.bfloat16)))
    check("GPTQ beats RTN on layer output SNR", gptq > rtn,
          f"RTN {rtn:.2f} dB -> GPTQ {gptq:.2f} dB ({gptq - rtn:+.2f})")

    # With an uncorrelated (identity) Hessian there is nothing to compensate
    # against, so GPTQ must collapse onto plain round-to-nearest.
    eye = torch.eye(K, device=dev)
    qt_i = gptq_quantize(W, *(inverse_hessian(eye.clone(), percdamp=0.0)[0],), spec)
    check("H = I makes GPTQ identical to RTN", torch.equal(qt_i.q, quantize(W, spec).q))

    # Asymmetric must work through the same path.
    qa = gptq_quantize(W, Hinv, QuantSpec(4, gk, "W", sym=False), dead)
    gptq_a = snr_db(ref, F.linear(X, qa.dequantize().to(torch.bfloat16)))
    check("GPTQ + asymmetric beats GPTQ + symmetric", gptq_a > gptq,
          f"sym {gptq:.2f} dB -> asym {gptq_a:.2f} dB ({gptq_a - gptq:+.2f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0" if HAVE_NPU else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "npu":
        torch.npu.set_device(dev)
    print(f"device: {dev}")

    t1_scale_is_bf16(dev)
    t2_ranges_and_pc(dev)
    t3_msd(dev)
    t4_gemm_exactness(dev)
    t5_group_major(dev)
    t6_dequant_vs_kernel(dev)
    t7_gptq(dev)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
