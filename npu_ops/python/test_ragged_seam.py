"""Torch-side seam check for ragged K: the full python -> host -> kernel path.

Compares torch.ops.npu.midgroup_w4a8_gemm against an fp32 matmul of the
DEQUANTISED weight, which is exact for an integer cube (see the project note on
fp32 matmul emulating int32).  What this adds over the lab gate is the parts the
lab never touches: w4a8_ops.quantize_weight's padded packing, the C++ host's
Kpad/2 row-length check, and midgroup_quant_a driving the GEMM.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import w4a8_ops as W  # noqa: E402

W.load()
dev = "npu:0"
torch.manual_seed(0)

GROUP = 1024
CASES = [
    # (K, why)
    (5120, "aligned regression (o_proj TP=1)"),
    (27648, "aligned regression (down_proj TP=1)"),
    (2560, "short group, full slices (o TP=2)"),
    (13824, "short group, full slices (down TP=2)"),
    (1280, "short group + short slice (o TP=4)"),
    (6912, "short group + short slice (down TP=4)"),
    (640, "single short group (o TP=8)"),
    (3456, "down TP=8"),
    (1057, "real padding: last group 33 -> 64"),
    (1025, "real padding: last group 1 -> 64"),
    (777, "one short group only"),
    (1023, "odd K"),
    (5121, "odd K, last group 1"),
]

N = 256
M_LIST = [1, 7, 16, 64, 128]
bad = 0
print(f"{'K':>7} {'M':>4} {'Kpad':>7} {'G':>3} {'lastGK':>7} {'SNR dB':>8}  note")
for K, why in CASES:
    w = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
    wq, ws, wk, wz = W.quantize_weight(w)
    kpad = W.k_pad_elems(K)
    assert wq.shape == (N, kpad // 2), f"packed {tuple(wq.shape)} != {(N, kpad // 2)}"
    # Reference weight: dequantise exactly what the kernel will see, per group.
    wdq = torch.zeros(N, K, dtype=torch.float32)
    codes = torch.zeros(N, kpad, dtype=torch.int32)
    lo = (wq.to(torch.int16) & 0xF)
    hi = ((wq.to(torch.int16) >> 4) & 0xF)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    codes[:, 0::2] = lo.to(torch.int32)
    codes[:, 1::2] = hi.to(torch.int32)
    G = W.num_groups(K)
    stride = kpad // G if G == 1 else GROUP  # ceil(GROUP/64)*64 == GROUP here
    for g in range(G):
        real = W.group_real(g, K)
        c = codes[:, g * stride:g * stride + real].float()
        s = ws[g].float().unsqueeze(1)
        z = wz[g].float().unsqueeze(1)
        wdq[:, g * GROUP:g * GROUP + real] = (c - z) * s

    wqd, wsd, wkd, wzd = (t.to(dev) for t in (wq, ws, wk, wz))
    for M in M_LIST:
        x = (torch.randn(M, K) * 0.5).to(torch.bfloat16)
        y = W.linear(x.to(dev), wqd, wsd, wkd, wzd).float().cpu()
        # The activation is quantised on device; dequantise the same way for the
        # reference by round-tripping x through the op's own quantiser is not
        # available on CPU, so compare against the fp32 product of the exact
        # dequantised weight and the ORIGINAL x -- that folds activation
        # quantisation into the error, hence the 20 dB floor rather than 80.
        ref = x.float() @ wdq.t()
        err = (y - ref)
        snr = 10 * torch.log10((ref ** 2).sum() / (err ** 2).sum().clamp_min(1e-30))
        ok = snr.item() > 20.0 and torch.isfinite(y).all()
        if not ok:
            bad += 1
        print(f"{K:>7} {M:>4} {kpad:>7} {G:>3} {W.group_real(G - 1, K):>7} "
              f"{snr.item():>8.2f}  {'' if ok else 'FAIL '}{why if M == M_LIST[0] else ''}")

print(f"\n{'ALL PASS' if not bad else str(bad) + ' FAILURES'} "
      f"({len(CASES) * len(M_LIST)} shape/M combinations)")
sys.exit(1 if bad else 0)
