#!/usr/bin/env python3
"""P3 seam check: does the AscendC kernel compute what fakequant_lab's algorithm says?

Every other check in this lab compares the kernel against a C++ reference that
lives in the same directory and shares its conventions.  That cannot catch a
DISAGREEMENT BETWEEN THE ALGORITHM AND THE KERNEL -- a zero-point sign flip, a
different code range, an fp32-vs-bf16 scale -- because both sides would be wrong
together.  So here fakequant_lab drives:

  1. fakequant_lab quantises a random weight and activation with its own
     primitives (`quant.quantize`, which is what P2's PPL numbers used),
  2. its `FakeQuantLinear._kernel_gemm` produces the expected output,
  3. the same integer codes and bf16 scales are handed to the kernel via
     `--load-dir`,
  4. the two outputs are compared.

Anything that disagrees between `quant.py` and `midgroup_w4a8_gemm.inc` shows up
here and nowhere else.

usage: python scripts/seam_fakequant.py [--m 64] [--n 512] [--k 5120]
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM = os.path.dirname(LAB)
sys.path.insert(0, VLLM)

from fakequant_lab.fake_linear import FakeQuantLinear, build_configs  # noqa: E402
from fakequant_lab.quant import quantize  # noqa: E402

GK = 1024


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    """bf16 tensor -> raw uint16, the on-device representation."""
    return t.to(torch.bfloat16).view(torch.int16).cpu().numpy().astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--tile-m", type=int, default=64, choices=(16, 64, 128))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=os.path.join(LAB, "build", "seam_data"))
    args = ap.parse_args()

    M, N, K = args.m, args.n, args.k
    if M % args.tile_m:
        print(f"M={M} must be a multiple of tileM={args.tile_m} "
              f"(the harness pads, and the padding rows are not part of the seam)")
        return 2
    if K % GK:
        print(f"K={K} must be a multiple of GK={GK}")
        return 2
    G = K // GK

    torch.manual_seed(args.seed)

    # ---- 1. fakequant_lab quantises, exactly as P2's runs did ----------------
    cfg = build_configs(GK, include_asym=True)[-1]
    assert cfg.name == f"w4a8-mg{GK}-asym", cfg.name

    base = nn.Linear(K, N, bias=False)
    base.weight.data = (torch.randn(N, K) * 0.02).to(torch.bfloat16).float()
    lin = FakeQuantLinear(base, cfg, mode="kernel")

    x = (torch.randn(M, K) * 0.5).to(torch.bfloat16).float()
    qa = quantize(x, cfg.a)

    assert lin.w_zero is not None, "asym config produced no zero point"

    # ---- 2. the algorithm's answer ------------------------------------------
    y_ref = lin._kernel_gemm(x).to(torch.bfloat16)

    # ---- 3. hand the same bytes to the kernel -------------------------------
    # Kernel layout is GROUP-MAJOR: [G, M] and [G, N]; fakequant is [M, G] / [N, G].
    os.makedirs(args.dir, exist_ok=True)
    qa.q.to(torch.int8).cpu().numpy().tofile(f"{args.dir}/a_q.bin")
    lin.q_w.to(torch.int8).cpu().numpy().tofile(f"{args.dir}/w_q.bin")
    bf16_bits(qa.scale.T.contiguous()).tofile(f"{args.dir}/a_scale_g.bin")
    bf16_bits(lin.w_scale.T.contiguous()).tofile(f"{args.dir}/w_scale_g.bin")
    bf16_bits(lin.w_zero.T.contiguous()).tofile(f"{args.dir}/w_zero_g.bin")

    binary = os.path.join(LAB, "build",
                          f"midgroup_w4a8_gemm_m{args.tile_m}_g{GK}_asym_test")
    if not os.path.exists(binary):
        print(f"missing {binary}; run scripts/build.sh first")
        return 2
    out_dir = f"{args.dir}/out"
    cmd = [binary, "--rows", str(M), "--cols", str(N), "--k", str(K),
           "--load-dir", args.dir, "--dump", "--out-dir", out_dir]
    env = dict(os.environ)
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        return 1

    y_dev = torch.from_numpy(
        np.fromfile(f"{out_dir}/y_dev.bin", dtype=np.uint16).astype(np.int16)
    ).view(torch.bfloat16).reshape(M, N)

    # ---- 4. compare ---------------------------------------------------------
    a, b = y_ref.float(), y_dev.float()
    err = a - b
    snr = 10 * torch.log10(a.pow(2).sum() / err.pow(2).sum().clamp_min(1e-30))
    exact = int((y_ref.view(torch.int16) == y_dev.view(torch.int16)).sum())

    print(f"config     {cfg.name}   M={M} N={N} K={K} G={G} tileM={args.tile_m}")
    print(f"zero point range  [{int(lin.w_zero.float().min())}, "
          f"{int(lin.w_zero.float().max())}]  (code units)")
    print(f"bit-exact  {exact}/{M * N} bf16 elements")
    print(f"max |err|  {err.abs().max():.3e}   (|y| max {a.abs().max():.3e})")
    print(f"SNR        {snr:.2f} dB")

    ok = snr.item() >= 60.0
    print("RESULT     " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
