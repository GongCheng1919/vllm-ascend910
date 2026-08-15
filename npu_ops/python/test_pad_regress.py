"""Bitwise regression for moving the TILE_M pad into midgroup_quant_a.

Two modes, both run on the NPU with the SAME weights and activations:

  --mode golden : load the PRE-CHANGE .so and reproduce the old wrapper exactly
                  (pad x into a zero buffer in python, then quant, then gemm,
                  then slice).  Dumps y to a .npz.
  --mode new    : load the current build and call the new path (no python pad).
                  Compares against the .npz bit-for-bit.

The old and new .so cannot coexist in one process (same op names), hence two
runs.  Weights/activations are regenerated from the same seed on CPU, so the
inputs are identical without shipping them through the file.

usage:
  .venv/bin/python pad_regress.py --mode golden --lib <old.so> --out y.npz
  .venv/bin/python pad_regress.py --mode new    --ref y.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

VLLM = "/home/gongcheng/PureInt8LLMPretraining/vllm"
sys.path.insert(0, os.path.join(VLLM, "npu_ops", "python"))

# The padding is M-side only, so N is kept small to keep the CPU-side weight
# quantisation quick.  Both real K values are covered (5120 = 5 groups, an
# uneven number, and 27648 = 27 groups, the most groups any QwQ layer has).
SHAPES = [
    ("qkv", 6144, 5120),
    ("o", 512, 5120),
    ("down", 512, 27648),
]
ROWS = [1, 2, 3, 15, 16, 17, 33, 64, 65, 100, 128, 129, 200, 256]


def make_inputs(idx, N, K, m, dev):
    # NOT hash(): python salts str hashing per process, and these two runs are
    # separate processes -- that silently gave every combo different inputs.
    g = torch.Generator().manual_seed(1000 + idx)
    w = torch.randn(N, K, generator=g).to(torch.bfloat16)
    gx = torch.Generator().manual_seed(m * 7919 + idx)
    x = torch.randn(m, K, generator=gx).to(torch.bfloat16).to(dev)
    return w, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("golden", "new"), required=True)
    ap.add_argument("--lib", default=None, help="explicit .so for --mode golden")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()

    import torch_npu  # noqa: F401
    dev = "npu:0"

    if args.lib:
        torch.ops.load_library(args.lib)
    import w4a8_ops as W
    W.load()

    results = {}
    for idx, (name, N, K) in enumerate(SHAPES):
        w, _ = make_inputs(idx, N, K, 1, dev)
        wq, ws, wk, wz = W.quantize_weight(w)
        wq, ws, wk, wz = (t.to(dev) for t in (wq, ws, wk, wz))
        for m in ROWS:
            _, x = make_inputs(idx, N, K, m, dev)
            if args.mode == "golden":
                tile = 16 if m <= 16 else (64 if m <= 64 else 128)
                mp = (m + tile - 1) // tile * tile
                x2 = x
                if mp != m:
                    buf = torch.zeros(mp, K, dtype=x.dtype, device=dev)
                    buf[:m].copy_(x)
                    x2 = buf
                a = torch.ops.npu.midgroup_quant_a(x2)
                y = torch.ops.npu.midgroup_w4a8_gemm(a[0], a[1], a[2], a[3],
                                                     wq, ws, wk, wz, K)
                y = y[:m]
            else:
                y = W.linear(x, wq, ws, wk, wz)
            torch.npu.synchronize()
            results[f"{name}_{m}"] = y.cpu().view(torch.int16).numpy()

    if args.mode == "golden":
        np.savez(args.out, **results)
        print(f"[golden] wrote {len(results)} tensors to {args.out}")
        return 0

    ref = np.load(args.ref)
    bad = 0
    for k, v in results.items():
        r = ref[k]
        if r.shape != v.shape:
            print(f"FAIL {k}: shape {v.shape} != golden {r.shape}")
            bad += 1
            continue
        n_diff = int((r != v).sum())
        if n_diff:
            print(f"FAIL {k}: {n_diff}/{r.size} bf16 words differ")
            bad += 1
    print(f"\n{len(results) - bad}/{len(results)} shape/M combos BIT-IDENTICAL "
          f"to the pre-change build")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
