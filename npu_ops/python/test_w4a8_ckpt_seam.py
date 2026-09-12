"""Seam check: exported checkpoint == runtime conversion, bitwise.

The exported checkpoint replaces `process_weights_after_loading`'s BF16->int4
conversion.  If the two disagree, every number measured before this change stops
being comparable to every number measured after it, and nothing would say so --
the model would just be quietly different.  So the gate is BITWISE, not SNR.

Two things are checked, because they can fail independently:

  1. per-projection : exported tensors == W.quantize_weight(original BF16)
  2. fusion         : concat(q,k,v) along N == quantize_weight(fused qkv)
                      concat(gate,up)      == quantize_weight(fused gate_up)

(2) is the load-bearing assumption of the whole layout: vLLM fuses q/k/v and
gate/up at load time, so if fusing changed the quantisation the checkpoint would
be wrong in a way (1) alone cannot see.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4a8_ops as W  # noqa: E402

SUF = ("weight_packed", "weight_scale", "weight_zero", "weight_ksum")


def _open_map(path: str):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    handles: dict[str, object] = {}

    def get(key: str) -> torch.Tensor:
        f = idx[key]
        if f not in handles:
            handles[f] = safe_open(os.path.join(path, f), framework="pt")
        return handles[f].get_tensor(key)
    return idx, get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/QwQ-32B")
    ap.add_argument("--ckpt", default="models/QwQ-32B-W4A8-MG")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 31, 63])
    args = ap.parse_args()

    _, src_get = _open_map(args.src)
    _, ck_get = _open_map(args.ckpt)
    fails = 0
    checks = 0

    for li in args.layers:
        pfx = f"model.layers.{li}"
        projs = [f"{pfx}.self_attn.{p}_proj" for p in ("q", "k", "v", "o")] + \
                [f"{pfx}.mlp.{p}_proj" for p in ("gate", "up", "down")]

        # ---- 1. per-projection, bitwise ----
        for name in projs:
            w = src_get(f"{name}.weight").to(torch.bfloat16)
            ref = W.quantize_weight(w)                       # (q, s[G,N], k[G,N], z[G,N])
            got = (ck_get(f"{name}.weight_packed"),
                   ck_get(f"{name}.weight_scale").T.contiguous(),
                   ck_get(f"{name}.weight_ksum").T.contiguous(),
                   ck_get(f"{name}.weight_zero").T.contiguous())
            for tag, a, b in zip(("packed", "scale", "ksum", "zero"), ref, got):
                checks += 1
                if not (a.shape == b.shape and torch.equal(a, b)):
                    fails += 1
                    print(f"FAIL {name}.{tag}: ref{tuple(a.shape)} vs got{tuple(b.shape)}")

        # ---- 2. fusion, bitwise ----
        for fused_name, members in (("qkv", ("q", "k", "v")), ("gate_up", ("gate", "up"))):
            base = f"{pfx}.self_attn" if fused_name == "qkv" else f"{pfx}.mlp"
            ws = [src_get(f"{base}.{m}_proj.weight").to(torch.bfloat16) for m in members]
            ref = W.quantize_weight(torch.cat(ws, dim=0))
            cat = []
            for i, suf in enumerate(("weight_packed", "weight_scale", "weight_ksum", "weight_zero")):
                parts = [ck_get(f"{base}.{m}_proj.{suf}") for m in members]
                t = torch.cat(parts, dim=0)                  # exported layout: [N, *]
                cat.append(t if i == 0 else t.T.contiguous())
            for tag, a, b in zip(("packed", "scale", "ksum", "zero"), ref, cat):
                checks += 1
                if not (a.shape == b.shape and torch.equal(a, b)):
                    fails += 1
                    print(f"FAIL fuse {base}.{fused_name}.{tag}: "
                          f"ref{tuple(a.shape)} vs got{tuple(b.shape)}")
        print(f"[seam] layer {li:3d}  ok", flush=True)

    print(f"\n[seam] {checks - fails}/{checks} bitwise checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
