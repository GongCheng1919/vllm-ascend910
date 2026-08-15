"""Tier 1 / Tier 2 SNR export for every Linear of a streamed Qwen2 model.

Vocabulary follows `NPU-OP/snr_midgroup_sweep.py`:

  Tier 1  single tensor  — SNR of fake_quant(X) vs X, and fake_quant(W) vs W
  Tier 2  single GEMM    — SNR of the fake-quant GEMM output vs the bf16 output

Both tiers are *isolated*: the activations fed in come from the bf16 stream, so
no error from earlier layers is folded in.  Tier 3 (error accumulated through a
layer) is what `runner.run_configs` reports as `layer_snr`.

Activations are captured with forward hooks on the unpatched layer, so the
capture itself cannot perturb the numbers.

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python -m fakequant_lab.run_tiers \
        --model models/QwQ-32B --gk 1024 --windows 4 --out results/tier12_gk1024.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    HAVE_NPU = True
except Exception:
    HAVE_NPU = False

from .data import wikitext2_windows
from .fake_linear import TARGET_LINEARS, FakeQuantLinear, build_configs
from .model_stream import StreamingQwen2
from .quant import fake_quant, snr_db

TIER1_PASS_DB = 32.0
TIER2_PASS_DB = 35.0


class _Capture:
    """Forward hooks that stash each target Linear's 2-D input."""

    def __init__(self, layer, names=TARGET_LINEARS):
        self.layer, self.names = layer, names
        self.x: Dict[str, torch.Tensor] = {}
        self._handles: List = []

    def __enter__(self):
        for name in self.names:
            mod = self.layer.get_submodule(name)
            mod._fq_name = name
            self._handles.append(mod.register_forward_pre_hook(self._hook))
        return self

    def _hook(self, mod, inputs):
        x = inputs[0]
        self.x[mod._fq_name] = x.reshape(-1, x.shape[-1]).detach()

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        # `with ... as cap` leaves `cap` bound after the block, so drop the
        # layer and the captured activations here rather than pinning a whole
        # extra layer for the rest of the stream.
        self.layer = None
        self.x.clear()
        return None


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/QwQ-32B")
    ap.add_argument("--gk", type=int, default=1024)
    ap.add_argument("--device", default="npu:0" if HAVE_NPU else "cpu")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=4, help="calibration windows")
    ap.add_argument("--layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="fakequant_lab/results/tier12.csv")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "npu":
        torch.npu.set_device(dev)

    ids = wikitext2_windows(args.model, args.split, args.seq, args.windows)
    print(f"calibration: {tuple(ids.shape)} tokens from wikitext-2/{args.split}")

    model = StreamingQwen2(args.model, dev)
    n_layers = args.layers or model.num_layers
    configs = [c for c in build_configs(args.gk) if not c.is_bf16]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    t0 = time.perf_counter()

    h = torch.cat([model.embed_ids(ids[i:i + 1]) for i in range(ids.shape[0])])
    pos = torch.arange(args.seq, device=dev).unsqueeze(0)
    pe = model.position_embeddings(h[:1], pos)

    for li in range(n_layers):
        layer = model.load_layer(li)
        with _Capture(layer) as cap:
            # One window at a time keeps the captured activations small.
            outs = []
            for b in range(h.shape[0]):
                outs.append(layer(h[b:b + 1], attention_mask=None, position_ids=pos,
                                  position_embeddings=pe))
                if b == 0:
                    captured = dict(cap.x)      # tiers use the first window only
            h = torch.cat(outs)
            del outs

        for name in TARGET_LINEARS:
            X = captured[name]
            W = layer.get_submodule(name).weight.data
            y_ref = F.linear(X, W)
            for cfg in configs:
                fq = FakeQuantLinear(layer.get_submodule(name), cfg, mode="kernel")
                rows.append(dict(
                    layer=li, linear=name, config=cfg.name,
                    M=X.shape[0], N=W.shape[0], K=W.shape[1],
                    tier1_x_db=round(snr_db(X, fake_quant(X, cfg.a)), 3),
                    tier1_w_db=round(snr_db(W, fake_quant(W, cfg.w)), 3),
                    tier2_db=round(snr_db(y_ref, fq._kernel_gemm(X)), 3),
                ))
                del fq
        del layer, captured
        if dev.type == "npu":
            torch.npu.empty_cache()
        worst = min(r["tier2_db"] for r in rows if r["layer"] == li)
        print(f"  layer {li:3d}/{n_layers}  worst tier2 = {worst:6.2f} dB"
              f"   [{time.perf_counter() - t0:6.1f}s]", flush=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out_path}")

    print("\nworst layer per config (Tier1 pass >= 32 dB, Tier2 pass >= 35 dB):")
    for cfg in configs:
        sub = [r for r in rows if r["config"] == cfg.name]
        w_x = min(sub, key=lambda r: r["tier1_x_db"])
        w_w = min(sub, key=lambda r: r["tier1_w_db"])
        w_2 = min(sub, key=lambda r: r["tier2_db"])
        print(f"  {cfg.name:>12s}  "
              f"T1-X {w_x['tier1_x_db']:6.2f} ({w_x['linear']}@L{w_x['layer']})  "
              f"T1-W {w_w['tier1_w_db']:6.2f} ({w_w['linear']}@L{w_w['layer']})  "
              f"T2 {w_2['tier2_db']:6.2f} ({w_2['linear']}@L{w_2['layer']})  "
              f"{'PASS' if (w_x['tier1_x_db'] >= TIER1_PASS_DB and w_w['tier1_w_db'] >= TIER1_PASS_DB and w_2['tier2_db'] >= TIER2_PASS_DB) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
