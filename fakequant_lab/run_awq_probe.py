"""Channel-scaling pre-probe for W4A8-mg: can per-input-channel scaling help?

P1 found `W4A8-mg`'s GEMM noise is pinned by the *weight* side (int4 W ~16 dB)
while the *activation* side has 15-20 dB of headroom (int8 mid-group A 30-37 dB).
A per-input-channel scale is free at inference — it folds into the preceding norm
or the preceding Linear's output channels — unlike shrinking GK, which costs
2.4-4.1x kernel time (`MGCKPT/P0_COST.md`).  So it is worth knowing whether one
exists that pays.

    W' = W · diag(s),   X' = X · diag(1/s),   X'W'^T = XW^T exactly

Two opposite families, because the direction is not obvious a priori:

  `awq`   s = mean|X_j|^alpha      makes W harder, A easier  (AWQ / SmoothQuant
                                   direction; designed for W4A16 and for fixing
                                   *activation* outliers)
  `wflat` s = rms(W[:,j])^-beta    makes W easier, A harder  (the mirror; this is
                                   the direction our error budget actually wants)

`wflat` also answers the structural question directly: it can only help if the
weight difficulty is shared across output rows, i.e. genuinely per-input-channel.
The `w_only` columns repeat the probe with A left in bf16, isolating the weight
side from the activation side.

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python -m fakequant_lab.run_awq_probe \
        --gk 1024 --out fakequant_lab/results/awq_probe_gk1024.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    HAVE_NPU = True
except Exception:
    HAVE_NPU = False

from .data import wikitext2_windows
from .fake_linear import TARGET_LINEARS, Config, FakeQuantLinear
from .model_stream import StreamingQwen2
from .quant import QuantSpec, fake_quant, snr_db
from .run_tiers import _Capture

# (label, alpha on mean|X_j|, beta on rms(W[:,j]))
SCALINGS: List[Tuple[str, float, float]] = [
    ("base",      0.00, 0.00),
    ("awq.25",    0.25, 0.00),
    ("awq.5",     0.50, 0.00),
    ("wflat.5",   0.00, 0.50),
    ("wflat1",    0.00, 1.00),
    ("mix",       0.25, 0.50),
]


def channel_scale(X: torch.Tensor, W: torch.Tensor, alpha: float, beta: float
                  ) -> Optional[torch.Tensor]:
    """`s_j = mean|X_j|^alpha · rms(W[:,j])^-beta`, geometric-mean normalized.

    A global factor on `s` cancels exactly (both operands are quantized with
    per-group amax scales), so the normalization is cosmetic; it just keeps the
    printed intermediates in a sane range.
    """
    if alpha == 0.0 and beta == 0.0:
        return None
    s = torch.ones(W.shape[1], dtype=torch.float32, device=W.device)
    if alpha:
        s = s * X.float().abs().mean(dim=0).clamp_min(1e-8).pow(alpha)
    if beta:
        s = s * W.float().pow(2).mean(dim=0).sqrt().clamp_min(1e-8).pow(-beta)
    return s / s.log().mean().exp()


@torch.no_grad()
def probe(X: torch.Tensor, W: torch.Tensor, w_spec: QuantSpec,
          a_spec: Optional[QuantSpec], s: Optional[torch.Tensor]) -> Tuple[float, float]:
    """Returns (Tier-2 GEMM SNR, Tier-1 weight SNR) under scaling `s`.

    `a_spec=None` leaves the activation in bf16 (weight-only, the AWQ setting).
    """
    y_ref = F.linear(X, W)
    if s is None:
        Xp, Wp = X, W
    else:
        # Both operands are bf16 in deployment: W' is quantized from a bf16
        # tensor, X' arrives from a preceding op that absorbed 1/s.
        Wp = (W.float() * s).to(torch.bfloat16)
        Xp = (X.float() / s).to(torch.bfloat16)
    t1w = snr_db(Wp, fake_quant(Wp, w_spec))

    stub = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device,
                           dtype=torch.bfloat16)
    stub.weight.data = Wp
    if a_spec is None:
        w_deq = fake_quant(Wp, w_spec).to(torch.bfloat16)
        y = F.linear(Xp, w_deq)
    else:
        y = FakeQuantLinear(stub, Config("probe", w_spec, a_spec), "kernel")._kernel_gemm(Xp)
    return snr_db(y_ref, y), t1w


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/QwQ-32B")
    ap.add_argument("--gk", type=int, default=1024)
    ap.add_argument("--device", default="npu:0" if HAVE_NPU else "cpu")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="fakequant_lab/results/awq_probe.csv")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "npu":
        torch.npu.set_device(dev)

    ids = wikitext2_windows(args.model, args.split, args.seq, 1)
    model = StreamingQwen2(args.model, dev)
    n_layers = args.layers or model.num_layers
    a_spec = QuantSpec(8, args.gk, "A")
    w_sym = QuantSpec(4, args.gk, "W")
    w_asym = QuantSpec(4, args.gk, "W", sym=False)
    print(f"layers={n_layers}  gk={args.gk}  scalings={[s[0] for s in SCALINGS]}", flush=True)

    h = model.embed_ids(ids[:1])
    pos = torch.arange(args.seq, device=dev).unsqueeze(0)
    pe = model.position_embeddings(h, pos)

    rows: List[dict] = []
    t0 = time.perf_counter()
    for li in range(n_layers):
        layer = model.load_layer(li)
        with _Capture(layer) as cap:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
            captured = dict(cap.x)

        for name in TARGET_LINEARS:
            X, W = captured[name], layer.get_submodule(name).weight.data
            ax = X.float().abs().mean(dim=0)
            wr = W.float().pow(2).mean(dim=0).sqrt()
            row = dict(layer=li, linear=name, K=W.shape[1], N=W.shape[0],
                       x_chan_spread=round((ax.max() / ax.median()).item(), 2),
                       w_chan_spread=round((wr.max() / wr.median()).item(), 3))
            for label, alpha, beta in SCALINGS:
                s = channel_scale(X, W, alpha, beta)
                t2, t1 = probe(X, W, w_sym, a_spec, s)
                row[f"sym_{label}"] = round(t2, 3)
                row[f"t1w_{label}"] = round(t1, 3)
                row[f"asym_{label}"] = round(probe(X, W, w_asym, a_spec, s)[0], 3)
                row[f"wonly_{label}"] = round(probe(X, W, w_sym, None, s)[0], 3)
            rows.append(row)
        del layer, captured
        if dev.type == "npu":
            torch.npu.empty_cache()
        print(f"  layer {li:3d}/{n_layers}   [{time.perf_counter() - t0:6.1f}s]", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")

    labels = [s[0] for s in SCALINGS]
    for fam, title in (("t1w", "Tier-1 weight SNR (int4 sym)"),
                       ("wonly", "Tier-2, weight-only (A in bf16) — the AWQ objective"),
                       ("sym", "Tier-2, full W4A8-mg (int4 sym)"),
                       ("asym", "Tier-2, full W4A8-mg (int4 asym)")):
        print(f"\n{title} — median over layers, dB")
        print(f"{'linear':>18s} " + " ".join(f"{l:>9s}" for l in labels))
        for name in TARGET_LINEARS:
            sub = [r for r in rows if r["linear"] == name]
            print(f"{name:>18s} " +
                  " ".join(f"{st.median([r[f'{fam}_{l}'] for r in sub]):9.2f}" for l in labels))
        base = st.median([r[f"{fam}_base"] for r in rows])
        gains = {l: st.median([r[f"{fam}_{l}"] - r[f"{fam}_base"] for r in rows])
                 for l in labels if l != "base"}
        best = max(gains, key=gains.get)
        print(f"{'ALL (gain vs base)':>18s} " + f"{base:9.2f} " +
              " ".join(f"{gains[l]:+9.2f}" for l in labels if l != "base")
              + f"    best: {best} {gains[best]:+.2f} dB")

    print("\nper-input-channel spread (max/median), median over layers:")
    for name in TARGET_LINEARS:
        sub = [r for r in rows if r["linear"] == name]
        print(f"  {name:>18s}  activation {st.median([r['x_chan_spread'] for r in sub]):8.1f}x"
              f"   weight {st.median([r['w_chan_spread'] for r in sub]):6.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
