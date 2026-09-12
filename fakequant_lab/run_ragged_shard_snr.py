"""Does sharding K for TP cost accuracy under the ragged-K route?  Real weights.

THE QUESTION.  Under TP the row-parallel weight is split along K and each shard is
quantised INDEPENDENTLY, so the effective group layout changes:

    TP=1, GK=1024, K=5120            -> 1024 x5                       (what P2 measured)
    TP=2, GK=1024, K_rank=2560, ragged -> per shard 1024,1024,512      (the NEW route)
    TP=2, GK=512,  K_rank=2560         -> per shard 512 x5             (the OLD route)

A short final group is FINER than a full one, so the ragged route should be no
worse than TP=1 -- the claim this script exists to check rather than assert.  The
old route is finer still; that was its one redeeming property, and it is priced
against its speed in MGCKPT/P6.5_TP.md.

Reported as SNR of the fake-quantised weight against the bf16 original, per
tensor, over the two ROW-PARALLEL projections (o_proj, down_proj) of a few real
QwQ-32B layers.  Column-parallel projections are not affected (N is split, K is
whole), so they are not in the table.

usage: python fakequant_lab/run_ragged_shard_snr.py [--model models/QwQ-32B] [--layers 4]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fakequant_lab import quant as q  # noqa: E402


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.float()
    err = got.float() - ref
    num = (ref * ref).sum()
    den = (err * err).sum().clamp_min(1e-30)
    return float(10.0 * torch.log10(num / den))


def fq_sharded(w: torch.Tensor, tp: int, gk: int) -> torch.Tensor:
    """Fake-quantise as TP would: split K into `tp` shards, quantise each alone.

    Each shard goes through `quantize_weight`'s grouping rule: full groups of `gk`
    plus a final short group of whatever remains (the ragged layout).  gk=None
    means "gk = K_rank / tp_of_the_old_route", handled by the caller.
    """
    K = w.shape[1]
    assert K % tp == 0, f"K={K} not divisible by tp={tp}"
    kr = K // tp
    out = torch.empty_like(w, dtype=torch.float32)
    for s in range(tp):
        sh = w[:, s * kr:(s + 1) * kr]
        n_full = (kr + gk - 1) // gk          # ceil: last group may be short
        pieces = []
        for g in range(n_full):
            lo = g * gk
            hi = min(kr, lo + gk)
            piece = sh[:, lo:hi].float()
            spec = q.QuantSpec(4, hi - lo, "W", sym=False)
            pieces.append(q.fake_quant(piece, spec))
        out[:, s * kr:(s + 1) * kr] = torch.cat(pieces, dim=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/QwQ-32B")
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    from fakequant_lab.model_stream import ShardedWeights  # noqa: WPS433

    want = []
    for li in range(args.layers):
        want.append(f"model.layers.{li}.self_attn.o_proj.weight")
        want.append(f"model.layers.{li}.mlp.down_proj.weight")

    sw = ShardedWeights(args.model)
    tensors = {}
    for name in want:
        if sw.has(name):
            tensors[name] = sw.get(name, "cpu", torch.bfloat16)
    if not tensors:
        print(f"no row-parallel weights found under {args.model}")
        return 1

    print(f"{'tensor':>44} {'K':>6} | {'TP=1':>7} "
          f"{'TP2 new':>8} {'TP2 old':>8} {'TP4 new':>8} {'TP4 old':>8} {'TP8 new':>8}")
    print("-" * 110)
    agg = {k: [] for k in ("tp1", "n2", "o2", "n4", "o4", "n8")}
    for name in want:
        w = tensors.get(name)
        if w is None:
            continue
        K = w.shape[1]
        r = {
            "tp1": snr_db(w, fq_sharded(w, 1, 1024)),
            "n2": snr_db(w, fq_sharded(w, 2, 1024)),    # ragged
            "o2": snr_db(w, fq_sharded(w, 2, 512)),     # old route
            "n4": snr_db(w, fq_sharded(w, 4, 1024)),
            "o4": snr_db(w, fq_sharded(w, 4, 256)),
            "n8": snr_db(w, fq_sharded(w, 8, 1024)),
        }
        for k, v in r.items():
            agg[k].append(v)
        short = name.replace("model.layers.", "L").replace(".weight", "")
        print(f"{short:>44} {K:>6} | {r['tp1']:>7.2f} {r['n2']:>8.2f} {r['o2']:>8.2f} "
              f"{r['n4']:>8.2f} {r['o4']:>8.2f} {r['n8']:>8.2f}")

    print("-" * 110)
    mean = {k: sum(v) / len(v) for k, v in agg.items() if v}
    print(f"{'mean':>44} {'':>6} | {mean['tp1']:>7.2f} {mean['n2']:>8.2f} "
          f"{mean['o2']:>8.2f} {mean['n4']:>8.2f} {mean['o4']:>8.2f} {mean['n8']:>8.2f}")
    print()
    print("READ IT AS: 'new' is the shipped route (GK=1024, ragged tail).  It must be")
    print(">= TP=1, because sharding only ADDS group boundaries.  'old' (GK=1024/TP) is")
    print("finer and so slightly better on SNR -- that is the accuracy it bought with")
    print("the speed it lost (see MGCKPT/P6.5_TP.md).")
    for tp, nk, ok in ((2, "n2", "o2"), (4, "n4", "o4")):
        d_new = mean[nk] - mean["tp1"]
        d_old = mean[ok] - mean["tp1"]
        print(f"  TP={tp}: new {d_new:+.2f} dB vs TP=1, old {d_old:+.2f} dB vs TP=1")
    print(f"  TP=8: new {mean['n8'] - mean['tp1']:+.2f} dB vs TP=1 "
          "(no old route: GK=128 is below the kernel floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
