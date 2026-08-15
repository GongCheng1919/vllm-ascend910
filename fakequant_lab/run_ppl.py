"""WikiText-2 perplexity for all five fake-quant configs in a single weight pass.

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python -m fakequant_lab.run_ppl \
        --model models/QwQ-32B --gk 1024 --windows 0 --out results/ppl_gk1024.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
    HAVE_NPU = True
except Exception:
    HAVE_NPU = False

from .data import wikitext2_windows
from .fake_linear import Config, build_configs
from .quant import QuantSpec
from .model_stream import StreamingQwen2
from .runner import run_configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/QwQ-32B")
    ap.add_argument("--gk", type=int, default=1024)
    ap.add_argument("--device", default="npu:0" if HAVE_NPU else "cpu")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--mode", default="dequant",
                    choices=("dequant", "kernel", "fast", "exact"),
                    help="dequant = ordinary fake quant (bf16 matmul, default); "
                         "kernel = reproduce the AscendC integer arithmetic (8-19x slower)")
    ap.add_argument("--states-device", default=None,
                    help="where hidden states live between layers; default = compute device")
    ap.add_argument("--configs", default="", help="comma-separated subset; bf16 is always kept")
    ap.add_argument("--asym", action="store_true",
                    help="also run int4 weights with a per-group zero point (not the frozen spec)")
    ap.add_argument("--gptq", default=None, action="append", nargs="?", const="",
                    help="`name=DIR` — evaluate config `name` with GPTQ weights from a "
                         "run_gptq.py output dir. Repeatable. `name` may be a new label, "
                         "in which case a config is created from --gk/--asym.")
    ap.add_argument("--gk-sweep", default="",
                    help="comma-separated GKs; runs bf16 + w8a8 + w4a8-mg at each GK in ONE "
                         "weight pass, instead of the five-config table")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="fakequant_lab/results/ppl.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "npu":
        torch.npu.set_device(dev)

    ids = wikitext2_windows(args.model, args.split, args.seq,
                            args.windows or None)
    print(f"eval set: {tuple(ids.shape)} = {ids.numel()} tokens "
          f"from wikitext-2/{args.split}", flush=True)

    model = StreamingQwen2(args.model, dev)
    if args.layers:
        model.num_layers = args.layers
    if args.gk_sweep:
        gks = [int(s) for s in args.gk_sweep.split(",")]
        configs = [Config("bf16", None, None),
                   Config("w8a8", QuantSpec(8, None, "W"), QuantSpec(8, None, "A"))]
        for g in gks:
            configs.append(Config(f"w4a8-mg{g}",
                                  QuantSpec(4, g, "W"), QuantSpec(8, g, "A")))
            if args.asym:
                configs.append(Config(f"w4a8-mg{g}-asym",
                                      QuantSpec(4, g, "W", sym=False), QuantSpec(8, g, "A")))
    else:
        configs = build_configs(args.gk, include_asym=args.asym)
    if args.configs:
        keep = {s.strip() for s in args.configs.split(",")} | {"bf16"}
        configs = [c for c in configs
                   if c.name in keep or c.name.split("-mg")[0] in keep]

    # GPTQ-calibrated weights; the W spec comes from the directory's own config.json
    # so the evaluation cannot silently disagree with how the weights were made.
    weight_dirs: dict = {}
    for entry in (args.gptq or []):
        if not entry:
            continue
        name, _, path = entry.partition("=")
        meta = json.loads((Path(path) / "config.json").read_text())
        w = QuantSpec(meta["bits"], meta["gk"], "W", sym=meta["sym"])
        a = QuantSpec(8, meta["gk"], "A")
        configs = [c for c in configs if c.name != name] + [Config(name, w, a)]
        weight_dirs[name] = path
        print(f"gptq: {name} <- {path}  ({w.label}, {meta['nsamples']} calib windows)")

    print(f"configs: {[c.name for c in configs]}  mode={args.mode}  layers={model.num_layers}")

    res = run_configs(model, ids, configs, batch_size=args.batch_size, mode=args.mode,
                      states_device=args.states_device, weight_dirs=weight_dirs)

    print(f"\n{'config':>18s}  {'PPL':>9s}  {'ΔPPL/bf16':>10s}  {'ΔPPL/w8a8':>10s}  "
          f"{'KL(nats)':>9s}  {'logitSNR':>8s}")
    base_bf16, base_w8a8 = res.ppl["bf16"], res.ppl.get("w8a8")
    for name in res.configs:
        p = res.ppl[name]
        d16 = (p / base_bf16 - 1) * 100
        d8 = (p / base_w8a8 - 1) * 100 if base_w8a8 else float("nan")
        snr, kl = res.logit_snr.get(name), res.kl.get(name)
        print(f"{name:>18s}  {p:9.4f}  {d16:9.3f}%  {d8:9.3f}%  "
              f"{(f'{kl:9.5f}' if kl is not None else '        -')}  "
              f"{(f'{snr:8.2f}' if snr is not None else '       -')}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        model=args.model, gk=args.gk, seq=args.seq, mode=args.mode,
        windows=int(ids.shape[0]), tokens=int(ids.numel()), layers=model.num_layers,
        split=args.split, seconds=res.seconds,
        ppl=res.ppl, nll=res.nll, kl=res.kl, logit_snr=res.logit_snr,
        layer_snr=res.layer_snr,
    ), indent=2))
    print(f"\nwrote {out}  ({res.seconds:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
