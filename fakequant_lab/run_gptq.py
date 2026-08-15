"""Sequential GPTQ over a streamed Qwen2 model, writing quantized weights to disk.

Standard GPTQ ordering: each layer is calibrated on activations produced by the
*already quantized* layers before it, so the error the algorithm compensates for
is the error the deployed model will actually see.

Per layer the flow is
  1. forward the calibration set once with bf16 weights, accumulating a Hessian
     per distinct Linear *input* (q/k/v share one, gate/up share one — that is
     where most of the Hessian cost would otherwise be duplicated);
  2. factorize each Hessian once, GPTQ-quantize every Linear that consumes it,
     and write the dequantized result back into the layer;
  3. forward the calibration set again, now quantized, to produce the hidden
     states the next layer will be calibrated on.

Output: one safetensors file per layer under `--out`, holding `q` (int8 codes),
`scale` and, when asymmetric, `zero` for each Linear.  Re-runs skip layers that
are already written, so a long run can be resumed.

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python -m fakequant_lab.run_gptq \\
        --gk 1024 --asym --nsamples 128 --out fakequant_lab/gptq_gk1024_asym
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

try:
    import torch_npu  # noqa: F401
    HAVE_NPU = True
except Exception:
    HAVE_NPU = False

from .data import c4_windows, wikitext2_windows
from .fake_linear import TARGET_LINEARS
from .gptq import HessianAccumulator, gptq_quantize, inverse_hessian
from .model_stream import StreamingQwen2
from .quant import QTensor, QuantSpec, fake_quant, snr_db

# Linears that share an input, and therefore a Hessian.
INPUT_GROUPS: Dict[str, List[str]] = {
    "attn_in": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "o_in": ["self_attn.o_proj"],
    "mlp_in": ["mlp.gate_proj", "mlp.up_proj"],
    "down_in": ["mlp.down_proj"],
}


class _HessianHooks:
    """Forward-pre-hooks accumulating one Hessian per distinct Linear input."""

    def __init__(self, layer, device):
        self.layer, self.device = layer, device
        self.acc: Dict[str, HessianAccumulator] = {}
        # One batch of inputs per group, kept so the driver can report the metric
        # GPTQ actually optimizes (layer output error) rather than weight error.
        self.probe: Dict[str, torch.Tensor] = {}
        self._handles: List = []

    def __enter__(self):
        for gname, names in INPUT_GROUPS.items():
            mod = self.layer.get_submodule(names[0])   # representative
            self.acc[gname] = HessianAccumulator(mod.in_features, self.device)
            mod._fq_group = gname
            self._handles.append(mod.register_forward_pre_hook(self._hook))
        return self

    def _hook(self, mod, inputs):
        g = mod._fq_group
        x = inputs[0].detach()
        self.acc[g].add(x)
        if g not in self.probe:
            self.probe[g] = x.reshape(-1, x.shape[-1]).clone()

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return None


def _layer_path(out: Path, li: int) -> Path:
    return out / f"layer_{li:03d}.safetensors"


def _save_layer(path: Path, qts: Dict[str, QTensor]) -> None:
    blob = {}
    for name, qt in qts.items():
        blob[f"{name}.q"] = qt.q.cpu()
        blob[f"{name}.scale"] = qt.scale.cpu()
        if qt.zero is not None:
            blob[f"{name}.zero"] = qt.zero.cpu()
    tmp = path.with_suffix(".tmp")
    save_file(blob, str(tmp))
    tmp.rename(path)          # atomic, so a killed run never leaves a half file


def load_layer_qtensors(path: Path, spec: QuantSpec, device) -> Dict[str, QTensor]:
    blob = load_file(str(path), device=str(device))
    out: Dict[str, QTensor] = {}
    for name in TARGET_LINEARS:
        out[name] = QTensor(blob[f"{name}.q"], blob[f"{name}.scale"],
                            blob.get(f"{name}.zero"), spec)
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/QwQ-32B")
    ap.add_argument("--gk", type=int, default=1024)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--asym", action="store_true", help="per-group weight zero point")
    ap.add_argument("--device", default="npu:0" if HAVE_NPU else "cpu")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=128, help="calibration windows")
    ap.add_argument("--calib", default="c4", choices=("c4", "wikitext2"),
                    help="calibration corpus; c4 avoids the in-domain advantage "
                         "of calibrating and scoring on the same dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--fp64-hessian", action="store_true",
                    help="factorize in float64 (CPU); slower, more robust")
    ap.add_argument("--layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "npu":
        torch.npu.set_device(dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = QuantSpec(args.bits, args.gk, "W", sym=not args.asym)

    if args.calib == "c4":
        ids = c4_windows(args.model, args.seq, args.nsamples)
    else:
        ids = wikitext2_windows(args.model, args.split, args.seq, args.nsamples)
    print(f"calibration: {tuple(ids.shape)} from {args.calib}  spec={spec.label}", flush=True)

    model = StreamingQwen2(args.model, dev)
    n_layers = args.layers or model.num_layers
    (out / "config.json").write_text(json.dumps(dict(
        model=args.model, bits=args.bits, gk=args.gk, sym=spec.sym,
        nsamples=int(ids.shape[0]), seq=args.seq, calib=args.calib, split=args.split,
        percdamp=args.percdamp, blocksize=args.blocksize, layers=n_layers), indent=2))

    states = [model.embed_ids(ids[i:i + 1]) for i in range(ids.shape[0])]
    pos = torch.arange(args.seq, device=dev).unsqueeze(0)
    pe = model.position_embeddings(states[0], pos)

    def fwd(layer, i):
        return layer(states[i], attention_mask=None, position_ids=pos,
                     position_embeddings=pe)

    t0 = time.perf_counter()
    for li in range(n_layers):
        layer = model.load_layer(li)
        path = _layer_path(out, li)

        if path.exists():
            qts = load_layer_qtensors(path, spec, dev)
            for name, qt in qts.items():
                layer.get_submodule(name).weight.data = qt.dequantize().to(model.dtype)
            for i in range(len(states)):
                states[i] = fwd(layer, i)
            print(f"  layer {li:3d}/{n_layers}  (resumed from disk) "
                  f"[{time.perf_counter() - t0:7.1f}s]", flush=True)
            del layer, qts
            continue

        # 1. collect Hessians on the un-quantized layer
        with _HessianHooks(layer, dev) as hooks:
            for i in range(len(states)):
                fwd(layer, i)
            acc, probe = hooks.acc, hooks.probe

        # 2. quantize
        qts: Dict[str, QTensor] = {}
        gain: Dict[str, float] = {}
        for gname, names in INPUT_GROUPS.items():
            Hinv, dead = inverse_hessian(acc[gname].H, args.percdamp, args.fp64_hessian)
            acc[gname].free()
            X = probe[gname]
            for name in names:
                mod = layer.get_submodule(name)
                W = mod.weight.data
                qt = gptq_quantize(W, Hinv, spec, dead, args.blocksize)
                deq = qt.dequantize()
                # GPTQ minimizes ||(W - Ŵ)X||, not ||W - Ŵ||, so weight SNR is the
                # wrong yardstick — it is *expected* to get worse.  Report the
                # layer-output SNR against RTN, which is what the algorithm buys.
                y_ref = F.linear(X, W)
                gain[name] = (snr_db(y_ref, F.linear(X, deq.to(model.dtype)))
                              - snr_db(y_ref, F.linear(X, fake_quant(W, spec).to(model.dtype))))
                mod.weight.data = deq.to(model.dtype)
                qts[name] = qt
                del deq, y_ref
            del Hinv, dead
            if dev.type == "npu":
                torch.npu.empty_cache()
        del acc, probe

        _save_layer(path, qts)

        # 3. propagate with the quantized layer
        for i in range(len(states)):
            states[i] = fwd(layer, i)

        print(f"  layer {li:3d}/{n_layers}  GPTQ-vs-RTN output SNR gain (dB) "
              + " ".join(f"{n.split('.')[-1]}={gain[n]:+5.2f}" for n in TARGET_LINEARS)
              + f"  [{time.perf_counter() - t0:7.1f}s]", flush=True)
        del layer, qts
        if dev.type == "npu":
            torch.npu.empty_cache()

    print(f"\nwrote {n_layers} layers -> {out}  ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
