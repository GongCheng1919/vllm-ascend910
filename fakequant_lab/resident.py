"""A resident fake-quant model — an ordinary `Qwen2ForCausalLM` you can generate with.

The layer-streaming engine in `model_stream.py` is right for perplexity and SNR
(one weight pass, many configs) but has no KV cache, so it cannot decode.  Task
evaluation — especially generative reasoning tasks, which is where error along a
long chain of thought would show up — needs autoregressive generation.

In `dequant` mode the fake-quant model *is* a plain bf16 model:

  * the weight side is a one-off transform — replace each Linear's weight with
    its quantize/dequantize round trip, in place;
  * the activation side is an elementwise op — a forward-pre-hook that replaces
    each Linear's input with its per-token, per-group round trip.

So nothing about the module graph changes: `generate()`, KV cache, and lm_eval's
HF wrapper all work unmodified, at roughly bf16 speed.

Weights are spread over several NPUs with accelerate's `device_map` (QwQ-32B is
62 GB in bf16, so it needs at least two 64 GB cards).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .fake_linear import TARGET_LINEARS, Config
from .quant import QuantSpec, quantize


def build_device_map(num_layers: int, devices: Sequence[str]) -> Dict[str, str]:
    """Split decoder layers evenly; keep embedding/head on the first and last card."""
    n = len(devices)
    per = (num_layers + n - 1) // n
    dm = {"model.embed_tokens": devices[0], "model.rotary_emb": devices[0],
          "model.norm": devices[-1], "lm_head": devices[-1]}
    for i in range(num_layers):
        dm[f"model.layers.{i}"] = devices[min(i // per, n - 1)]
    return dm


def _act_quant_hook(a_spec: QuantSpec):
    """Replace a Linear's input with its fake-quantized round trip."""

    def hook(mod, args):
        x = args[0]
        return (quantize(x, a_spec).dequantize().to(x.dtype),) + tuple(args[1:])

    return hook


@torch.no_grad()
def load_fakequant_model(model_path: str, cfg: Config, devices: Sequence[str],
                         gptq_dir: Optional[str] = None,
                         dtype=torch.bfloat16, verbose: bool = True):
    """Load the model and apply `cfg` in place.  Returns the patched model.

    `gptq_dir` supplies calibrated weights from `run_gptq.py`; without it the
    weight side is round-to-nearest.  The `bf16` config touches nothing, so it is
    bit-identical to the unmodified checkpoint.
    """
    from transformers import AutoConfig, Qwen2ForCausalLM

    conf = AutoConfig.from_pretrained(model_path)
    dm = build_device_map(conf.num_hidden_layers, devices)
    model = Qwen2ForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm)
    model.eval()

    if cfg.is_bf16:
        if verbose:
            print("config bf16: model left untouched")
        return model

    from .run_gptq import load_layer_qtensors

    handles = []
    for li, layer in enumerate(model.model.layers):
        qts = None
        if gptq_dir:
            qts = load_layer_qtensors(
                Path(gptq_dir) / f"layer_{li:03d}.safetensors", cfg.w,
                next(layer.parameters()).device)
        for name in TARGET_LINEARS:
            mod = layer.get_submodule(name)
            qt = qts[name] if qts else quantize(mod.weight.data, cfg.w)
            mod.weight.data = qt.dequantize().to(dtype)
            handles.append(mod.register_forward_pre_hook(_act_quant_hook(cfg.a)))
            del qt
        del qts
        if verbose and (li + 1) % 16 == 0:
            print(f"  patched {li + 1}/{conf.num_hidden_layers} layers", flush=True)

    model._fq_handles = handles          # keep them alive
    if verbose:
        print(f"config {cfg.name}: W={cfg.w.label} A={cfg.a.label}"
              + (f"  weights from {gptq_dir}" if gptq_dir else "  weights RTN"))
    return model
