"""Run several fake-quant configs over a streamed Qwen2 model in one weight pass.

All configs advance layer by layer in lockstep, so the weights are read once and
every config sees exactly the same layer objects.  The bf16 config is evaluated
first at each layer and becomes the reference for per-layer hidden-state SNR
(Tier 3 in `snr_midgroup_sweep.py`'s vocabulary).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .fake_linear import Config, PatchedLayer
from .model_stream import StreamingQwen2


@dataclass
class RunResult:
    configs: List[str]
    ppl: Dict[str, float] = field(default_factory=dict)
    nll: Dict[str, float] = field(default_factory=dict)
    # layer_snr[cfg][i] = SNR (dB) of the hidden state after layer i vs bf16.
    layer_snr: Dict[str, List[float]] = field(default_factory=dict)
    logit_snr: Dict[str, float] = field(default_factory=dict)
    # Mean KL(bf16 || config) over next-token distributions, in nats.  This is
    # the metric to gate on: logit SNR counts common-mode logit shifts that
    # softmax cancels, so it reads far worse than the model actually behaves.
    kl: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    n_tokens: int = 0


class _SnrAccum:
    """Accumulates signal/noise power so SNR is over the whole eval set, not a mean of dBs."""

    def __init__(self) -> None:
        self.sig = 0.0
        self.noise = 0.0

    def add(self, ref: torch.Tensor, test: torch.Tensor) -> None:
        r = ref.float()
        d = test.float() - r
        self.sig += torch.sum(r * r).item()
        self.noise += torch.sum(d * d).item()

    @property
    def db(self) -> float:
        if self.noise == 0.0:
            return float("inf")
        if self.sig == 0.0:
            return float("-inf")
        return 10.0 * torch.log10(torch.tensor(self.sig / self.noise)).item()


def _head_metrics(model: StreamingQwen2, hidden: torch.Tensor, targets: torch.Tensor,
                  ref_hidden: Optional[torch.Tensor], acc: Dict[str, "_SnrAccum"],
                  key: str, chunk: int = 512) -> tuple[float, float, int]:
    """Sum of token NLLs, sum of KL(ref || this), and the token count.

    The vocab is 152k wide, so materializing logits for a full 2048-token window
    costs ~1.2 GB in fp32; slicing keeps that bounded.  When `ref_hidden` is
    given, the bf16 logits are recomputed per slice so both live at once only
    within a slice.
    """
    nll_sum, kl_sum, count = 0.0, 0.0, 0
    S = hidden.shape[1]
    for s in range(0, S, chunk):
        e = min(s + chunk, S)
        logits = model.head(hidden[:, s:e]).float()
        tgt = targets[:, s:e]
        if bool((tgt >= 0).any()):
            nll_sum += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       tgt.reshape(-1), ignore_index=-100,
                                       reduction="sum").item()
            count += int((tgt >= 0).sum())
        if ref_hidden is not None:
            ref = model.head(ref_hidden[:, s:e]).float()
            acc[key].add(ref, logits)
            # KL(P_bf16 || P_cfg), summed over tokens.
            kl_sum += F.kl_div(F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1),
                               log_target=True, reduction="sum").item()
            del ref
        del logits
    return nll_sum, kl_sum, count


@torch.no_grad()
def run_configs(
    model: StreamingQwen2,
    input_ids: torch.Tensor,                 # [nwin, S] int64
    configs: Sequence[Config],
    *,
    batch_size: int = 1,
    mode: str = "dequant",
    states_device: Optional[torch.device] = None,
    compute_ppl: bool = True,
    verbose: bool = True,
    weight_dirs: Optional[Dict[str, str]] = None,
) -> RunResult:
    """`weight_dirs` maps a config name to a `run_gptq.py` output directory; those
    configs then use the calibrated weights instead of round-to-nearest."""
    dev = model.device
    states_device = torch.device(states_device) if states_device is not None else dev
    names = [c.name for c in configs]
    assert names[0] == "bf16", "bf16 must be first: it is the SNR reference"

    nwin, S = input_ids.shape
    batches = [input_ids[i:i + batch_size] for i in range(0, nwin, batch_size)]
    pos_ids = torch.arange(S, device=dev).unsqueeze(0)

    # Hidden states for every config, kept as one tensor per batch.
    states: Dict[str, List[torch.Tensor]] = {
        c.name: [model.embed_ids(b).to(states_device) for b in batches] for c in configs
    }
    pe_cache = model.position_embeddings(
        torch.empty(1, S, model.config.hidden_size, device=dev, dtype=model.dtype), pos_ids)

    res = RunResult(configs=list(names))
    res.layer_snr = {n: [] for n in names if n != "bf16"}
    t0 = time.perf_counter()

    for li in range(model.num_layers):
        layer = model.load_layer(li)
        accum = {n: _SnrAccum() for n in names if n != "bf16"}
        for cfg in configs:
            w_qts = None
            if weight_dirs and cfg.name in weight_dirs:
                from pathlib import Path

                from .run_gptq import load_layer_qtensors
                w_qts = load_layer_qtensors(
                    Path(weight_dirs[cfg.name]) / f"layer_{li:03d}.safetensors",
                    cfg.w, dev)
            with PatchedLayer(layer, cfg, mode=mode, w_qts=w_qts):
                for bi in range(len(batches)):
                    h = states[cfg.name][bi].to(dev, non_blocking=True)
                    out = layer(h, attention_mask=None, position_ids=pos_ids,
                                position_embeddings=pe_cache)
                    if cfg.name != "bf16":
                        accum[cfg.name].add(states["bf16"][bi].to(dev), out)
                    states[cfg.name][bi] = out.to(states_device)
                    del h, out
        for n in accum:
            res.layer_snr[n].append(accum[n].db)
        del layer, w_qts
        if dev.type == "npu":
            torch.npu.empty_cache()
        if verbose:
            worst = min((res.layer_snr[n][-1] for n in accum), default=float("nan"))
            print(f"  layer {li:3d}/{model.num_layers}  "
                  + "  ".join(f"{n}={res.layer_snr[n][-1]:6.2f}dB" for n in accum)
                  + f"   [{time.perf_counter() - t0:6.1f}s]", flush=True)

    if compute_ppl:
        logit_acc = {n: _SnrAccum() for n in names if n != "bf16"}
        for cfg in configs:
            is_ref = cfg.name == "bf16"
            total, kl_total, count = 0.0, 0.0, 0
            for bi, b in enumerate(batches):
                h = states[cfg.name][bi].to(dev)
                # Next-token targets; the last position of a window has no target.
                tgt = torch.full_like(b, -100)
                tgt[:, :-1] = b[:, 1:]
                ref_h = None if is_ref else states["bf16"][bi].to(dev)
                t, k, c = _head_metrics(model, h, tgt.to(dev), ref_h, logit_acc, cfg.name)
                total += t
                kl_total += k
                count += c
                del h, ref_h
            res.nll[cfg.name] = total / max(count, 1)
            res.ppl[cfg.name] = float(torch.exp(torch.tensor(res.nll[cfg.name])))
            if not is_ref:
                res.kl[cfg.name] = kl_total / max(nwin * S, 1)
            res.n_tokens = count
        res.logit_snr = {n: a.db for n, a in logit_acc.items()}

    res.seconds = time.perf_counter() - t0
    return res
