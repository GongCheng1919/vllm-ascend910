"""Fake-quant `nn.Linear` replacement + the five P1 comparison configs.

Two modes, differing only in how faithfully the arithmetic is reproduced.  The
*quantization algorithm* — bit width, group boundaries, bf16 scales, zero point —
is identical in both; that is what determines model quality.

`dequant` (default) — ordinary fake quant, and what P2 evaluates with:

    y = bf16_matmul( deq(A_q, as).to(bf16), deq(W_q, ws).to(bf16) )

  Computationally a plain bf16 model (measured 1.1-2.5x bf16), so it decodes at
  normal speed and can be used for generative task evaluation.

`kernel` — reproduces the AscendC kernel's arithmetic exactly:

    for each K-group g:
        partial = A_q[:, g] @ W_q[:, g].T         # exact integer sum
        acc    += partial * as[:, g] * ws[:, g]   # fp32
    y = acc.to(bf16)

  `partial` uses an fp32 matmul: the operands are small integers (|A_q| <= 127,
  |W_q| <= 8) exact in fp32 and in the cube's hf32 input format, and
  |partial| <= GK*127*8 ~ 1e6 < 2^24, so the fp32 accumulator reproduces the
  cube's int32 sum.  `run_selftest.py` T4 checks that against a CPU int64
  reference.  Costs 8-19x bf16 (G matmuls per Linear), so it is a verification
  tool, not the default.

The only difference `dequant` introduces is rounding the dequantized operands to
bf16, which the kernel never does — it keeps int8 codes and applies the scales
after the integer sum.  `run_selftest.py` T6 measures that at ~27 dB below the
quantization noise itself, i.e. ~0.2% of the total noise power.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import QTensor, QuantSpec, quantize, w_ksum

# Older runs and scripts used these names; keep them working.
MODE_ALIASES = {"fast": "dequant", "exact": "kernel"}

# Every Linear the kernel will own.  Attention QK/PV matmuls, norms, embedding
# and lm_head stay bf16 (`MID_GROUP_ROADMAP.md` §5 "覆盖范围").
TARGET_LINEARS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


@dataclass(frozen=True)
class Config:
    """One row of the P1 comparison table (`MID_GROUP_ROADMAP.md` §5)."""

    name: str
    w: Optional[QuantSpec]      # None for the bf16 control
    a: Optional[QuantSpec]

    @property
    def is_bf16(self) -> bool:
        return self.w is None


def build_configs(gk: int, include_asym: bool = False) -> List[Config]:
    """The five configs at one group size, exactly as frozen in the roadmap §5.

    `w8a8-mg` isolates "mid-group itself" from "W down to 4 bit"; without it the
    two variables are tangled on the same curve.  `w4a8-pc` is the config the
    already-implemented per-channel kernel corresponds to (lower-bound
    reference).

    `include_asym` adds a sixth, *non-frozen* config: int4 weights with a
    per-group zero point.  It is off by default so the default run is the
    frozen spec; P1 measured it to be worth ~1.4 dB of weight SNR, which is
    what shrinking GK by 4x buys — at a far lower kernel cost.  See
    `reports/midgroup/P1_harness.md` §4.
    """
    cfgs = [
        Config("bf16", None, None),
        Config("w8a8", QuantSpec(8, None, "W"), QuantSpec(8, None, "A")),
        Config(f"w8a8-mg{gk}", QuantSpec(8, gk, "W"), QuantSpec(8, gk, "A")),
        Config(f"w4a8-mg{gk}", QuantSpec(4, gk, "W"), QuantSpec(8, gk, "A")),
        Config("w4a8-pc", QuantSpec(4, None, "W"), QuantSpec(8, None, "A")),
    ]
    if include_asym:
        cfgs.append(Config(f"w4a8-mg{gk}-asym",
                           QuantSpec(4, gk, "W", sym=False), QuantSpec(8, gk, "A")))
    return cfgs


def config_by_name(name: str, gk: int) -> Config:
    for c in build_configs(gk):
        if c.name == name or c.name.startswith(name + "-mg"):
            return c
    raise KeyError(f"unknown config {name!r} (have {[c.name for c in build_configs(gk)]})")


class FakeQuantLinear(nn.Module):
    """Drop-in replacement for an `nn.Linear` inside a decoder layer.

    Holds the quantized weight for *one* config.  The bf16 config is handled by
    simply not wrapping the Linear at all, which is what makes the bf16 control
    bit-exact with the unpatched model.
    """

    def __init__(self, base: nn.Linear, cfg: Config, mode: str = "dequant",
                 w_qt: Optional[QTensor] = None):
        super().__init__()
        assert not cfg.is_bf16, "bf16 config must not be wrapped"
        mode = MODE_ALIASES.get(mode, mode)
        assert mode in ("dequant", "kernel"), mode
        assert cfg.w.gk == cfg.a.gk, (
            f"A and W must share the group boundary (got W {cfg.w.gk} / A {cfg.a.gk}); "
            "the kernel has a single flush point per group"
        )
        self.cfg = cfg
        self.mode = mode
        self.bias = base.bias
        self.out_features = base.out_features
        self.in_features = base.in_features

        # `w_qt` lets a calibrated quantizer (GPTQ) supply the weight instead of
        # round-to-nearest; everything downstream is identical.
        qw = w_qt if w_qt is not None else quantize(base.weight.data, cfg.w)
        self.register_buffer("q_w", qw.q, persistent=False)              # [N, K] int8
        self.register_buffer("w_scale", qw.scale, persistent=False)      # [N, G] bf16
        self.register_buffer("w_zero", qw.zero, persistent=False)        # [N, G] bf16 or None
        if mode == "dequant":
            # Pre-dequantize once; the per-forward cost then matches plain bf16.
            self.register_buffer("w_deq", qw.dequantize().to(torch.bfloat16), persistent=False)

    # -- introspection used by the exporters -------------------------------
    def w_ksum(self) -> torch.Tensor:
        """`KS[n,g]` for the MSD correction term; see `quant.w_ksum`."""
        return w_ksum(self.q_w, self.cfg.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading, K = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, K)

        if self.mode == "dequant":
            y = F.linear(quantize(x2, self.cfg.a).dequantize().to(torch.bfloat16), self.w_deq)
        else:
            y = self._kernel_gemm(x2).to(x.dtype)

        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*leading, self.out_features)

    def _kernel_gemm(self, x2: torch.Tensor) -> torch.Tensor:
        """Group-wise exact-integer GEMM with fp32 rescale, as the kernel does.

        With an asymmetric weight (`spec.sym=False`) the per-group zero point
        contributes a rank-1 term:

            Σ_{k∈g} A_q·(W_q - wz) = Σ_{k∈g} A_q·W_q  -  wz[n,g] · Σ_{k∈g} A_q[m,k]

        i.e. one extra per-group activation row-sum, which the per-token
        quantize kernel can emit in the same pass that writes the int4 planes.
        """
        M, K = x2.shape
        qa = quantize(x2, self.cfg.a)                     # [M,K] int8, [M,G] bf16
        G, gl = self.cfg.a.groups(K), self.cfg.a.group_len(K)

        acc = torch.zeros(M, self.out_features, dtype=torch.float32, device=x2.device)
        a_s32, w_s32 = qa.scale.float(), self.w_scale.float()
        for g in range(G):
            sl = slice(g * gl, (g + 1) * gl)
            a_g = qa.q[:, sl].float()
            partial = F.linear(a_g, self.q_w[:, sl].float())                  # exact
            if self.w_zero is not None:
                partial -= a_g.sum(dim=1, keepdim=True) * self.w_zero[:, g].float().unsqueeze(0)
            acc.addcmul_(partial, a_s32[:, g : g + 1] * w_s32[:, g].unsqueeze(0))
        return acc


# ── Patching a decoder layer ────────────────────────────────────────────────

def _get_parent(layer: nn.Module, dotted: str):
    parts = dotted.split(".")
    obj = layer
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


class PatchedLayer:
    """Context manager: swap a decoder layer's Linears for one config, restore after.

    Usage:
        with PatchedLayer(layer, cfg) as _:
            out = layer(hidden, ...)

    For the bf16 config this is a no-op, so the control path runs the original
    modules — that is the Exit-Criteria "插桩无副作用" guarantee, checked in
    `run_selftest.py`.
    """

    def __init__(self, layer: nn.Module, cfg: Config, mode: str = "dequant",
                 names: tuple = TARGET_LINEARS,
                 w_qts: Optional[Dict[str, QTensor]] = None):
        self.layer, self.cfg, self.mode, self.names = layer, cfg, mode, names
        self.w_qts = w_qts or {}
        self._saved: Dict[str, nn.Module] = {}
        self.wrapped: Dict[str, FakeQuantLinear] = {}

    def __enter__(self) -> "PatchedLayer":
        if self.cfg.is_bf16:
            return self
        for name in self.names:
            parent, attr = _get_parent(self.layer, name)
            base = getattr(parent, attr)
            self._saved[name] = base
            fq = FakeQuantLinear(base, self.cfg, self.mode, self.w_qts.get(name))
            self.wrapped[name] = fq
            setattr(parent, attr, fq)
        return self

    def __exit__(self, *exc) -> None:
        for name, base in self._saved.items():
            parent, attr = _get_parent(self.layer, name)
            setattr(parent, attr, base)
        self._saved.clear()
        self.wrapped.clear()
        return None
