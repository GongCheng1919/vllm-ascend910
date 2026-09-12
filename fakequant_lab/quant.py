"""Mid-group / per-channel symmetric fake quantization primitives.

Implements the quantization scheme frozen in `MID_GROUP_ROADMAP.md` §1/§5:

    y[m,n] = Σ_g  as[m,g] · ws[n,g] · ( Σ_{k∈g} A_q[m,k] · W_q[n,k] )

    A_q: [M,K] int8,  as: [M, K/GK] bf16     (per-token   × per-group, dynamic)
    W_q: [N,K] int4,  ws: [N, K/GK] bf16     (per-channel × per-group, static)

Two rules that are easy to get wrong and that this module enforces:

1.  **Scales are bf16, not fp32.**  The kernel stores and multiplies bf16
    scales; a fake-quant that keeps fp32 scales systematically over-estimates
    accuracy.  Every scale produced here is rounded to bf16 *before* it is used
    to quantize, so the simulated tensor is exactly what the kernel would see.

2.  **MSD splitting is not modelled.**  `A_q = 16·h + l + 8` is exact inside
    int32 (`MID_GROUP_ROADMAP.md` §1), so it cannot change any number here.
    `msd_split` is provided only so P3/P4 can byte-check the packing.

Group axis is always the last dim (K).  `GK = None` means per-channel, i.e. a
single group spanning the whole row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# int4 uses [-8, 7] but the scale is max|w|/7 (symmetric), matching the roadmap.
QMAX = {4: 7, 8: 127}
QMIN = {4: -8, 8: -128}


@dataclass(frozen=True)
class QuantSpec:
    """How one operand (W or A) is quantized."""

    bits: int                    # 4 or 8
    gk: Optional[int]            # group size along K; None = per-channel
    name: str = ""
    sym: bool = True             # False = asymmetric (per-group zero point)

    def __post_init__(self) -> None:
        assert self.bits in QMAX, f"unsupported bits={self.bits}"
        assert self.gk is None or self.gk > 0, f"bad gk={self.gk}"

    def groups(self, K: int) -> int:
        if self.gk is None:
            return 1
        assert K % self.gk == 0, f"K={K} not divisible by gk={self.gk}"
        return K // self.gk

    def group_len(self, K: int) -> int:
        return K if self.gk is None else self.gk

    @property
    def label(self) -> str:
        g = "pc" if self.gk is None else f"g{self.gk}"
        return f"int{self.bits}-{g}" + ("" if self.sym else "-asym")


@dataclass
class QTensor:
    """A quantized tensor: `x ≈ scale · (q - zero)`, grouped along the last dim."""

    q: torch.Tensor              # [..., K] int8, codes in [QMIN, QMAX]
    scale: torch.Tensor          # [..., G] bf16
    zero: Optional[torch.Tensor] # [..., G] bf16, in code units; None if symmetric
    spec: QuantSpec

    def dequantize(self) -> torch.Tensor:
        K = self.q.shape[-1]
        G, gl = self.spec.groups(K), self.spec.group_len(K)
        qg = self.q.float().reshape(*self.q.shape[:-1], G, gl)
        if self.zero is not None:
            qg = qg - self.zero.float().unsqueeze(-1)
        return (qg * self.scale.float().unsqueeze(-1)).reshape(*self.q.shape)


def _tiny_where_zero(scale: torch.Tensor) -> torch.Tensor:
    """A zero group would give scale 0; use the smallest positive normal bf16 so
    codes stay 0 and dequant stays 0 instead of producing NaN."""
    return torch.where(scale == 0, torch.full_like(scale, torch.finfo(torch.bfloat16).tiny), scale)


def group_params(xg: torch.Tensor, bits: int, sym: bool
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Quantization parameters for one group, from its values.

    `xg` is `[..., gl]` — the members of a single group along the last dim.
    Returns `(scale, zero)` broadcastable as `[..., 1]`; `zero is None` when
    symmetric.  **The scale is rounded to bf16 here**, before anything is
    quantized with it, so every caller inherits that rule (the kernel stores
    bf16 scales; quantizing with an fp32 scale would over-estimate accuracy).

    GPTQ calls this per group on the *error-compensated* weights, which is why
    it is factored out of `quantize`.
    """
    qmax, qmin = QMAX[bits], QMIN[bits]
    if sym:
        scale = _tiny_where_zero((xg.abs().amax(dim=-1, keepdim=True) / qmax).to(torch.bfloat16))
        return scale, None
    mx = xg.amax(dim=-1, keepdim=True)
    mn = xg.amin(dim=-1, keepdim=True)
    rng = mx - mn
    scale = _tiny_where_zero((rng / (qmax - qmin)).to(torch.bfloat16))
    # Put the code range at [qmin, qmax] so int4 still stores as signed
    # nibbles: x ~ scale*(q - zero), zero = qmin - round(mn/scale).
    zero = (float(qmin) - torch.round(mn / scale.float())).to(torch.bfloat16)

    # CONSTANT group (mx == mn): there is no range to spread, and the formula
    # above degenerates catastrophically -- `_tiny_where_zero` floors the scale at
    # bf16-tiny but then `round(mn/tiny)` makes |zero| ~1e35, which stays finite in
    # bf16 and only explodes later, in the kernel's rank-1 term `wz * a_ksum`
    # (|a_ksum| ~1e5 => inf in fp32).  The guard was half-applied: it protected the
    # scale and left the zero point to blow up.
    #
    # For a constant group `scale = |v|/qmax, zero = 0` is EXACT: q = round(v/scale)
    # = +/-qmax and dequant returns v.  Only reachable when a group is constant,
    # which for a 1024-element weight group never happens in practice -- but a
    # RAGGED last group can hold a single element, where it always happens.  So
    # this changes nothing P2 measured and fixes every ragged K whose last group
    # has one member (K=1025, K=5121, ...).
    const = (rng == 0)
    scale = torch.where(const,
                        _tiny_where_zero((mx.abs() / qmax).to(torch.bfloat16)),
                        scale)
    zero = torch.where(const, torch.zeros_like(zero), zero)
    return scale, zero


def quantize_with(x: torch.Tensor, scale: torch.Tensor, zero: Optional[torch.Tensor],
                  bits: int) -> torch.Tensor:
    """Apply given `(scale, zero)` to `x`, returning int8 codes."""
    q = torch.round(x / scale.float())
    if zero is not None:
        q = q + zero.float()
    return q.clamp_(QMIN[bits], QMAX[bits]).to(torch.int8)


def dequantize_with(q: torch.Tensor, scale: torch.Tensor,
                    zero: Optional[torch.Tensor]) -> torch.Tensor:
    """Inverse of `quantize_with`, in fp32."""
    qf = q.float()
    if zero is not None:
        qf = qf - zero.float()
    return qf * scale.float()


def quantize(x: torch.Tensor, spec: QuantSpec) -> QTensor:
    """Fake quantization along the last dim (round-to-nearest).

    Symmetric (`spec.sym`, the frozen default):  scale = max|x| / QMAX.
    Asymmetric: scale = (max - min) / (2^bits - 1) with a per-group zero point,
    which the kernel can absorb as a rank-1 correction — see `fake_linear`.
    """
    K = x.shape[-1]
    G, gl = spec.groups(K), spec.group_len(K)
    xg = x.float().reshape(*x.shape[:-1], G, gl)

    scale, zero = group_params(xg, spec.bits, spec.sym)
    q = quantize_with(xg, scale, zero, spec.bits)

    shape_g = (*x.shape[:-1], G)
    return QTensor(
        q.reshape(*x.shape),
        scale.squeeze(-1).reshape(shape_g),
        None if zero is None else zero.squeeze(-1).reshape(shape_g),
        spec,
    )


def fake_quant(x: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Quantize then dequantize, returning fp32.  Convenience for Tier-1 SNR."""
    return quantize(x, spec).dequantize()


def w_ksum(q_w: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Per-group weight sums `KS[n,g] = Σ_{k∈g} W_q[n,k]`, int32 [N, G].

    Needed by the MSD correction term `8·KS` in the kernel
    (`REPORT_W4A8.md` §1).  Weight-side one-off preprocessing; produced here so
    P3/P4 can ship it alongside the quantized weights.
    """
    K = q_w.shape[-1]
    G = spec.groups(K)
    gl = spec.group_len(K)
    # torch.sum promotes integral types to int64; the kernel wants int32.
    return q_w.to(torch.int32).reshape(*q_w.shape[:-1], G, gl).sum(dim=-1).to(torch.int32)


def to_group_major(t: torch.Tensor) -> torch.Tensor:
    """`[*, G] -> [G, *]`, the layout the kernel reads (`P0_COST.md` D4).

    group-major keeps each group's TILE_N/TILE_M scales contiguous in GM so the
    kernel gets them with one DataCopy.  Applies to `ws`, `as` and `w_ksum`.
    """
    return t.transpose(-1, -2).contiguous()


def msd_split(q_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split int8 activations into the two int4 planes the cube consumes.

        a = 16*h + l + 8,   h = a >> 4,   l = (a & 15) - 8,   both in [-8, 7]

    Exact for all 256 int8 values (`REPORT_W4A8.md` §1).  `l` is *stored* as the
    nibble `(a & 15) ^ 8`, which the cube reads back as the signed int4 value
    `(a & 15) - 8`; this returns the value, so P3 packing code must still apply
    the `^ 8` when it writes nibbles.  Numerically a no-op, so the fake-quant
    path never calls this; it exists for P3/P4 byte checks.
    """
    a = q_a.to(torch.int32)
    hi = a >> 4                                   # arithmetic shift -> [-8, 7]
    lo = (a & 15) - 8                             # signed value of the ^8 nibble
    return hi.to(torch.int8), lo.to(torch.int8)


def msd_recombine(hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    """Inverse of `msd_split`: `a = 16*h + l + 8`."""
    return (16 * hi.to(torch.int32) + lo.to(torch.int32) + 8).to(torch.int8)


def snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB.  `inf` when the two are bit-identical."""
    r = ref.float()
    d = test.float() - r
    p_sig = torch.sum(r * r).item()
    p_noise = torch.sum(d * d).item()
    if p_noise == 0.0:
        return float("inf")
    if p_sig == 0.0:
        return float("-inf")
    return 10.0 * torch.log10(torch.tensor(p_sig / p_noise)).item()
