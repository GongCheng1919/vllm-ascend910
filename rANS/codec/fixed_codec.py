# Copyright (c) 2026. Fixed-rate lossless BF16 weight codec (encoder side).
"""Encoder for the codec that ``csrc/kernels/ans_fixed_decode.cpp`` decodes.

Replaces the rANS path of :mod:`tools.ans.rans` for weights that are decoded on
the NPU. rANS reaches ~10.6 bits/weight but needs one entropy-coder state
transition per weight, which measures at ~1.6 cycles/weight on the AIV -- four
times the budget set by simply reading BF16 from HBM. This codec gives up 1.7
bits/weight and decodes in 0.44, because it is shifts and ors only: no state,
no table gather, no serial dependency.

Why it is shaped this way (all measured, see tools/ans/bitplane_scan.py):

* ``H(mantissa) = 6.973`` of 7 bits and it is independent of the exponent, so
  the mantissa is stored raw. Entropy-coding it is pure waste.
* ``H(sign, exp) = 3.641`` of 9 bits, and exponents span only ~24 values per
  tensor, with a narrow local range. So the exponent gets a 4-bit offset from a
  per-group base.
* The base is two-level -- a super base per 256 weights plus a 4-bit delta per
  16 -- because one uint16 base per 16 weights would cost a whole bit per
  weight, a quarter of the exponent field, for no gain in the wide-group rate
  (13.00 vs 12.33 bits/weight, both 0.062% wide).

Layout, per BLOCK of 16384 weights, with block-local index ``q`` split into
quarter ``r = q // 4096`` and slot ``j = q % 4096``::

    lplane[(r // 2) * 4096 + j]  byte (r & 1)   sign<<7 | mantissa
    eplane[j]                    nibble r       exponent - group base
    dplane[j']                   nibble r'      group base - super base
    sbase[s]                                    min exponent of super group s

Extracting one nibble or byte yields 4096 values that land in a CONTIGUOUS run
of the output, so the decoder needs no transpose and no strided store. That is
what the interleaving buys; it is not an arbitrary permutation.

A group whose exponent spread still will not fit 4 bits moves to a wide list
carrying raw L and exponent bytes, decoded by a second vectorised pass. On real
weights that is 0.06% of groups.

Wide groups are NOT free, and this is the one place the scheme can lose: a wide
group pays its narrow-path bits (which are then ignored) *plus* its raw bytes.
A tensor whose exponents span the full float range measures at 46 bits/weight --
almost three times worse than not compressing at all. So the encoder refuses:
:func:`encode_fixed` returns ``None`` when the payload would not beat plain
BF16, and the caller leaves that tensor uncompressed. That bounds the worst case
at exactly 16 bits/weight and is what makes the scheme safe on an unseen model,
rather than any claim that it "degrades gracefully".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Must match csrc/kernels/ans_fixed_decode.cpp exactly. The block is the unit
# the decoder interleaves within: extracting one nibble of the E plane yields
# block/4 values that land in a contiguous output run.
#
# It is a parameter because the fused decode+GEMM kernel needs the block to be
# exactly one Cube tile. Two AIVs share a Cube, and a block's four quarters all
# live in the SAME E-plane words -- so if the two AIVs split a tile by rows they
# both read the whole E plane, doubling the payload read from 4 to 8 bits per
# weight and cancelling the entire compression gain. Splitting by k-tile instead
# keeps each AIV on its own planes, and that only works if one tile is one block.
ANS_BLOCK = 16384
ANS_QUARTER = ANS_BLOCK // 4
ANS_GROUP = 16
ANS_SUPER = 256
ANS_GROUPS_PER_BLOCK = ANS_BLOCK // ANS_GROUP
ANS_DQUARTER = ANS_GROUPS_PER_BLOCK // 4
ANS_OFF_MAX = 15
ANS_DELTA_MAX = 15


@dataclass
class FixedEncoded:
    """Payload planes, all little-endian uint16 except ``wide_group``."""

    lplane: np.ndarray
    eplane: np.ndarray
    dplane: np.ndarray
    sbase: np.ndarray
    wide_group: np.ndarray
    wide_l: np.ndarray
    wide_exp: np.ndarray
    n_weights: int = 0
    n_groups: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def payload_bytes(self) -> int:
        return int(
            self.lplane.nbytes + self.eplane.nbytes + self.dplane.nbytes
            + self.sbase.nbytes + self.wide_group.nbytes
            + self.wide_l.nbytes + self.wide_exp.nbytes
        )

    @property
    def bits_per_weight(self) -> float:
        return self.payload_bytes * 8.0 / float(self.n_weights)


def _split_fields(bf16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(exponent byte, sign|mantissa byte) of raw BF16 bit patterns."""
    u = np.asarray(bf16, dtype=np.uint16).ravel()
    exp = ((u >> 7) & 0xFF).astype(np.uint8)
    lo = (((u >> 8) & 0x80) | (u & 0x7F)).astype(np.uint8)
    return exp, lo


def pad_to_block(bf16: np.ndarray, block: int = ANS_BLOCK) -> tuple[np.ndarray, int]:
    """Zero-pad to a whole number of blocks; returns (padded, original size)."""
    u = np.asarray(bf16, dtype=np.uint16).ravel()
    n = int(u.size)
    rem = n % block
    if rem:
        u = np.concatenate([u, np.zeros(block - rem, dtype=np.uint16)])
    return u, n


def _check_block(block: int) -> None:
    if block % ANS_SUPER or (block // 4) % 16 or (block // ANS_GROUP) % 4:
        raise ValueError(f"block {block} must be a multiple of {ANS_SUPER} and keep "
                         "quarter/nibble packing aligned")


# Refuse to encode above this share of payload bits versus plain BF16. Real
# projection weights land at 0.771; anything near 1.0 is a tensor this codec is
# the wrong tool for.
MAX_BITS_PER_WEIGHT = 15.0


def encode_fixed(bf16: np.ndarray, max_bits: float = MAX_BITS_PER_WEIGHT,
                 block: int = ANS_BLOCK) -> FixedEncoded | None:
    """Encode raw BF16 bit patterns; size must be a multiple of ``block``.

    Returns ``None`` if the result would not comfortably beat plain BF16, in
    which case the caller should store the tensor uncompressed. Pass
    ``max_bits=64`` to force encoding (tests, measurement).
    """
    _check_block(block)
    u = np.asarray(bf16, dtype=np.uint16).ravel()
    n = int(u.size)
    if n == 0 or n % block:
        raise ValueError(f"n_weights {n} must be a non-zero multiple of {block}")
    n_blocks = n // block
    n_groups = n // ANS_GROUP
    n_supers = n // ANS_SUPER

    exp, lo = _split_fields(u)

    # --- two-level base ------------------------------------------------
    sup_min = exp.reshape(n_supers, ANS_SUPER).min(axis=1).astype(np.int16)
    grp = exp.reshape(n_groups, ANS_GROUP)
    grp_min = grp.min(axis=1).astype(np.int16)
    grp_max = grp.max(axis=1).astype(np.int16)
    sup_of_grp = np.repeat(sup_min, ANS_SUPER // ANS_GROUP)
    delta = np.minimum(grp_min - sup_of_grp, ANS_DELTA_MAX).astype(np.uint8)
    gbase = (sup_of_grp + delta).astype(np.uint8)

    # A group the 4-bit offset cannot cover goes to the wide list.
    is_wide = (grp_max - gbase.astype(np.int16)) > ANS_OFF_MAX
    wide_idx = np.flatnonzero(is_wide).astype(np.int32)

    off = (exp.astype(np.int16) - np.repeat(gbase, ANS_GROUP).astype(np.int16))
    # Wide groups' narrow-path bits are don't-care; the second pass overwrites
    # those outputs. Zero them so the payload is deterministic.
    off = np.where(np.repeat(is_wide, ANS_GROUP), 0, off).astype(np.uint16) & 0xF

    # --- interleaved planes --------------------------------------------
    quarter = block // 4
    dquarter = block // ANS_GROUP // 4
    lo_b = lo.reshape(n_blocks, 4, quarter).astype(np.uint16)
    lplane = (lo_b[:, 0::2, :] | (lo_b[:, 1::2, :] << 8)).reshape(n_blocks, 2 * quarter)
    lplane = np.ascontiguousarray(lplane.reshape(-1), dtype=np.uint16)

    off_b = off.reshape(n_blocks, 4, quarter)
    eplane = np.zeros((n_blocks, quarter), dtype=np.uint16)
    for r in range(4):
        eplane |= off_b[:, r, :] << (4 * r)
    eplane = np.ascontiguousarray(eplane.reshape(-1))

    delta_b = delta.reshape(n_blocks, 4, dquarter).astype(np.uint16)
    dplane = np.zeros((n_blocks, dquarter), dtype=np.uint16)
    for r in range(4):
        dplane |= (delta_b[:, r, :] & 0xF) << (4 * r)
    dplane = np.ascontiguousarray(dplane.reshape(-1))

    # --- wide list ------------------------------------------------------
    if wide_idx.size:
        rows = wide_idx.astype(np.int64)[:, None] * ANS_GROUP + np.arange(ANS_GROUP)
        wide_l = lo[rows].astype(np.uint16).reshape(-1)
        wide_exp = exp[rows].astype(np.uint16).reshape(-1)
    else:
        wide_l = np.zeros(0, dtype=np.uint16)
        wide_exp = np.zeros(0, dtype=np.uint16)

    enc = FixedEncoded(
        lplane=lplane,
        eplane=eplane,
        dplane=dplane,
        sbase=sup_min.astype(np.uint16),
        wide_group=wide_idx,
        wide_l=wide_l,
        wide_exp=wide_exp,
        n_weights=n,
        n_groups=n_groups,
        meta={
            "block": block,
            "group": ANS_GROUP,
            "super": ANS_SUPER,
            "off_bits": 4,
            "n_wide": int(wide_idx.size),
            "wide_frac": float(wide_idx.size) / float(n_groups),
        },
    )
    return enc if enc.bits_per_weight <= max_bits else None


# Cube-aligned 2D tiling for the fused decode+GEMM kernel. n0 * k0 == ANS_BLOCK
# is deliberate: one encoder block IS one Cube tile, so tile t's planes sit at
# t * (ANS_BLOCK/2), t * (ANS_BLOCK/4), ... with no offset table at all. That is
# the structural win of a fixed-rate code over rANS here -- the rANS path needs
# per-tile byte offsets, symbol counts and a codebook, all of which the kernel
# must chase before it can decode anything.
GEMM_N0 = 64
GEMM_K0 = 128
GEMM_BLOCK = GEMM_N0 * GEMM_K0          # 8192, one Cube tile == one encoder block


def pad_gemm_weight(codes: np.ndarray, n_out: int, n_in: int,
                    n0: int = GEMM_N0, k0: int = GEMM_K0) -> tuple[np.ndarray, int, int]:
    """Pad W to multiples of (n0, k0) with zeros, which are valid BF16."""
    w = np.asarray(codes, dtype=np.uint16).reshape(n_out, n_in)
    n_pad = (n_out + n0 - 1) // n0 * n0
    k_pad = (n_in + k0 - 1) // k0 * k0
    if n_pad == n_out and k_pad == n_in:
        return w, n_pad, k_pad
    out = np.zeros((n_pad, k_pad), dtype=np.uint16)
    out[:n_out, :n_in] = w
    return out, n_pad, k_pad


def tile_gemm_weight(w_pad: np.ndarray, n0: int = GEMM_N0, k0: int = GEMM_K0) -> np.ndarray:
    """(n_pad, k_pad) -> flat tile-major order, tile (ni, kj) row-major inside.

    Tile order matches the kernel's loop nest: ni outer, kj inner. Row-major
    inside a tile is what ScatterTile expects to copy into the GM panel.
    """
    n_pad, k_pad = w_pad.shape
    return (w_pad.reshape(n_pad // n0, n0, k_pad // k0, k0)
                 .transpose(0, 2, 1, 3)
                 .reshape(-1))


def encode_fixed_gemm(codes: np.ndarray, n_out: int, n_in: int,
                      n0: int = GEMM_N0, k0: int = GEMM_K0,
                      max_bits: float = MAX_BITS_PER_WEIGHT) -> FixedEncoded | None:
    """Encode W in Cube tiles. ``meta`` carries the shape the kernel needs.

    Adds ``wide_tile_offsets``: wide groups are already sorted by group index
    and groups never straddle a tile, so a prefix index per tile lets the kernel
    find its own wide groups without scanning the whole list.
    """
    block = n0 * k0
    w_pad, n_pad, k_pad = pad_gemm_weight(codes, n_out, n_in, n0, k0)
    enc = encode_fixed(tile_gemm_weight(w_pad, n0, k0), max_bits=max_bits, block=block)
    if enc is None:
        return None

    tiles_n, tiles_k = n_pad // n0, k_pad // k0
    n_tiles = tiles_n * tiles_k
    groups_per_tile = block // ANS_GROUP
    tile_of_wide = enc.wide_group // groups_per_tile
    enc.meta.update({
        "n_out": n_out, "n_in": n_in, "n_pad": n_pad, "k_pad": k_pad,
        "n0": n0, "k0": k0, "tiles_n": tiles_n, "tiles_k": tiles_k, "n_tiles": n_tiles,
        "wide_tile_offsets": np.searchsorted(
            tile_of_wide, np.arange(n_tiles + 1), side="left").astype(np.int32),
    })
    return enc


def decode_fixed_gemm(enc: FixedEncoded) -> np.ndarray:
    """Inverse of :func:`encode_fixed_gemm`, back to (n_out, n_in) uint16."""
    m = enc.meta
    flat = decode_fixed(enc)
    w = (flat.reshape(m["tiles_n"], m["tiles_k"], m["n0"], m["k0"])
             .transpose(0, 2, 1, 3)
             .reshape(m["n_pad"], m["k_pad"]))
    return np.ascontiguousarray(w[: m["n_out"], : m["n_in"]])


def to_torch_payload(enc: FixedEncoded, device=None) -> dict:
    """Planes as torch tensors in the dtypes ``torch.ops._C_ascend.ans_fixed_decode``
    expects.

    Torch has no uint16, so every uint16 plane is handed over as int16 with the
    same bits -- the kernel only ever shifts and masks them.

    An empty wide list still has to be a real allocation, because the kernel
    takes its address unconditionally even though the loop body never runs for
    it. That padding is why the op takes ``n_wide`` explicitly instead of
    reading ``wide_group.numel()``: pass ``enc.meta["n_wide"]``, or the decoder
    would work through phantom wide groups and overwrite real output.
    """
    import torch

    def u16(a: np.ndarray, min_elems: int = 0) -> "torch.Tensor":
        if a.size < min_elems:
            a = np.zeros(min_elems, dtype=np.uint16)
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint16).view(np.int16))
        return t.to(device) if device is not None else t

    def i32(a: np.ndarray, min_elems: int = 0) -> "torch.Tensor":
        if a.size < min_elems:
            a = np.zeros(min_elems, dtype=np.int32)
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32))
        return t.to(device) if device is not None else t

    return {
        "lplane": u16(enc.lplane),
        "eplane": u16(enc.eplane),
        "dplane": u16(enc.dplane),
        "sbase": u16(enc.sbase),
        "wide_group": i32(enc.wide_group, min_elems=8),
        "wide_l": u16(enc.wide_l, min_elems=ANS_GROUP),
        "wide_exp": u16(enc.wide_exp, min_elems=ANS_GROUP),
    }


def decode_fixed(enc: FixedEncoded) -> np.ndarray:
    """Bit-exact CPU decoder; the reference the NPU kernel is checked against."""
    n = enc.n_weights
    block = int(enc.meta.get("block", ANS_BLOCK))
    n_blocks = n // block
    quarter = block // 4
    dquarter = block // ANS_GROUP // 4

    dplane = enc.dplane.reshape(n_blocks, dquarter)
    delta = np.zeros((n_blocks, 4, dquarter), dtype=np.uint8)
    for r in range(4):
        delta[:, r, :] = ((dplane >> (4 * r)) & 0xF).astype(np.uint8)
    delta = delta.reshape(-1)
    gbase = (np.repeat(enc.sbase.astype(np.uint8), ANS_SUPER // ANS_GROUP) + delta).astype(np.uint8)

    eplane = enc.eplane.reshape(n_blocks, quarter)
    off = np.zeros((n_blocks, 4, quarter), dtype=np.uint8)
    for r in range(4):
        off[:, r, :] = ((eplane >> (4 * r)) & 0xF).astype(np.uint8)
    exp = (np.repeat(gbase, ANS_GROUP) + off.reshape(-1)).astype(np.uint8)

    lplane = enc.lplane.reshape(n_blocks, 2, quarter)
    lo = np.zeros((n_blocks, 4, quarter), dtype=np.uint8)
    for r in range(4):
        lo[:, r, :] = ((lplane[:, r // 2, :] >> (8 * (r & 1))) & 0xFF).astype(np.uint8)
    lo = lo.reshape(-1)

    out = ((lo.astype(np.uint16) & 0x80) << 8) | (exp.astype(np.uint16) << 7) \
        | (lo.astype(np.uint16) & 0x7F)

    if enc.wide_group.size:
        rows = enc.wide_group.astype(np.int64)[:, None] * ANS_GROUP + np.arange(ANS_GROUP)
        wl = enc.wide_l.reshape(-1, ANS_GROUP).astype(np.uint16)
        we = enc.wide_exp.reshape(-1, ANS_GROUP).astype(np.uint16)
        out[rows.reshape(-1)] = (((wl & 0x80) << 8) | (we << 7) | (wl & 0x7F)).reshape(-1)
    return out
