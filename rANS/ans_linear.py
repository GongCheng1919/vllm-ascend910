# Copyright (c) 2026.
"""Lossless fixed-rate BF16 weight compression, decoded inside the GEMM.

Standalone counterpart to `_encode_fixed_gemm_layer` / `_fused_fixed_linear` in
the vllm-ascend checkout's `vllm_ascend/quantization/methods/ans_bf16.py`. That
one is bound to vllm-ascend's `AscendLinearScheme`; this one is a plain
`nn.Module` so it can be used against stock vLLM / transformers here.

**Keep the two in sync** -- the encode layout and the apply sequence are the
same code path, and the kernel is shared by symlink.

What the codec is: sign+mantissa raw, exponent as a 4-bit offset from a
two-level base. 12.33 bits/weight on real projection weights, lossless. See
`codec/fixed_codec.py` and `../../awork/tools/ans/CHECKPOINT.md`.

Cost, measured: the fused op is ~3.4x a bf16 matmul on device time, which
dilutes to ~1.6x per token end to end. This buys weight capacity, not speed.
"""

from __future__ import annotations

import numpy as np
import torch

from .codec.fixed_codec import encode_fixed_gemm

__all__ = ["AnsFusedLinear", "swap_linears", "ops_available"]

SKIP_NAMES = ("lm_head", "embed")

# Below this the fused op is host-bound: ~140 us of launch cost per call no
# matter the shape, against a device time that finishes sooner. A 48x5120
# projection saves 0.47 MB and costs the full 140 us. Measured on
# Qwen3.8-27B: the 96 linear_attn.in_proj_a/b layers were 8% of the decode
# step for 45 MB out of 45 GiB. Pass min_bytes=0 to compress everything.
MIN_BYTES = 4 * 2**20


def ops_available() -> bool:
    return hasattr(torch.ops.rans, "ans_fixed_decode_gemm")


class AnsFusedLinear(torch.nn.Module):
    """nn.Linear whose weight only ever exists compressed.

    The weight is encoded on whatever device it is currently on -- pass a CPU
    module and only the compressed planes need to reach the NPU, which is what
    lets a model larger than HBM be loaded at all.
    """

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        w = lin.weight.data
        if w.dtype is not torch.bfloat16:
            raise TypeError(f"AnsFusedLinear needs bfloat16, got {w.dtype}")
        n_out, n_in = int(w.shape[0]), int(w.shape[1])
        codes = np.ascontiguousarray(
            w.detach().to("cpu").contiguous().view(torch.uint16).numpy().ravel())
        enc = encode_fixed_gemm(codes, n_out, n_in)
        if enc is None:
            raise ValueError(f"codec declined {n_out}x{n_in}: would not beat bf16")

        dev = w.device
        def _u16(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint16).view(np.int16)).to(dev)
        def _i32(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(dev)

        P = torch.nn.Parameter
        self.lplane = P(_u16(enc.lplane), False)
        self.eplane = P(_u16(enc.eplane), False)
        self.dplane = P(_u16(enc.dplane), False)
        self.sbase = P(_u16(enc.sbase), False)
        self.wide_group = P(_i32(enc.wide_group if enc.wide_group.size else np.zeros(8, np.int32)), False)
        # Padded to one group so the pointer is never null when there are no
        # wide groups. The PADDING IS NOT A COUNT -- the count comes from
        # wide_tile_off.
        self.wide_l = P(_u16(enc.wide_l if enc.wide_l.size else np.zeros(16, np.uint16)), False)
        self.wide_exp = P(_u16(enc.wide_exp if enc.wide_exp.size else np.zeros(16, np.uint16)), False)
        self.wide_tile_off = P(_i32(enc.meta["wide_tile_offsets"]), False)
        self.bias = lin.bias

        # Plain python ints. Reading these back off a device tensor costs a
        # device-to-host sync EACH, and forward() needs all four every call --
        # that measured ~0.48 ms per linear before it was cached.
        self.shape4 = (n_out, n_in, int(enc.meta["n_pad"]), int(enc.meta["k_pad"]))
        self.bits_per_weight = float(enc.bits_per_weight)
        self.payload_bytes = int(enc.payload_bytes)
        self.orig_bytes = n_out * n_in * 2

    def decode_weight(self) -> torch.Tensor:
        """Decode back to a dense [n_out, n_in] bf16 tensor. For verification."""
        n_out, n_in, n_pad, k_pad = self.shape4
        out = torch.empty(n_pad, k_pad, dtype=torch.bfloat16, device=self.lplane.device)
        torch.ops.rans.ans_fixed_decode_tiles(
            self.lplane, self.eplane, self.dplane, self.sbase,
            self.wide_group, self.wide_l, self.wide_exp, self.wide_tile_off, out)
        return out[:n_out, :n_in]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_out, n_in, n_pad, k_pad = self.shape4
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.dtype is not torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        if k_pad != n_in:                      # the kernel reduces over k_pad
            pad = torch.zeros(x2.shape[0], k_pad, dtype=x2.dtype, device=x2.device)
            pad[:, :n_in] = x2
            x2 = pad
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = torch.ops.rans.ans_fixed_decode_gemm(
            x2, self.lplane, self.eplane, self.dplane, self.sbase,
            self.wide_group, self.wide_l, self.wide_exp, self.wide_tile_off,
            n_pad, k_pad, 0)
        if n_pad != n_out:
            y = y[:, :n_out]
        if self.bias is not None:
            y = y + self.bias
        if y.dtype is not x.dtype:
            y = y.to(x.dtype)
        return y.reshape(*shape[:-1], n_out)


def _decoder_blocks(model):
    """The decoder layer list, across the layouts we have met.

    Qwen2/Qwen3 use model.model.layers; the Qwen3.5 (qwen3_5) multimodal
    checkpoints nest it under model.model.language_model.layers.
    """
    for path in ("model.language_model.layers", "model.layers",
                 "language_model.model.layers", "layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("could not find the decoder layer list on this model")


def _shares_storage(a, b) -> bool:
    """Tying in transformers assigns the same Parameter, so identity catches it.

    `data_ptr()` is the fallback for a head tied by copy -- but every meta
    tensor reports 0, so it must not be consulted on a meta build or an
    untied head looks tied.
    """
    if a is b:
        return True
    if a.device.type == "meta" or b.device.type == "meta":
        return False
    return a.data_ptr() == b.data_ptr()


def _head_module(model):
    """(parent, attr) of the output projection, which is NOT in a decoder block.

    Returns (None, None) if there is no plain bf16 nn.Linear head, or if the
    head shares storage with the input embedding -- compressing a tied head
    would silently destroy embed_tokens, which is read by index, not by GEMM.
    """
    for path in ("lm_head", "model.lm_head", "language_model.lm_head"):
        parent, attr = model, path.split(".")
        for part in attr[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                break
        else:
            head = getattr(parent, attr[-1], None)
            if not isinstance(head, torch.nn.Linear):
                continue
            if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
                return None, None
            emb = getattr(model, "get_input_embeddings", lambda: None)()
            if emb is not None and _shares_storage(head.weight, emb.weight):
                return None, None
            return parent, attr[-1]
    return None, None


def swap_linears(model, limit: int = 0, skip=SKIP_NAMES, progress=False, verify=False,
                 min_bytes: int = MIN_BYTES, include_head: bool = False):
    """Replace every projection nn.Linear in the decoder blocks.

    Only the language-model decoder blocks: `_decoder_blocks` returns
    `model.language_model.layers` on the multimodal checkpoints, so the vision
    tower, its merger, lm_head and embed_tokens all stay bf16. On Qwen3.8-27B
    that is 496 of the model's 607 linears, 45.36 of 51.7 GiB.

    `include_head` also compresses lm_head, which `SKIP_NAMES` otherwise drops
    because it is outside the blocks. On Qwen3.8-27B that is 248320x5120 =
    2.4 GiB for 0.55 GiB back, and at M=1 the fused op and the bf16 matmul tie
    exactly (6501 vs 6502 us), so the capacity is free. It is off by default
    because a tied head must never be touched -- see `_head_module`.

    Returns (n, orig_bytes, comp_bytes, n_checked, n_wrong). `verify` decodes
    each weight back and compares bit-for-bit; it needs the weight to already
    be on the NPU, so it is off by default (CPU-side compression is the point).
    """
    n = orig = comp = checked = wrong = 0
    for block in _decoder_blocks(model):
        for parent in block.modules():
            for name, child in list(parent.named_children()):
                if not isinstance(child, torch.nn.Linear) or child.weight.numel() == 0:
                    continue
                if any(s in name for s in skip) or child.weight.dtype is not torch.bfloat16:
                    continue
                if child.weight.numel() * child.weight.element_size() < min_bytes:
                    continue
                if limit and n >= limit:
                    return n, orig, comp, checked, wrong
                new = AnsFusedLinear(child)
                if verify and child.weight.device.type != "cpu":
                    checked += 1
                    wrong += 0 if torch.equal(
                        new.decode_weight().view(torch.int16),
                        child.weight.data.view(torch.int16)) else 1
                setattr(parent, name, new)
                n += 1
                orig += new.orig_bytes
                comp += new.payload_bytes
                if progress and n % 32 == 0:
                    print(f"    ... {n} linears, {comp / 2**30:.1f} GiB", flush=True)

    if include_head and not (limit and n >= limit):
        parent, attr = _head_module(model)
        head = getattr(parent, attr) if parent is not None else None
        if (head is not None and head.weight.dtype is torch.bfloat16
                and head.weight.numel() * head.weight.element_size() >= min_bytes):
            new = AnsFusedLinear(head)
            if verify and head.weight.device.type != "cpu":
                checked += 1
                wrong += 0 if torch.equal(
                    new.decode_weight().view(torch.int16),
                    head.weight.data.view(torch.int16)) else 1
            setattr(parent, attr, new)
            n += 1
            orig += new.orig_bytes
            comp += new.payload_bytes
            if progress:
                print(f"    ... + lm_head, {comp / 2**30:.1f} GiB", flush=True)
    return n, orig, comp, checked, wrong
