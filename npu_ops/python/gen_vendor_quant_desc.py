#!/usr/bin/env python3
"""Synthesise a ModelSlim `quant_model_description.json` for Qwen3.8-27B.

WHY THIS EXISTS, and what it is NOT.

The delivery matrix wants a `-native` arm: vllm-ascend's OWN W8A8 / W4A8 kernels
on this model's shapes.  For QwQ-32B that arm loaded a vendor checkpoint.  For
Qwen3.8-27B **no vendor-quantised checkpoint exists** -- probed on ModelScope
2026-09-05, the only Qwen3.8-27B repos are `Qwen/Qwen3.8-27B` (BF16) and
`unsloth/Qwen3.8-27B-GGUF`; every `<org>/Qwen3.8-27B-W8A8`-shaped id 404s.

But vllm-ascend clearly INTENDS to support it: `model_type "qwen3_5"` has an
entry in `packed_modules_model_mapping` (qkv_proj, gate_up_proj, in_proj_qkvz,
in_proj_ba), so the quantised path is written for this architecture.  What is
missing is only the calibrated weights.

Since EVERY arm in this matrix runs `load_format=dummy`, no arm has calibrated
weights and no arm makes an accuracy claim.  So the missing piece is not needed
for a THROUGHPUT comparison: a description file is enough to make vLLM allocate
int8/int4 parameters and dispatch `AscendW8A8DynamicLinearMethod` /
`AscendW4A8DynamicLinearMethod`.  That is a real measurement of the vendor's
kernels on the real geometry.

  THIS IS NOT "the vendor's shipping build".  Call the arm
  `w8a8-native(synth-desc)`.  Two things are ours, not theirs:
    1. the LAYER SET -- chosen to match our arm exactly, so the ratio isolates
       the GEMM (the same reason W8A8Linear goes through our own patch rather
       than a vendor checkpoint);
    2. the calibration -- there is none, in any arm.
  A real vendor build would differ on (1): on QwQ-32B the shipped W8A8 leaves
  every `down_proj` in FLOAT, which is a calibration decision no synthetic file
  can reproduce.

usage:
  python npu_ops/python/gen_vendor_quant_desc.py \
      --src models/Qwen3.8-27B --dst models/Qwen3.8-27B-W8A8-Synth --type W8A8_DYNAMIC
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

# Layers our own patch converts (`vllm_engine_patch._supported`: 2-D and
# N % 128 == 0), expressed over CHECKPOINT names.  Keeping the two sets equal is
# the whole point -- an arm that quantises a different set is measuring a
# different model, not a different kernel.
QUANT_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2",
    "merger.linear_fc1", "merger.linear_fc2",
)
# The by-design BF16 set (P10 D5), same three classes the `qwen38` expected-skip
# policy declares.  `in_proj_b`/`in_proj_a` are the two shards vLLM fuses into
# `in_proj_ba`; they must agree or vllm-ascend raises.
FLOAT_SUFFIXES = (
    "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.conv1d",
)
# N = 4304, not a multiple of 128 -> our kernel skips it, so the vendor arm must
# skip it too or the two arms cover different layers.
FLOAT_CONTAINS = ("visual.blocks.", "mlp.linear_fc1")


def build(src: str, quant_type: str, group_size: int, extra_float: tuple = (),
          float_substr: tuple = ()):
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    names = sorted(idx["weight_map"])
    desc: dict = {"model_quant_type": quant_type.split("_")[0]}
    if quant_type.startswith("W4A8"):
        # AscendW4A8DynamicLinearMethod.__init__ reads these two straight out of
        # quant_description; the default group_size is 256, which does not divide
        # the vision tower's K=1152.
        desc["group_size"] = group_size
        desc["version"] = "0"
    n_q = 0
    for name in names:
        if not name.endswith(".weight"):
            desc[name] = "FLOAT"
            continue
        stem = name[: -len(".weight")]
        float_by_design = (
            any(stem.endswith(s) for s in FLOAT_SUFFIXES)
            or all(c in stem for c in FLOAT_CONTAINS)
            or any(stem.endswith(s) for s in extra_float)
            or any(sub in stem for sub in float_substr)
        )
        quantisable = any(stem.endswith(s) for s in QUANT_SUFFIXES)
        if quantisable and not float_by_design:
            desc[name] = quant_type
            n_q += 1
        else:
            desc[name] = "FLOAT"
    # `conv1d.weight` is 3-D but vllm-ascend still wraps it in a LinearBase, and
    # every LinearBase prefix MUST have an entry: a missing one makes
    # is_layer_skipped_ascend() return False and then get_linear_quant_type()
    # dies on KeyError instead of falling back.
    return desc, n_q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/Qwen3.8-27B")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--type", default="W8A8_DYNAMIC",
                    choices=["W8A8_DYNAMIC", "W4A8_DYNAMIC"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--layers", type=int, default=0,
                    help="truncate to L layers (0 = the real 64); layer_types is "
                         "truncated with it, which is what keeps the 3:1 "
                         "GDN/full-attention mix")
    ap.add_argument("--float-substr", action="append", default=[],
                    help="force FLOAT for any layer whose name CONTAINS this, "
                         "e.g. `visual.` for W4A8: vllm-ascend's own "
                         "AscendW4A8DynamicLinearMethod.apply never flattens x, "
                         "so the vision tower's 3-D input aborts the engine with "
                         "`AclNN_Parameter_Error(EZ1001): x's dim should be in "
                         "range [2, 2]. actual is [3]`.  That is the vendor "
                         "path's limitation, not a shape our arms share -- ours "
                         "reshape -- so the vendor W4A8 arm covers the language "
                         "model only.  Decode is unaffected (the vision tower "
                         "does not run on a text prompt); the WEIGHT FOOTPRINT "
                         "is, and must be reported with that caveat.")
    ap.add_argument("--float-suffix", action="append", default=[],
                    help="additionally force these layer suffixes to FLOAT, e.g. "
                         "`mlp.linear_fc2` for W4A8 where group_size does not "
                         "divide the vision tower's K=4304")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for f in os.listdir(args.src):
        if f in ("config.json", "model.safetensors.index.json"):
            continue
        d = os.path.join(args.dst, f)
        if not os.path.exists(d):
            shutil.copy2(os.path.join(args.src, f), d)

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    if args.layers:
        cfg["text_config"]["num_hidden_layers"] = args.layers
        cfg["text_config"]["layer_types"] = cfg["text_config"]["layer_types"][:args.layers]
    # No `quantization_config` here, exactly like the vendor checkpoints: the
    # method is declared only in quant_model_description.json and is reached by
    # asking vLLM for `quantization="ascend"`.
    json.dump(cfg, open(os.path.join(args.dst, "config.json"), "w"), indent=2)

    desc, n_q = build(args.src, args.type, args.group_size,
                      tuple(args.float_suffix), tuple(args.float_substr))
    if args.layers:
        keep = {f".layers.{i}." for i in range(args.layers)}
        desc = {k: v for k, v in desc.items()
                if ".layers." not in k or any(t in k for t in keep)}
        n_q = sum(1 for v in desc.values() if v == args.type)
    out = os.path.join(args.dst, "quant_model_description.json")
    json.dump(desc, open(out, "w"), indent=2)
    n_f = sum(1 for v in desc.values() if v == "FLOAT")
    print(f"{out}: {n_q} x {args.type}, {n_f} x FLOAT, {len(desc)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
