# Copyright (c) 2026.
"""Run a real model with every projection weight compressed, and time it.

Compression happens on the CPU and only the planes reach the NPU, so a model
that does not fit in HBM in bf16 can still be loaded.

Two claims are checked, and they must be checked apart:

  the codec is lossless   -> decoded weights compared bit-for-bit
  the fused GEMM is right -> generated tokens identical; logits may still move
                             by a rounding ULP because the fused op sums in a
                             different order from aten's matmul

Usage::

    python3 e2e.py bind/librans.so --model <hf dir> [--mode both|bf16|ans]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def bench(fn, warmup=3, rep=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("so")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("both", "bf16", "ans"), default="both")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--head", action="store_true",
                    help="also compress lm_head (2.4 GiB on Qwen3.8-27B)")
    args = ap.parse_args()

    import torch_npu  # noqa: F401
    from transformers import (AutoConfig, AutoModelForCausalLM,
                              AutoModelForImageTextToText, AutoTokenizer)

    torch.ops.load_library(args.so)
    from rANS.ans_linear import ops_available, swap_linears, _decoder_blocks
    if not ops_available():
        print(f"[e2e] rans ops not registered by {args.so}")
        return 1
    torch.npu.set_device(args.device)
    dev = f"npu:{args.device}"

    tok = AutoTokenizer.from_pretrained(args.model)
    # Qwen3.8-27B is Qwen3_5ForConditionalGeneration: AutoModelForCausalLM picks
    # the text-only class, hands it the composite config and dies on
    # `'Qwen3_5Config' object has no attribute 'vocab_size'`. Text-only
    # generation on the multimodal class works fine.
    arch = AutoConfig.from_pretrained(args.model).architectures or [""]
    auto = (AutoModelForImageTextToText if arch[0].endswith("ForConditionalGeneration")
            else AutoModelForCausalLM)
    t0 = time.perf_counter()
    model = auto.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if args.max_layers:
        blocks = _decoder_blocks(model)
        while len(blocks) > args.max_layers:
            del blocks[-1]
    print(f"  loaded on CPU in {time.perf_counter() - t0:.0f} s, "
          f"{len(_decoder_blocks(model))} decoder layers")

    if args.mode == "ans":       # compress before anything reaches the device
        t0 = time.perf_counter()
        n, orig, comp, checked, wrong = swap_linears(model, progress=True,
                                                     include_head=args.head)
        print(f"  compressed {n} linears in {time.perf_counter() - t0:.0f} s: "
              f"{orig / 2**30:.1f} -> {comp / 2**30:.1f} GiB "
              f"({orig / max(comp, 1):.3f}x, {8.0 * comp / (orig // 2):.3f} bits/weight)")
        print(f"  weight bit-exactness NOT checked here (compressed on CPU); "
              f"run tests/test_tiles.py for these shapes")
        print(f"  {sum(1 for _ in model.modules())} modules, "
              f"{sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30:.1f} GiB "
              f"of parameters left on the CPU")

    t0 = time.perf_counter()
    model = model.to(dev).eval()
    torch.npu.synchronize()
    print(f"  on {dev} in {time.perf_counter() - t0:.0f} s, "
          f"HBM {torch.npu.memory_allocated(args.device) / 2**30:.1f} GiB")

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)
    SHAPES = [(1, 1), (1, 128), (1, 512)]

    def measure(tag):
        with torch.no_grad():
            logits = model(ids).logits.float().cpu()
            lat = {s: bench(lambda s=s: model(torch.randint(0, 1000, s, device=dev)))
                   for s in SHAPES}
            t0 = time.perf_counter()
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
            torch.npu.synchronize()
            gen = time.perf_counter() - t0
        print(f"  [{tag}] " + "  ".join(f"{b}x{s}: {lat[(b, s)]*1e3:7.1f} ms" for b, s in SHAPES)
              + f"  | generate {args.max_new} tok = {args.max_new/gen:.1f} tok/s")
        return logits, out, lat

    if args.mode != "both":
        print(f"\n=== {args.mode} ===")
        measure(args.mode)
        print("\n[status] single-mode run, no comparison made")
        return 0

    print("\n=== baseline (bf16) ===")
    b_log, b_out, b_lat = measure("bf16")
    n, orig, comp, checked, wrong = swap_linears(model, verify=True,
                                                 include_head=args.head)
    torch.npu.empty_cache()
    print(f"\n=== ANS fused: {n} linears, {orig/2**20:.0f} -> {comp/2**20:.0f} MiB "
          f"({orig/max(comp,1):.3f}x) ===")
    print(f"  HBM {torch.npu.memory_allocated(args.device) / 2**30:.1f} GiB resident")
    print(f"  decoded weights bit-identical: {checked - wrong}/{checked} checked")
    a_log, a_out, a_lat = measure("ans ")

    delta = (a_log - b_log).abs().max().item()
    scale = b_log.abs().max().item()
    same = torch.equal(b_out.cpu(), a_out.cpu())
    print(f"\n  logits max|delta| = {delta:g} ({delta/max(scale,1e-9):.2e} of max|logit|)"
          f"   tokens identical = {same}")
    print(f"  {'shape':>8}{'bf16 ms':>10}{'ans ms':>10}{'ratio':>8}")
    for b, s in SHAPES:
        print(f"  {f'{b}x{s}':>8}{b_lat[(b,s)]*1e3:10.1f}{a_lat[(b,s)]*1e3:10.1f}"
              f"{a_lat[(b,s)]/b_lat[(b,s)]:8.2f}x")
    ok = wrong == 0 and checked == n and same and delta <= 4.0 * scale * 2.0**-8
    print(f"\n[status] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
