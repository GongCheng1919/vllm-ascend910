"""How much context fits, bf16 vs ANS, on one 910B4.

The capacity claim for Qwen3.8-27B is not OOM-vs-runs (both fit) -- it is how
much is left over for KV. So: prefill a growing prompt in segments, carrying the
cache, and record where each mode runs out.

Segments, not one long forward, on purpose. Without flash-linear-attention the
chunk path materialises [1, 48, seq/64, 64, 64] float32 tensors, so a single
long prefill would measure the fallback's activation memory rather than the
model's KV. Segmented prefill keeps that bounded and lets the cache grow.

Usage::

    python3 tests/capacity_qwen38.py bind/librans.so --model <hf dir>
"""
from __future__ import annotations

import argparse, gc, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SEG = 2048


def sweep(model, dev, target, tag):
    """Grow the cache SEG tokens at a time. Returns (tokens reached, peak GiB)."""
    from transformers import DynamicCache
    torch.npu.empty_cache(); torch.npu.reset_peak_memory_stats(dev)
    base = torch.npu.memory_allocated(dev) / 2**30
    cache, n = None, 0
    try:
        with torch.no_grad():
            while n < target:
                ids = torch.randint(0, 1000, (1, SEG), device=dev)
                out = model(ids, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                del out
                n += SEG
                cur = torch.npu.memory_allocated(dev) / 2**30
                print(f"  [{tag}] {n:>7} tok   {cur:6.1f} GiB "
                      f"(+{cur - base:5.2f} over weights)", flush=True)
    except RuntimeError as e:
        msg = str(e).split("\n")[0][:90]
        print(f"  [{tag}] STOPPED at {n} tok: {msg}", flush=True)
    peak = torch.npu.max_memory_allocated(dev) / 2**30
    del cache
    gc.collect(); torch.npu.empty_cache()
    return n, base, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("so"); ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--target", type=int, default=262144)
    args = ap.parse_args()

    import torch_npu  # noqa: F401
    from transformers import AutoConfig, AutoModelForImageTextToText
    torch.ops.load_library(args.so)
    from rANS.ans_linear import swap_linears
    torch.npu.set_device(args.device); dev = f"npu:{args.device}"

    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = model.to(dev).eval()
    torch.npu.synchronize()
    print(f"bf16 weights on {dev}: "
          f"{torch.npu.memory_allocated(args.device) / 2**30:.1f} GiB\n")
    n_bf, base_bf, peak_bf = sweep(model, dev, args.target, "bf16")

    t0 = time.perf_counter()
    n, orig, comp, _, _ = swap_linears(model)
    torch.npu.empty_cache()
    print(f"\ncompressed {n} linears in {time.perf_counter() - t0:.0f} s: "
          f"{orig / 2**30:.1f} -> {comp / 2**30:.1f} GiB\n")
    n_ans, base_ans, peak_ans = sweep(model, dev, args.target, "ans ")

    print(f"\n  {'':<6}{'weights':>10}{'context':>12}{'peak':>9}")
    print(f"  {'bf16':<6}{base_bf:>9.1f}G{n_bf:>12}{peak_bf:>8.1f}G")
    print(f"  {'ans':<6}{base_ans:>9.1f}G{n_ans:>12}{peak_ans:>8.1f}G")
    if n_bf:
        print(f"\n  context: {n_ans / n_bf:.2f}x   weights: {base_bf / base_ans:.3f}x smaller")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
