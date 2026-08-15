"""Swap a loaded vLLM model's Linear layers onto the mid-group W4A8 kernels.

This is P4 step 3 (`MGCKPT/P4_E2E.md` §2): the first time these kernels run
inside the real engine rather than a hand-built layer.

**Why it converts from BF16 instead of loading the stock W4A8 checkpoint.**
`models/QwQ-32B-W4A8-Random` is quantised at group_size=128 by the vendor
toolchain, and its weights are already int4 on disk.  Our kernel is GK=1024
(compile-time), so those tensors are the wrong granularity; recovering a bf16
weight from them and re-quantising would be a double quantisation with worse
accuracy than either.  Converting a BF16 model instead keeps the numerics
bit-identical to `fakequant_lab` -- the same quantiser P2 measured -- at the
cost of needing the BF16 weights resident during load.  Producing a checkpoint
already in our layout is P4 step 1 and removes that cost.

**Dispatch.**  Unlike `vllm_midgroup_linear.should_use_midgroup`, this converter
uses the kernel on every projection at every M.  That policy is only valid
against a BF16 baseline, which is what this path replaces: CKPT-A measures
1.45-2.52x vs BF16 across batch 1-128, i.e. it never loses.  The 0.94x at
batch=128 is against W8A8, a baseline this converter does not compete with.
Falling back per-M would mean keeping the BF16 weight resident, which defeats
the memory win.

usage:
    from vllm_engine_patch import convert_model
    llm = LLM(model=..., ...)
    n = convert_model(llm)
"""
from __future__ import annotations

import gc
import os
import sys
import time
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4a8_ops as W  # noqa: E402
from vllm_midgroup_linear import MidGroupW4A8Linear  # noqa: E402

GROUP = 1024
TILE_N = 128


class W8A8Linear:
    """Per-channel INT8 weight + per-token dynamic INT8 activation.

    The baseline this project has to beat.  Deliberately built the SAME way as
    `MidGroupW4A8Linear` -- converted from the same BF16 weight at the same
    point in loading, over the same set of layers -- so a W4A8-vs-W8A8 number
    isolates the GEMM and not the checkpoint, the layer count, or the
    conversion path.

    It is the same op pair `bench_decode_graph.py` calls `w8a8`, which is where
    CKPT-A's 1.30x denominator came from, so the engine and hand-built calibers
    stay comparable.

    NOTE this is not `models/QwQ-32B-W8A8`.  Measuring the vendor checkpoint is
    a separate question (it brings its own quantiser and its own layer set); it
    would answer "is their build faster", not "is our GEMM faster".
    """

    def __init__(self, weight_bf16: torch.Tensor, name: str = "", device: str = "npu:0"):
        import torch_npu  # noqa: F401
        self.name = name
        w = weight_bf16.to(torch.float32)
        # Symmetric per-output-channel, the standard W8A8 weight quantiser.
        scale = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
        q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
        # npu_quant_matmul wants the weight as [K, N]; keep the transposed view
        # exactly as bench_decode_graph.py does -- the format the op is happy
        # with is part of what that bench validated.
        self._q = q.to(device)
        self.wt = self._q.t()
        self.ws = scale.to(torch.float32).to(device)
        self.N, self.K = weight_bf16.shape

    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        import torch_npu
        qx, pts = torch_npu.npu_dynamic_quant(x)
        y = torch_npu.npu_quant_matmul(qx, self.wt, self.ws, pertoken_scale=pts,
                                       output_dtype=torch.bfloat16)
        return y if bias is None else y + bias


class MidGroupLinearMethod:
    """Stands in for vLLM's LinearMethodBase on a converted layer.

    Only `apply` is ever called after loading; `create_weights` and
    `process_weights_after_loading` have already run on the original method.
    """

    def __init__(self, mg, name: str):
        self.mg = mg
        self.name = name

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mg(x, bias)


def patch_unquantized_linear(verbose: bool = True, arm: str = "w4a8") -> None:
    """Convert at LOAD time, by wrapping `UnquantizedLinearMethod`.

    Must be called BEFORE constructing the LLM.  This is the hook that matters
    for measurement: vllm-ascend captures ACL graphs during engine init, so a
    conversion done after `LLM(...)` returns would leave the captured graphs
    pointing at the BF16 matmuls (and at a weight this converter frees).
    Converting inside `process_weights_after_loading` puts the kernels in place
    before capture, which is the whole point of P4 step 3.
    """
    # Patch the class the platform actually instantiates.  vllm-ascend's
    # subclass calls super().process_weights_after_loading(layer) and THEN does
    # `layer.weight.data = maybe_trans_nz(...)`, so patching only the vLLM base
    # deletes the weight out from under it -- `AscendQKVParallelLinear object
    # has no attribute 'weight'` at load time.  Converted layers need no NZ
    # transform anyway: the kernel consumes its own packed layout.
    try:
        from vllm_ascend.ops.linear import (
            AscendUnquantizedLinearMethod as _Target)
    except ImportError:
        from vllm.model_executor.layers.linear import (
            UnquantizedLinearMethod as _Target)

    if getattr(_Target, "_midgroup_patched", False):
        return
    # Both arms convert the SAME layers at the SAME point -- that is the whole
    # point of routing W8A8 through this patch instead of loading a vendor
    # checkpoint.  Only `build` differs.
    build = {"w4a8": MidGroupW4A8Linear, "w8a8": W8A8Linear}[arm]
    if arm == "w4a8":
        W.load()
    stock = _Target.process_weights_after_loading
    stats = {"converted": 0, "skipped": 0, "t": 0.0, "names": [], "arm": arm}

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = getattr(layer, "weight", None)
        if w is None or not _supported(w.data):
            stats["skipped"] += 1
            return stock(self, layer)
        t0 = time.perf_counter()
        name = getattr(layer, "prefix", "") or f"{tuple(w.data.shape)}"
        if stats["converted"] < 8:
            stats["names"].append(f"{name} {tuple(w.data.shape)}")
        mg = build(w.data, name=name, device=str(w.data.device))
        layer.quant_method = MidGroupLinearMethod(mg, name)
        del layer.weight
        layer._midgroup = mg
        stats["converted"] += 1
        stats["t"] += time.perf_counter() - t0
        if stats["converted"] % 64 == 0:
            gc.collect()
            torch.npu.empty_cache()
            if verbose:
                print(f"[midgroup] {stats['converted']} linears converted "
                      f"({stats['t']:.0f}s)", flush=True)

    _Target.process_weights_after_loading = process_weights_after_loading
    _Target._midgroup_patched = True
    _Target._midgroup_stats = stats
    if verbose:
        print(f"[midgroup] {_Target.__name__} patched for arm={arm}; "
              "conversion happens during model load")


def _supported(w: torch.Tensor) -> bool:
    """The kernel's shape contract (host-side TORCH_CHECKs, mirrored here)."""
    if w is None or w.dim() != 2:
        return False
    n, k = w.shape
    return (k % GROUP == 0) and (n % TILE_N == 0)


def _iter_linears(model):
    from vllm.model_executor.layers.linear import LinearBase
    for name, mod in model.named_modules():
        if isinstance(mod, LinearBase):
            yield name, mod


def get_model(llm):
    """Reach the nn.Module inside an LLM handle (TP=1, in-process worker)."""
    try:
        return llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass
    # v1 engine layout
    ex = llm.llm_engine.engine_core.engine_core.model_executor
    return ex.driver_worker.model_runner.model


def convert_model(target, skip: tuple = (), verbose: bool = True) -> int:
    """Convert every supported Linear in place.  Returns the number converted.

    `target` may be an `LLM` or the model module itself.  The BF16 weight of
    each converted layer is dropped as soon as its mid-group tensors exist, so
    peak memory is (BF16 model) and steady state is about a quarter of it.
    """
    W.load()
    model = target if isinstance(target, torch.nn.Module) else get_model(target)
    converted, skipped, t0 = 0, [], time.perf_counter()
    for name, mod in _iter_linears(model):
        if any(s in name for s in skip):
            skipped.append((name, "skip-list"))
            continue
        w = getattr(mod, "weight", None)
        if w is None:
            skipped.append((name, "no weight"))
            continue
        if not _supported(w.data):
            skipped.append((name, f"shape {tuple(w.data.shape)}"))
            continue
        mg = MidGroupW4A8Linear(w.data, name=name, device=str(w.data.device))
        mod.quant_method = MidGroupLinearMethod(mg, name)
        # Drop the BF16 weight: nothing reads it once quant_method is swapped,
        # and holding it would keep the model at BF16 footprint.
        del mod.weight
        mod._midgroup = mg
        converted += 1
        if converted % 32 == 0:
            gc.collect()
            torch.npu.empty_cache()
    gc.collect()
    torch.npu.empty_cache()
    if verbose:
        dt = time.perf_counter() - t0
        print(f"[midgroup] converted {converted} linears in {dt:.1f}s")
        seen = set()
        for n, why in skipped:
            tag = why.split()[0]
            if tag not in seen:
                seen.add(tag)
                print(f"[midgroup] skipped e.g. {n}: {why}")
        print(f"[midgroup] skipped {len(skipped)} total")
    return converted


def stats() -> dict:
    """Conversion counters from the load-time patch (empty if never patched)."""
    try:
        from vllm_ascend.ops.linear import (
            AscendUnquantizedLinearMethod as _Target)
    except ImportError:
        from vllm.model_executor.layers.linear import (
            UnquantizedLinearMethod as _Target)
    return getattr(_Target, "_midgroup_stats", {})
