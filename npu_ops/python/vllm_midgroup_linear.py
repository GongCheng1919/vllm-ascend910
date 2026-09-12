"""vLLM-Ascend linear method backed by the P3 mid-group W4A8 kernels.

Replaces `AscendW4A8DynamicLinearMethod.apply`, which calls
`torch_npu.npu_weight_quant_batchmatmul` -- a WEIGHT-ONLY path that dequantises
the int4 weight to bf16 and never lets the int8 activation reach the cube
(`reports/W4A8_PERGROUP_ANALYSIS.md`).  That is why W4A8 came out slower than
BF16 in the Phase 3 baseline.  Here the activation really is quantised and the
GEMM really runs on the int4 cube.

**Scope: decode.**  P3 §4.7 measured mid-group per projection against W8A8:

    qkv      1.32x (M=1)  1.15x (M=64)  1.07x (M=128)  0.80x (M=512)
    o        1.25x        1.13x         1.07x          0.83x
    gate_up  1.67x        1.38x         1.20x          1.17x
    down     1.63x        1.34x         1.20x          1.18x

so `gate_up`/`down` win everywhere while `qkv`/`o` cross over around M~256 --
at high-concurrency decode too, not just prefill.  `should_use_midgroup`
encodes that; everything else falls back to the stock path.

**This is at-risk plumbing**: P2's accuracy gate has NOT passed (GLUE -4.5
points), so end-to-end numbers from this are engineering validation only.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4a8_ops as W  # noqa: E402

# Above this many rows, qkv/o are slower than W8A8 (P3 §4.7); gate_up/down never are.
QKV_O_MAX_M = 256
_WIDE_PROJ_HINTS = ("gate_up", "gate", "up", "down", "mlp")


def should_use_midgroup(name: str, m: int) -> bool:
    """Per-(projection, M) dispatch; see the table above."""
    lowered = name.lower()
    if any(h in lowered for h in _WIDE_PROJ_HINTS):
        return True
    return m <= QKV_O_MAX_M


class MidGroupW4A8Linear:
    """Holds one layer's weight-side tensors in the kernel's layout.

    Built from a bf16 weight via `fakequant_lab.quant` (RTN asymmetric, GK=1024)
    -- the same quantiser P2 measured, so the deployed weight and the accuracy
    runs cannot drift apart.
    """

    def __init__(self, weight_bf16: torch.Tensor, name: str = "", device: str = "npu:0"):
        W.load()
        self.name = name
        wq, ws, wk, wz = W.quantize_weight(weight_bf16.to(torch.bfloat16).cpu())
        self.w_q = wq.to(device)
        self.w_scale = ws.to(device)
        self.w_ksum = wk.to(device)
        self.w_zero = wz.to(device)
        self.K = weight_bf16.shape[1]
        self.N = weight_bf16.shape[0]

    def __call__(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        lead, K = x.shape[:-1], x.shape[-1]
        assert K == self.K, f"{self.name}: K {K} != weight K {self.K}"
        x2 = x.reshape(-1, K).to(torch.bfloat16).contiguous()
        # NO SHAPE-DEPENDENT PYTHON HERE, deliberately.  `midgroup_quant_a`
        # emits its four outputs already padded to the GEMM's TILE_M, so this
        # wrapper is a straight line: quant -> gemm -> slice.
        #
        # Every earlier version had an `if mp != m:` pad here and every one of
        # them broke under the engine.  dynamo folds that branch away while
        # tracing (`int()` on the row count does NOT prevent it), the pad never
        # runs, and the GEMM host's fallback pads all FOUR quantised tensors
        # instead -- 3840 PadV3 + 3840 MemSet per profile window, 33% of decode
        # device time, visible ONLY inside vLLM and not in a hand-built
        # NPUGraph.  See P4_E2E.md D10.  The rule that came out of it: shape
        # decisions live in the C++ host, which always runs, or they are op
        # parameters.  They do not live in python.
        #
        # `y[:rows]` is a view, not a kernel: the slice is free, and it is
        # branch-free because it is correct whether or not padding happened.
        rows = x2.shape[0]
        a_hi, a_lo, a_scale, a_ksum = torch.ops.npu.midgroup_quant_a(x2)
        y = torch.ops.npu.midgroup_w4a8_gemm(a_hi, a_lo, a_scale, a_ksum,
                                             self.w_q, self.w_scale,
                                             self.w_ksum, self.w_zero, K)
        y = y[:rows]
        if bias is not None:
            y = y + bias
        return y.reshape(*lead, self.N)


def patch_w4a8_dynamic(qkv_o_max_m: int = QKV_O_MAX_M) -> None:
    """Monkeypatch vllm_ascend's W4A8 linear method onto our kernels.

    Called before the engine builds its model.  Layers whose shape the kernel
    cannot take (K not a multiple of 1024, N not a multiple of 128) keep the
    stock path, so a partially-supported model still runs.
    """
    # vllm-ascend 0.23.0 moved this module (P10 Phase B symbol audit): the class
    # name and the `apply(self, layer, x, bias=None, tp_rank=None)` signature are
    # unchanged, only the path moved and the base class became AscendLinearScheme.
    #   <=0.13.0  vllm_ascend.quantization.w4a8_dynamic
    #   >=0.23.0  vllm_ascend.quantization.methods.w4a8
    mod = None
    for path in ("vllm_ascend.quantization.methods.w4a8",
                 "vllm_ascend.quantization.w4a8_dynamic"):
        try:
            mod = importlib.import_module(path)
            break
        except ImportError:
            continue
    if mod is None:
        raise ImportError(
            "AscendW4A8DynamicLinearMethod not found in either "
            "vllm_ascend.quantization.methods.w4a8 (>=0.23.0) or "
            "vllm_ascend.quantization.w4a8_dynamic (<=0.13.0)")

    cls = mod.AscendW4A8DynamicLinearMethod
    if getattr(cls, "_midgroup_patched", False):
        return
    stock_apply = cls.apply

    def apply(self, layer, x, bias=None, tp_rank=None):
        mg = getattr(layer, "_midgroup", None)
        if mg is None:
            return stock_apply(self, layer, x, bias=bias, tp_rank=tp_rank)
        m = x.reshape(-1, x.shape[-1]).shape[0]
        if not should_use_midgroup(getattr(layer, "_midgroup_name", ""), m):
            return stock_apply(self, layer, x, bias=bias, tp_rank=tp_rank)
        return mg(x, bias)

    cls.apply = apply
    cls._midgroup_patched = True
