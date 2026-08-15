"""torch.ops.npu.midgroup_* — the P3 W4A8 kernels, wrapped for PyTorch.

Importing this module loads `libvllm_w4a8_npu_ops.so`, which registers:

    torch.ops.npu.midgroup_quant_a(x) -> [a_hi, a_lo, a_scale, a_ksum]
    torch.ops.npu.midgroup_w4a8_gemm(a_hi, a_lo, a_scale, a_ksum,
                                     w_q, w_scale, w_ksum, w_zero, K) -> y

`quantize_weight` produces the weight-side tensors in the kernel's layout from a
bf16 weight, using `fakequant_lab.quant` so the deployed weights and P2's
accuracy runs come from the same quantiser.
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import torch

# Two build trees: `build` for the system interpreter (torch 2.7.1, used by the
# int4_cube_lab checks) and `build-venv` for vLLM's .venv (torch 2.8.0).  A .so
# built against the other torch loads but every at::Tensor arrives as garbage,
# so pick by the running interpreter's torch version.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _lib_path() -> str:
    venv = os.path.join(_HERE, "..", "build-venv", "libvllm_w4a8_npu_ops.so")
    plain = os.path.join(_HERE, "..", "build", "libvllm_w4a8_npu_ops.so")
    in_venv = "/.venv/" in os.path.abspath(sys.executable) or torch.__version__.startswith("2.8")
    first, second = (venv, plain) if in_venv else (plain, venv)
    return os.path.normpath(first if os.path.exists(first) else second)

GROUP = 1024


def tile_m_for(m: int) -> int:
    """TILE_M the GEMM picks for `m` rows.

    Mirrors `TileMFor` in BOTH host files.  Three definitions of one function is
    two too many, but the C++ ones must agree with each other at run time and
    this one only feeds shape inference (see `pad_rows`).
    """
    return 16 if m <= 16 else (64 if m <= 64 else 128)


def pad_rows(m: int) -> int:
    """Row count `midgroup_quant_a` actually emits for an `m`-row activation.

    Used only by the fake kernel.  Getting it wrong is not a correctness bug --
    the caller slices the GEMM output back to its own row count either way, and
    the real allocation comes from the C++ host -- but a matching fake keeps
    traced shapes honest.
    """
    t = tile_m_for(m)
    return (m + t - 1) // t * t


def load() -> None:
    """Load the op library (idempotent)."""
    if hasattr(torch.ops.npu, "midgroup_quant_a"):
        return
    import torch_npu  # noqa: F401  -- registers the PrivateUse1 backend first
    torch.ops.load_library(_lib_path())
    _register_fake()


_FAKE_DONE = False


def _register_fake() -> None:
    """Meta ('fake') kernels, required to run under torch.compile / ACL graph.

    vLLM traces the model with FakeTensors before capturing the graph.  A custom
    op with no fake implementation aborts that trace with
    `Unsupported: Operator does not support running with fake tensors`, which is
    what P4 step 3 hit the first time the engine tried to capture with these
    kernels in place.  Shapes here must match the host bindings exactly.
    """
    global _FAKE_DONE
    if _FAKE_DONE:
        return

    @torch.library.register_fake("npu::midgroup_quant_a")
    def _quant_a_fake(x):
        M, K = x.shape
        G = K // GROUP
        Mp = pad_rows(M)
        return (
            x.new_empty((Mp, K // 2), dtype=torch.int8),     # a_hi packed int4
            x.new_empty((Mp, K // 2), dtype=torch.int8),     # a_lo packed int4
            x.new_empty((G, Mp), dtype=torch.bfloat16),      # a_scale group-major
            x.new_empty((G, Mp), dtype=torch.int32),         # a_ksum (negated)
        )

    @torch.library.register_fake("npu::midgroup_w4a8_gemm")
    def _gemm_fake(a_hi, a_lo, a_scale, a_ksum, w_q, w_scale, w_ksum, w_zero, K):
        # a_hi already carries the padded row count, and the host's own Pad2D
        # no-ops on it, so the output row count is simply a_hi's.  Callers slice
        # back to their real row count themselves.
        return a_scale.new_empty((a_hi.shape[0], w_q.shape[0]),
                                 dtype=torch.bfloat16)

    _FAKE_DONE = True


def _fakequant_lab():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from fakequant_lab import quant  # noqa: WPS433
    return quant


def pack_int4(v: torch.Tensor) -> torch.Tensor:
    """[..., K] int8 codes in [-8,7] -> [..., K/2] packed, even index in the LOW nibble."""
    assert v.shape[-1] % 2 == 0, "K must be even to pack int4"
    lo = v[..., 0::2].to(torch.int16) & 0xF
    hi = v[..., 1::2].to(torch.int16) & 0xF
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed.view(torch.int8) if packed.dtype == torch.uint8 else packed


def quantize_weight(w: torch.Tensor, group: int = GROUP
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 [N,K] -> (w_q packed int4, w_scale, w_ksum, w_zero) in kernel layout.

    Uses `fakequant_lab.quant.quantize` with the frozen spec (int4, asymmetric,
    mid-group along K) so the deployed weight is the one P2 measured.

    `w_ksum` is the sum of the RAW codes and is deliberately NOT zero-corrected:
    it exists for the MSD `+8` bias, not for asymmetry.  The zero point enters
    through the separate `w_zero * a_ksum` term.  Mixing the two costs SNR
    without raising an error, so it is asserted at the seam instead.
    """
    q = _fakequant_lab()
    spec = q.QuantSpec(4, group, "W", sym=False)
    qt = q.quantize(w.float(), spec)
    assert qt.zero is not None, "asymmetric spec produced no zero point"

    N, K = w.shape
    G = K // group
    codes = qt.q.to(torch.int8).reshape(N, K)
    w_q = pack_int4(codes.cpu()).reshape(N, K // 2)
    # group-major [G, N]
    w_scale = qt.scale.to(torch.bfloat16).T.contiguous()
    w_zero = qt.zero.to(torch.bfloat16).T.contiguous()
    w_ksum = codes.cpu().reshape(N, G, group).sum(dim=-1).to(torch.int32).T.contiguous()
    return w_q, w_scale, w_ksum, w_zero


def linear(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor,
           wk: torch.Tensor, wz: torch.Tensor) -> torch.Tensor:
    """y = x @ W.T with mid-group W4A8; x is bf16 [..., K]."""
    load()
    lead, K = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, K).to(torch.bfloat16).contiguous()
    a_hi, a_lo, a_scale, a_ksum = torch.ops.npu.midgroup_quant_a(x2)
    y = torch.ops.npu.midgroup_w4a8_gemm(a_hi, a_lo, a_scale, a_ksum, wq, ws, wk, wz, K)
    # quant_a pads to TILE_M, so y has >= x2's rows; slice before reshaping.
    return y[:x2.shape[0]].reshape(*lead, y.shape[-1])
