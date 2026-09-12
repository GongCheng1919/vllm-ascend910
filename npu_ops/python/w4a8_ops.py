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


# Which build dir serves which interpreter.  The .so's ABI is tied to the torch it
# was compiled against, so this MUST NOT be a guess: an ABI-mismatched library at
# least fails loudly, but a merely STALE one loads fine and silently grades a
# different kernel -- the same failure shape as `.inc` edits not triggering a
# rebuild (P6.5 D4).  `W4A8_OPS_LIB` overrides everything.
_BUILD_BY_TORCH = {
    "2.10": "build-023",   # .venv-023: vllm 0.23.0, torch_npu 2.10, CANN 9.1.0 (P10 Phase B)
    "2.8": "build-venv",   # .venv:     vllm 0.13.0, torch_npu 2.8,  CANN 8.5.0
}


def _lib_path() -> str:
    override = os.environ.get("W4A8_OPS_LIB")
    if override:
        return os.path.normpath(override)

    SO = "libvllm_w4a8_npu_ops.so"
    mm = ".".join(torch.__version__.split(".")[:2])
    order = [_BUILD_BY_TORCH[mm]] if mm in _BUILD_BY_TORCH else []
    order += [d for d in ("build-023", "build-venv", "build") if d not in order]

    tried = []
    for d in order:
        cand = os.path.normpath(os.path.join(_HERE, "..", d, SO))
        tried.append(cand)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"no {SO} for torch {torch.__version__}; build one with "
        f"`PYTHON=$(which python) BUILD_DIR=<dir> bash npu_ops/build.sh`, or point "
        f"W4A8_OPS_LIB at it. Tried: " + ", ".join(tried))

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


# ---------------------------------------------------------------------------
# Ragged-K group geometry.  MUST agree with npu_ops/kernel/mg_kgeom.h, which the
# kernel, the C++ host and the CPU reference all read.  It is not shared code, so
# the C++ host asserts the row length it receives equals K_PAD_ELEMS//2 -- a
# python/C++ drift therefore fails loudly on the first call instead of producing
# a plausible-looking wrong answer.
CUBE_K_ELEMS = 64          # int4 C0 = 32 B = 64 elements: the op's own alignment


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def num_groups(K: int, gk: int = GROUP) -> int:
    return _ceil_div(K, gk)


def group_real(g: int, K: int, gk: int = GROUP) -> int:
    """Elements actually in group g; only the last one can be short."""
    return gk if g + 1 < num_groups(K, gk) else K - (num_groups(K, gk) - 1) * gk


def k_pad_elems(K: int, gk: int = GROUP) -> int:
    """Padded row length: each group rounded up to a whole int4 fractal."""
    g = num_groups(K, gk)
    stride = _ceil_div(gk, CUBE_K_ELEMS) * CUBE_K_ELEMS
    last = _ceil_div(group_real(g - 1, K, gk), CUBE_K_ELEMS) * CUBE_K_ELEMS
    return (g - 1) * stride + last


def quantize_weight(w: torch.Tensor, group: int = GROUP
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 [N,K] -> (w_q packed int4, w_scale, w_ksum, w_zero) in kernel layout.

    Uses `fakequant_lab.quant.quantize` with the frozen spec (int4, asymmetric,
    mid-group along K) so the deployed weight is the one P2 measured.

    `w_ksum` is the sum of the RAW codes and is deliberately NOT zero-corrected:
    it exists for the MSD `+8` bias, not for asymmetry.  The zero point enters
    through the separate `w_zero * a_ksum` term.  Mixing the two costs SNR
    without raising an error, so it is asserted at the seam instead.

    RAGGED K.  K need not be a multiple of `group`.  The final group is quantised
    on its REAL members and then zero-padded up to an int4 fractal in the packed
    layout (`k_pad_elems`).  The tail is quantised as its own single group rather
    than by padding the weight first: for an ASYMMETRIC spec, appending zeros
    changes (max - min), so padding before quantising would move the scale and the
    zero point of the REAL elements -- silently, and only for the last group.
    """
    q = _fakequant_lab()
    N, K = w.shape
    G = num_groups(K, group)
    k_full = (G - 1) * group
    tail = K - k_full

    parts_q, parts_s, parts_z = [], [], []
    if k_full:
        spec = q.QuantSpec(4, group, "W", sym=False)
        qt = q.quantize(w[:, :k_full].float(), spec)
        assert qt.zero is not None, "asymmetric spec produced no zero point"
        parts_q.append(qt.q.to(torch.int8).reshape(N, k_full))
        parts_s.append(qt.scale)
        parts_z.append(qt.zero)
    # The tail: ONE group whose width is exactly what is left.
    qt_t = q.quantize(w[:, k_full:].float(), q.QuantSpec(4, tail, "W", sym=False))
    assert qt_t.zero is not None, "asymmetric spec produced no zero point"
    parts_q.append(qt_t.q.to(torch.int8).reshape(N, tail))
    parts_s.append(qt_t.scale)
    parts_z.append(qt_t.zero)

    codes = torch.cat(parts_q, dim=1).cpu()               # [N, K], real elements
    w_scale = torch.cat(parts_s, dim=1).to(torch.bfloat16).T.contiguous()   # [G, N]
    w_zero = torch.cat(parts_z, dim=1).to(torch.bfloat16).T.contiguous()    # [G, N]

    # w_ksum over each group's REAL members (the pad is zero, so this equals the
    # sum over the padded span -- see kernel/mg_kgeom.h).
    ksum = torch.stack([codes[:, g * group:g * group + group_real(g, K, group)]
                        .to(torch.int32).sum(dim=1) for g in range(G)], dim=1)
    w_ksum = ksum.to(torch.int32).T.contiguous()          # [G, N]

    # Pack into the grouped PADDED layout: group g at element offset
    # g * ceil(group/64), its tail zero.
    kpad = k_pad_elems(K, group)
    stride = _ceil_div(group, CUBE_K_ELEMS) * CUBE_K_ELEMS
    padded = codes.new_zeros((N, kpad))
    for g in range(G):
        real = group_real(g, K, group)
        padded[:, g * stride:g * stride + real] = codes[:, g * group:g * group + real]
    w_q = pack_int4(padded).reshape(N, kpad // 2)
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
