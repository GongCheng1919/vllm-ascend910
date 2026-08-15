"""GPTQ weight quantization for the mid-group W4A8 harness.

P1 showed the `W4A8-mg` gap is entirely on the weight side, and that the error is
*unstructured scatter within a group* rather than a per-input-channel pattern
(`reports/midgroup/P1_harness.md` §4b).  That is exactly what GPTQ addresses: it
quantizes column by column and pushes each column's rounding error onto the
not-yet-quantized columns, weighted by the inverse Hessian of the layer input.

Two things are specific to this project:

* **Group boundaries must stay contiguous along K.**  The kernel flushes L0C once
  per `GK` columns, so activation reordering (`act_order` / `desc_act`) is not
  available — it would need a runtime gather on the activation.  Quantization
  therefore proceeds in natural column order.
* **Group parameters come from `quant.group_params`**, so the bf16-scale rule is
  the same one the rest of the harness (and the kernel) uses, and asymmetric
  weights are supported unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .quant import (QTensor, QuantSpec, dequantize_with, group_params,
                    quantize_with)


class HessianAccumulator:
    """`H = 2·E[X^T X]` over calibration activations for one Linear input.

    q/k/v share an input, and so do gate/up — build one accumulator per distinct
    input and reuse it, which is where most of the Hessian cost goes.
    """

    def __init__(self, K: int, device):
        self.H = torch.zeros(K, K, dtype=torch.float32, device=device)
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        """`x`: `[..., K]` activations entering the Linear."""
        x2 = x.reshape(-1, x.shape[-1]).float()
        m = x2.shape[0]
        # Running mean so H does not depend on how the calibration set is batched.
        self.H *= self.n / (self.n + m)
        self.n += m
        # addmm_ instead of `H += c * (x2.T @ x2)`: the temporary would be another
        # full [K, K] fp32 matrix — 3 GB for down_proj.
        self.H.addmm_(x2.T, x2, alpha=2.0 / self.n)

    def free(self) -> None:
        self.H = None


def inverse_hessian(H: torch.Tensor, percdamp: float = 0.01, fp64: bool = False
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Upper-triangular Cholesky factor of `H^-1`, plus the dead-column mask.

    Columns whose activations were identically zero over the calibration set
    carry no information; they are pinned to an identity diagonal so the
    factorization stays well posed, and their weights are quantized by plain RTN.

    `torch.linalg.cholesky` has no NPU kernel and silently falls back to CPU, so
    the factorization is done on CPU explicitly — same result, no per-call
    fallback warning, and it is the dominant cost for `down_proj` (K=27648:
    ~75 s for the three K^3 steps).
    """
    K = H.shape[0]
    diag = torch.arange(K, device=H.device)
    dead = H[diag, diag] == 0
    H[diag[dead], diag[dead]] = 1.0

    H[diag, diag] += percdamp * torch.diag(H).mean()

    # Cast on CPU, not on device: NPU has no float64, and `H.double()` there is
    # silently demoted back to float32 (with only a warning), so the ordering
    # decides whether fp64 is actually used.
    Hc = H.cpu()
    if fp64:
        Hc = Hc.double()
    L = torch.linalg.cholesky(Hc)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    return U.float().to(H.device), dead


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, Hinv: torch.Tensor, spec: QuantSpec,
                  dead: Optional[torch.Tensor] = None,
                  blocksize: int = 128) -> QTensor:
    """Quantize `W` `[N, K]` with GPTQ error compensation.

    `Hinv` is the upper Cholesky factor of the inverse Hessian from
    `inverse_hessian`.  `blocksize` must divide the group size so group
    boundaries land on block starts (128 divides 256/512/1024).
    """
    N, K = W.shape
    G, gl = spec.groups(K), spec.group_len(K)
    assert gl % blocksize == 0 or gl <= blocksize, \
        f"group length {gl} must be a multiple of blocksize {blocksize}"

    Wf = W.float().clone()
    if dead is not None:
        Wf[:, dead] = 0.0

    codes = torch.zeros(N, K, dtype=torch.int8, device=W.device)
    scales = torch.zeros(N, G, dtype=torch.bfloat16, device=W.device)
    zeros = None if spec.sym else torch.zeros(N, G, dtype=torch.bfloat16, device=W.device)
    s = z = None

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = Wf[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(i2 - i1):
            col = i1 + i
            if col % gl == 0:
                # Group parameters come from the *error-compensated* weights of
                # this group, which is what makes GPTQ+grouping better than
                # quantizing against the original values.
                g = col // gl
                s, z = group_params(Wf[:, col:col + gl], spec.bits, spec.sym)
                scales[:, g] = s.squeeze(-1).to(torch.bfloat16)
                if zeros is not None:
                    zeros[:, g] = z.squeeze(-1).to(torch.bfloat16)

            w = W1[:, i : i + 1]
            q = quantize_with(w, s, z, spec.bits)
            dq = dequantize_with(q, s, z)
            codes[:, col] = q.squeeze(-1)

            # Push this column's error onto the columns still to be quantized.
            err = (w - dq) / Hinv1[i, i]
            W1[:, i:] -= err @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i : i + 1] = err

        Wf[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return QTensor(codes, scales, zeros, spec)
