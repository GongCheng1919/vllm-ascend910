"""vLLM quantization backend for the mid-group W4A8 checkpoint.

Registers `quant_method="midgroup_w4a8"` so vLLM allocates INT4-SHAPED parameters
during model construction and loads the packed tensors straight off disk.  That
is the whole point: the older path built the model in BF16 and converted inside
`process_weights_after_loading`, so peak memory was the BF16 footprint and
single-card 64-layer load OOM'd (results/p9_singlecard_l64.log).

TENSOR CONTRACT (produced by `export_w4a8_checkpoint.py`)

    <proj>.weight_packed  [N, k_pad_elems(K)//2]  int8   two int4 codes per byte
    <proj>.weight_scale   [N, G]                  bf16
    <proj>.weight_zero    [N, G]                  bf16
    <proj>.weight_ksum    [N, G]                  int32  sum of RAW codes

All four carry `output_dim=0`, so vLLM's stock fused (`qkv_proj`,
`gate_up_proj`) and column-parallel loaders work unchanged.  The kernel wants
the scale-like tensors as [G, N]; that transpose happens once, after loading.

TENSOR-PARALLEL LIMIT -- read this before using TP>1.

A pre-quantised checkpoint fixes the group boundaries at export time.  Row
parallel shards K, and for QwQ-32B those shards do NOT land on group boundaries:

    o_proj    K=5120,  GK=1024 -> 5 groups; TP=2 cuts at 2560, TP=4 at 1280
    down_proj K=27648, GK=1024 -> 27 groups; TP=2 cuts at 13824

Every one of those cuts falls INSIDE a group, so a rank would be applying a scale
computed over elements it does not own.  This is the same constraint recorded as
"GK=1024/TP binds the checkpoint to the TP degree".  Column-parallel layers shard
N and are unaffected.  We therefore ASSERT rather than silently mis-scale: a
TP>1 deployment needs a checkpoint exported for that TP degree.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4a8_ops as W  # noqa: E402

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

TILE_N = 128


class MidGroupW4A8Config(QuantizationConfig):
    """Config block written into config.json by the exporter."""

    def __init__(self, group_size: int = 1024, sym: bool = False,
                 ignored_layers: Optional[list[str]] = None):
        super().__init__()
        self.group_size = group_size
        self.sym = sym
        self.ignored_layers = ignored_layers or ["lm_head"]
        # vllm-ascend reaches into `quant_config.quant_description` from places
        # that have nothing to do with linear layers -- e.g. RMSNorm asks whether
        # any key contains "norm.bias" (ops/layernorm.py:42).  It assumes every
        # quant config is its own AscendQuantConfig.  We have no norm biases, so
        # an empty dict is the honest answer and keeps those call sites working.
        self.quant_description: dict[str, str] = {}

    def __repr__(self) -> str:
        return f"MidGroupW4A8Config(group_size={self.group_size}, sym={self.sym})"

    @classmethod
    def get_name(cls) -> str:
        return "midgroup_w4a8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0            # not a CUDA compute capability; Ascend ignores it

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MidGroupW4A8Config":
        return cls(group_size=config.get("group_size", 1024),
                   sym=config.get("sym", False),
                   ignored_layers=config.get("ignored_layers", ["lm_head"]))

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["MidGroupW4A8LinearMethod"]:
        if not isinstance(layer, LinearBase):
            return None
        if any(ig in prefix for ig in self.ignored_layers):
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod
            return UnquantizedLinearMethod()
        return MidGroupW4A8LinearMethod(self, prefix)


class MidGroupW4A8LinearMethod(LinearMethodBase):

    def __init__(self, quant_config: MidGroupW4A8Config, prefix: str = ""):
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int],
                       input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs) -> None:
        gk = self.quant_config.group_size
        k = input_size_per_partition
        n = sum(output_partition_sizes)

        # Fail loudly on the two contracts the kernel cannot bend.
        if k != input_size:
            raise ValueError(
                f"{self.prefix}: row-parallel shard K={k} of {input_size} — a "
                f"pre-quantised mid-group checkpoint cannot be K-sharded unless "
                f"the shard lands on a group boundary (see module docstring). "
                f"Export a checkpoint for this TP degree, or run TP=1.")
        if n % TILE_N:
            raise ValueError(f"{self.prefix}: N={n} not a multiple of {TILE_N}")

        g = W.num_groups(k, gk)
        kpad = W.k_pad_elems(k, gk)
        loader = extra_weight_attrs.get("weight_loader")

        def add(name: str, t: torch.Tensor) -> None:
            p = torch.nn.Parameter(t, requires_grad=False)
            # output_dim=0 lets the stock fused/column-parallel loaders
            # concatenate q/k/v and gate/up along N for ALL four tensors --
            # which is why the scales are stored [N, G] and not [G, N].
            setattr(p, "output_dim", 0)
            if loader is not None:
                setattr(p, "weight_loader", loader)
            layer.register_parameter(name, p)

        add("weight_packed", torch.empty(n, kpad // 2, dtype=torch.int8))
        add("weight_scale", torch.empty(n, g, dtype=torch.bfloat16))
        add("weight_zero", torch.empty(n, g, dtype=torch.bfloat16))
        add("weight_ksum", torch.empty(n, g, dtype=torch.int32))
        layer.mg_k = k
        layer.mg_n = n

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """[N, G] -> [G, N], the layout the kernel reads.

        Done once, on tensors that are 1.3% of the weight, so the transient is
        negligible -- unlike the BF16->int4 conversion this path replaces.
        """
        W.load()
        for name in ("weight_scale", "weight_zero", "weight_ksum"):
            p = getattr(layer, name)
            t = p.data.t().contiguous()
            delattr(layer, name)
            layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
        layer.weight_packed = torch.nn.Parameter(
            layer.weight_packed.data.contiguous(), requires_grad=False)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Straight line, no shape-dependent python: `midgroup_quant_a` already
        # pads to the GEMM's TILE_M and the slice below is a view.  An `if`
        # here is what caused the 3840 PadV3/MemSet regression (P4_E2E D10).
        lead, k = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, k).to(torch.bfloat16).contiguous()
        rows = x2.shape[0]
        a_hi, a_lo, a_scale, a_ksum = torch.ops.npu.midgroup_quant_a(x2)
        y = torch.ops.npu.midgroup_w4a8_gemm(
            a_hi, a_lo, a_scale, a_ksum, layer.weight_packed,
            layer.weight_scale, layer.weight_ksum, layer.weight_zero, k)
        y = y[:rows]
        if bias is not None:
            y = y + bias
        return y.reshape(*lead, layer.mg_n)


def register() -> None:
    """Make `quant_method: midgroup_w4a8` usable.  Two steps, both required.

    1. Register the config class so vLLM can resolve the name in config.json.
    2. Add the name to `NPUPlatform.supported_quantization`.  vLLM validates the
       method against a per-platform ALLOWLIST during `ModelConfig` construction
       (`platforms/interface.py: verify_quantization`), and vllm-ascend ships a
       closed list -- so step 1 alone gets you
       "midgroup_w4a8 quantization is currently not supported in npu."

    Must run before `LLM(...)`; the check happens while the config is built.
    """
    from vllm.model_executor.layers.quantization import register_quantization_config
    try:
        register_quantization_config("midgroup_w4a8")(MidGroupW4A8Config)
    except ValueError:
        pass        # already registered in this process

    from vllm.platforms import current_platform
    allow = getattr(type(current_platform), "supported_quantization", None)
    if allow is not None and "midgroup_w4a8" not in allow:
        allow.append("midgroup_w4a8")
