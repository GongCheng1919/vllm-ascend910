"""Layer-streaming execution of a Qwen2 model (QwQ-32B) for fake-quant evaluation.

Why streaming instead of `device_map`: QwQ-32B is 62 GB in bf16 and we need to
run *five* quantization configs over it.  Holding one decoder layer on device at
a time costs ~1 GB, so all five configs' hidden states can live on the same card
and share a single pass over the weights — the weights are read from disk once,
not five times.  It also gives per-layer output SNR (Tier 3) for free, and
avoids depending on accelerate's NPU device placement.

The bf16 config runs the *unpatched* modules, so it is bit-exact with a plain
`Qwen2ForCausalLM` forward; `run_selftest_model.py` checks that on a small
random model instead of taking it on faith.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import (Qwen2DecoderLayer, Qwen2RMSNorm,
                                                      Qwen2RotaryEmbedding)


class ShardedWeights:
    """Random access to a sharded safetensors checkpoint, one tensor at a time."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.name_to_file: Dict[str, str] = json.loads(index.read_text())["weight_map"]
        else:
            single = "model.safetensors"
            assert (self.path / single).exists(), f"no safetensors checkpoint in {self.path}"
            with safe_open(self.path / single, framework="pt") as f:
                self.name_to_file = {k: single for k in f.keys()}
        self._open: Dict[str, object] = {}

    def _handle(self, fname: str):
        if fname not in self._open:
            self._open[fname] = safe_open(self.path / fname, framework="pt")
        return self._open[fname]

    def has(self, name: str) -> bool:
        return name in self.name_to_file

    def get(self, name: str, device, dtype=None) -> torch.Tensor:
        t = self._handle(self.name_to_file[name]).get_tensor(name)
        t = t.to(device)
        return t.to(dtype) if dtype is not None else t

    def prefix(self, prefix: str) -> List[str]:
        return [n for n in self.name_to_file if n.startswith(prefix)]

    def close(self) -> None:
        self._open.clear()


class StreamingQwen2:
    """Holds the small always-resident parts; hands out one decoder layer at a time."""

    def __init__(self, path: str | Path, device, dtype=torch.bfloat16,
                 attn_impl: str = "sdpa", num_layers: Optional[int] = None):
        self.device, self.dtype = torch.device(device), dtype
        self.weights = ShardedWeights(path)
        self.config = AutoConfig.from_pretrained(path)
        self.config._attn_implementation = attn_impl
        self.num_layers = num_layers or self.config.num_hidden_layers

        cfg, dev = self.config, self.device
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device="meta")
        self.embed.load_state_dict(
            {"weight": self.weights.get("model.embed_tokens.weight", dev, dtype)}, assign=True)

        self.norm = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to("meta")
        self.norm.load_state_dict(
            {"weight": self.weights.get("model.norm.weight", dev, dtype)}, assign=True)

        # tie_word_embeddings is False for QwQ-32B, but honour it anyway.
        head_name = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device="meta")
        self.lm_head.load_state_dict({"weight": self.weights.get(head_name, dev, dtype)},
                                     assign=True)

        self.rotary = Qwen2RotaryEmbedding(cfg).to(dev)

    # -- layer streaming ---------------------------------------------------
    def load_layer(self, idx: int) -> Qwen2DecoderLayer:
        """Build decoder layer `idx` with its weights resident on device."""
        with torch.device("meta"):
            layer = Qwen2DecoderLayer(self.config, idx)
        prefix = f"model.layers.{idx}."
        sd = {n[len(prefix):]: self.weights.get(n, self.device, self.dtype)
              for n in self.weights.prefix(prefix)}
        missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
        assert not missing, f"layer {idx} missing weights: {missing}"
        assert not unexpected, f"layer {idx} unexpected weights: {unexpected}"
        layer.eval()
        # load_state_dict(assign=True) wraps the loaded tensors in Parameters,
        # which default to requires_grad=True.  A caller that forgets no_grad
        # would then retain an autograd graph across the whole stream and hold
        # every layer's weights alive — 62 GB of them.
        layer.requires_grad_(False)
        return layer

    def position_embeddings(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        return self.rotary(hidden, position_ids)

    def embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids.to(self.device))

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden))
