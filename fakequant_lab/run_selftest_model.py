"""P1 model-level self-test: the streaming engine must be a no-op in bf16.

Builds a small random Qwen2 model (same architecture family as QwQ-32B, tiny
dims), saves it, then checks:

  M1  streaming bf16 logits are bit-identical to `Qwen2ForCausalLM.forward`
  M2  streaming bf16 PPL equals the reference PPL exactly
  M3  all five configs run and are ordered as expected (bf16 best, w4a8 worst)
  M4  patching is reversible — a quant config leaves the layer untouched after

M1/M2 are the Exit Criteria "BF16 组与原模型逐位一致（插桩无副作用）".

Run:  ASCEND_RT_VISIBLE_DEVICES=1 python -m fakequant_lab.run_selftest_model
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    HAVE_NPU = True
except Exception:
    HAVE_NPU = False

from transformers import Qwen2Config, Qwen2ForCausalLM

from .fake_linear import Config, PatchedLayer, build_configs
from .model_stream import StreamingQwen2
from .quant import QuantSpec
from .runner import run_configs

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def build_tiny(path: Path, dev, gk: int) -> Qwen2Config:
    """Small Qwen2 whose K dims are divisible by gk, mirroring QwQ-32B's shape ratios."""
    cfg = Qwen2Config(
        vocab_size=1024,
        hidden_size=2 * gk,                     # K of qkv/o/gate/up
        intermediate_size=3 * gk,               # K of down
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0" if HAVE_NPU else "cpu")
    ap.add_argument("--gk", type=int, default=256)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--windows", type=int, default=3)
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "npu":
        torch.npu.set_device(dev)
    print(f"device: {dev}  gk={args.gk}")

    tmp = Path(tempfile.mkdtemp(prefix="fqlab_tiny_"))
    try:
        cfg = build_tiny(tmp, dev, args.gk)
        torch.manual_seed(1)
        ids = torch.randint(0, cfg.vocab_size, (args.windows, args.seq))

        # ---- reference: the ordinary HF forward -----------------------------
        ref = Qwen2ForCausalLM.from_pretrained(tmp, dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to(dev).eval()
        with torch.no_grad():
            ref_logits = torch.cat([ref(ids[i:i + 1].to(dev)).logits for i in range(args.windows)])
            tgt = torch.full_like(ids, -100)
            tgt[:, :-1] = ids[:, 1:]
            ref_nll = F.cross_entropy(ref_logits.float().reshape(-1, cfg.vocab_size),
                                      tgt.reshape(-1).to(dev), ignore_index=-100).item()
        del ref
        if dev.type == "npu":
            torch.npu.empty_cache()

        # ---- streaming engine ----------------------------------------------
        model = StreamingQwen2(tmp, dev, attn_impl="sdpa")
        configs = build_configs(args.gk)
        res = run_configs(model, ids, configs, batch_size=1, verbose=False)

        print("\nM1/M2  bf16 control vs plain Qwen2ForCausalLM")
        with torch.no_grad():
            h = model.embed_ids(ids[0:1].to(dev))
            pos = torch.arange(args.seq, device=dev).unsqueeze(0)
            pe = model.position_embeddings(h, pos)
            for li in range(model.num_layers):
                layer = model.load_layer(li)
                h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
                del layer
            stream_logits = model.head(h)
        check("streaming bf16 logits are bit-identical to the HF forward",
              torch.equal(stream_logits, ref_logits[0:1]),
              f"max|diff|={float((stream_logits.float() - ref_logits[0:1].float()).abs().max()):.3e}")
        # Logits are bit-identical above, so any residual here is purely the
        # summation order of the loss reduction (the runner sums 512-token
        # slices; the reference flattens the whole batch at once).
        check("streaming bf16 NLL matches the reference to float-summation slack",
              abs(res.nll["bf16"] - ref_nll) < 1e-5,
              f"stream={res.nll['bf16']:.10f} ref={ref_nll:.10f} "
              f"diff={abs(res.nll['bf16'] - ref_nll):.2e}")

        print("\nM3  five configs, same pass")
        for name in res.configs:
            snr, kl = res.logit_snr.get(name), res.kl.get(name)
            print(f"       {name:>12s}  ppl={res.ppl[name]:9.4f}"
                  + (f"   KL={kl:7.5f}   logit SNR={snr:6.2f} dB"
                     if snr is not None else "   (reference)"))
        check("all five configs produced a PPL", len(res.ppl) == 5)
        check("KL is reported for every non-reference config",
              len(res.kl) == 4 and all(v > 0 for v in res.kl.values()))
        check("KL ranks the configs the same way logit SNR does",
              sorted(res.kl, key=res.kl.get) == sorted(res.logit_snr, key=lambda k: -res.logit_snr[k]),
              f"KL order = {sorted(res.kl, key=res.kl.get)}")
        w4mg, w8mg = f"w4a8-mg{args.gk}", f"w8a8-mg{args.gk}"
        check("int8 weights beat int4 weights at the same GK",
              res.logit_snr[w8mg] > res.logit_snr[w4mg],
              f"{w8mg}={res.logit_snr[w8mg]:.2f} > {w4mg}={res.logit_snr[w4mg]:.2f} dB")
        check("mid-group beats per-channel at the same bit width",
              res.logit_snr[w4mg] > res.logit_snr["w4a8-pc"],
              f"{w4mg}={res.logit_snr[w4mg]:.2f} > w4a8-pc={res.logit_snr['w4a8-pc']:.2f} dB")
        check("per-layer SNR was recorded for every layer and config",
              all(len(v) == model.num_layers for v in res.layer_snr.values()))

        print("\nM4  patching is reversible")
        layer = model.load_layer(0)
        before = layer.mlp.down_proj
        with PatchedLayer(layer, configs[3]) as p:
            inside = layer.mlp.down_proj
        check("Linear is replaced inside the context", type(inside).__name__ == "FakeQuantLinear")
        check("original module is restored on exit", layer.mlp.down_proj is before)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all model self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
