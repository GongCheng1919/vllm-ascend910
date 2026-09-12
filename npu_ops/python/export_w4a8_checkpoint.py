"""Export a BF16 HF checkpoint as a mid-group W4A8 checkpoint vLLM can load directly.

WHY THIS EXISTS.  Until now the W4A8 arm converted BF16 -> int4 inside
`process_weights_after_loading`, i.e. AFTER vLLM has already constructed the
whole model in BF16.  Peak memory is therefore the BF16 footprint (65.5 GB for
QwQ-32B), not the W4A8 one (19.0 GB), which:

  * makes single-card 64-layer load impossible -- it OOMs at `lm_head`, short by
    ~70 MB (results/p9_singlecard_l64.log);
  * leaves ~7.8 GiB of allocator holes, so only 29% of the bytes the quantiser
    saves come back as KV cache (results/p9_singlecard_l48.log);
  * costs 679 s of conversion on every engine start.

All three are the same root cause.  A checkpoint that is already int4 on disk
lets vLLM allocate int4-shaped parameters from the start and none of them happen.

LAYOUT.  Quantisation is per-projection, NOT on vLLM's fused `qkv_proj` /
`gate_up_proj`.  That is exact, not an approximation: groups run along K and every
scale/zero is per-output-row, so fusing along N cannot move a group boundary.
`--self-check` re-verifies it bitwise on the real tensors rather than trusting
the argument.

Scale-like tensors are stored TRANSPOSED as [N, G] even though the kernel wants
[G, N].  That is deliberate: [N, G] fuses along dim 0 exactly like the weight, so
vLLM's stock fused/sharded weight loaders work unchanged, and row-parallel K
sharding becomes a dim-1 slice.  The linear method transposes once at load.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4a8_ops as W  # noqa: E402

# The four projections the kernel takes over.  Everything else -- embeddings,
# lm_head, norms, and the q/k/v biases -- stays BF16, exactly as the runtime
# converter left it, so the two paths quantise the SAME set of layers.
QUANT_SUFFIXES = (
    ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
)


def is_quant_target(key: str) -> bool:
    return key.startswith("model.layers.") and key.endswith(QUANT_SUFFIXES)


def unpack_int4(packed: torch.Tensor, n_elems: int) -> torch.Tensor:
    """Inverse of `w4a8_ops.pack_int4`; even element in the LOW nibble."""
    b = packed.to(torch.int16) & 0xFF
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    out = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
    out = torch.where(out >= 8, out - 16, out)          # 4-bit two's complement
    return out[..., :n_elems].to(torch.int8)


def dequant_snr(w: torch.Tensor, wq, ws, wk, wz, group: int) -> float:
    """SNR(dB) of `ws * (code - wz)` against the original weight.

    Reconstructs through the PACKED tensor, so a packing bug shows up here and
    not only at the kernel seam.
    """
    N, K = w.shape
    G = W.num_groups(K, group)
    stride = -(-group // W.CUBE_K_ELEMS) * W.CUBE_K_ELEMS
    codes = unpack_int4(wq, W.k_pad_elems(K, group)).float()
    rec = torch.zeros(N, K)
    for g in range(G):
        real = W.group_real(g, K, group)
        c = codes[:, g * stride:g * stride + real]
        rec[:, g * group:g * group + real] = (
            c - wz[g].float().unsqueeze(1)) * ws[g].float().unsqueeze(1)
    err = (w.float() - rec)
    return 10 * torch.log10(w.float().pow(2).sum() / err.pow(2).sum().clamp_min(1e-30)).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/QwQ-32B")
    ap.add_argument("--dst", default="models/QwQ-32B-W4A8-MG")
    ap.add_argument("--group", type=int, default=W.GROUP)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    ap.add_argument("--layers", type=int, default=0,
                    help="export only the first N layers (smoke test); 0 = all")
    ap.add_argument("--self-check", type=int, default=2,
                    help="verify fused==concat and report dequant SNR on this many tensors")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)
    idx_path = os.path.join(src, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]

    # Group keys by the shard that holds them so each source file opens once.
    by_file: dict[str, list[str]] = {}
    for k, f in weight_map.items():
        by_file.setdefault(f, []).append(k)

    def keep(key: str) -> bool:
        if args.layers and key.startswith("model.layers."):
            return int(key.split(".")[2]) < args.layers
        return True

    out: dict[str, torch.Tensor] = {}
    shards: list[tuple[str, dict]] = []
    index: dict[str, str] = {}
    nbytes = 0
    n_quant = n_pass = 0
    checked = 0
    t0 = time.perf_counter()
    limit = args.shard_gb * 1e9

    def flush(final: bool = False) -> None:
        nonlocal out, nbytes
        if not out or (not final and nbytes < limit):
            return
        name = f"model-{len(shards) + 1:05d}.safetensors"
        shards.append((name, out))
        for k in out:
            index[k] = name
        save_file(out, os.path.join(dst, name), metadata={"format": "pt"})
        print(f"[export] wrote {name}  {len(out)} tensors  {nbytes/1e9:.2f} GB", flush=True)
        out, nbytes = {}, 0
        gc.collect()

    for fname in sorted(by_file):
        with safe_open(os.path.join(src, fname), framework="pt") as f:
            for key in sorted(by_file[fname]):
                if not keep(key):
                    continue
                t = f.get_tensor(key)
                if not is_quant_target(key):
                    out[key] = t.contiguous()
                    nbytes += t.numel() * t.element_size()
                    n_pass += 1
                    continue
                w = t.to(torch.bfloat16)
                wq, ws, wk, wz = W.quantize_weight(w, args.group)   # ws/wk/wz are [G,N]
                if checked < args.self_check:
                    snr = dequant_snr(w, wq, ws, wk, wz, args.group)
                    print(f"[check] {key}  {tuple(w.shape)}  dequant SNR = {snr:.2f} dB", flush=True)
                    checked += 1
                base = key[: -len(".weight")]
                # Stored [N, G]: fuses along dim 0 like the weight itself.
                for suf, v in (("weight_packed", wq), ("weight_scale", ws.T.contiguous()),
                               ("weight_zero", wz.T.contiguous()), ("weight_ksum", wk.T.contiguous())):
                    out[f"{base}.{suf}"] = v
                    nbytes += v.numel() * v.element_size()
                n_quant += 1
                if n_quant % 32 == 0:
                    print(f"[export] {n_quant} projections quantised "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
                flush()
                del w, t
    flush(final=True)

    json.dump({"metadata": {"total_size": sum(
        os.path.getsize(os.path.join(dst, n)) for n, _ in shards)},
        "weight_map": index},
        open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)

    # config.json + tokenizer, with the quant block the loader dispatches on.
    cfg = json.load(open(os.path.join(src, "config.json")))
    if args.layers:
        cfg["num_hidden_layers"] = args.layers
    cfg["quantization_config"] = {
        "quant_method": "midgroup_w4a8",
        "group_size": args.group,
        "weight_bits": 4,
        "activation_bits": 8,
        "sym": False,
        "packed_layout": "mg_kgeom_v1",
        "ignored_layers": ["lm_head"],
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for aux in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "added_tokens.json", "special_tokens_map.json", "generation_config.json",
                "configuration.json"):
        p = os.path.join(src, aux)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, aux))

    total = sum(os.path.getsize(os.path.join(dst, n)) for n, _ in shards)
    print(f"\n[export] done in {time.perf_counter()-t0:.0f}s")
    print(f"[export] {n_quant} projections quantised, {n_pass} tensors passed through")
    print(f"[export] {len(shards)} shards, {total/1e9:.2f} GB -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
