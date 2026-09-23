# rANS — lossless BF16 weight compression, decoded inside the GEMM

Ported here so it can run against **vLLM 0.23 / transformers 5.5.4**
(`.venv-023`), which is the environment that supports `qwen3_5`
(Qwen3.5-27B). The original lives in the vllm-ascend checkout at
`../../awork`, and that is still where the operator is integrated.

Despite the directory name the production codec is **fixed-rate**, not rANS.
Per-symbol rANS was measured at ~1.6 cycles/weight and abandoned; it survives
only as the entropy-bound reference (10.54 bits/weight). Everything here uses
the fixed code: sign+mantissa raw, exponent as a 4-bit offset from a two-level
base — **12.33 bits/weight, lossless, 1.30x**.

## Status

Verified in this environment (torch 2.10.0, torch_npu 2.10.0.post4, vLLM 0.23):

| | |
|---|---|
| `tests/test_tiles.py` | 14 shapes bit-exact, incl. 27648x5120, 5120x27648, 17408x5120, 5120x17408 |
| `tests/test_gemm.py` | 10 shapes, within one bf16 ULP of `torch.matmul` |
| `e2e.py` on Qwen3-1.7B | 196/196 weights bit-identical, tokens identical, 1.297x smaller |

## What it costs

Measured on 910B4 (`msprof` device time, weights streaming from HBM):

* the fused op is **3.4x** a bf16 matmul (3.5x across QwQ-32B's real shapes)
* end to end on QwQ-32B that dilutes to **~1.6x slower per token**, because a
  decode step is not only projections
* weights **1.296–1.297x** smaller

**This buys capacity, not speed.** On one 910B4 (61 GiB usable) QwQ-32B in bf16
needs 65.5 GiB and OOMs; compressed it is 48.3 GiB and runs. That is the point.

Full measurements and the reasoning: `../../awork/tools/ans/CHECKPOINT.md`.

## Use

```bash
bash scripts/build.sh                       # kernel .so + torch binding
bash scripts/run.sh tests/test_tiles.py     # bit-exactness
bash scripts/run.sh tests/test_gemm.py      # fused GEMM vs torch.matmul
bash scripts/run.sh tests/bench_qwq_layer.py   # per-projection cost

.venv-023/bin/python rANS/e2e.py rANS/bind/librans.so \
    --model <hf dir> --mode both            # from the vllm repo root
```

`--mode ans` compresses on the CPU and moves only the planes to the NPU, which
is what lets a model bigger than HBM load at all. `--mode both` needs the bf16
model to fit, and is the only mode that can compare.

In code:

```python
import torch
torch.ops.load_library("rANS/bind/librans.so")
from rANS.ans_linear import swap_linears
n, orig, comp, checked, wrong = swap_linears(model)   # model still on CPU
model = model.to("npu:0")
```

## Qwen3.5-27B

Not runnable yet, and vLLM 0.23 is only half the reason:

* `.venv-023` does support the architecture (`qwen3_5` is in transformers 5.5.4;
  `.venv` at transformers 4.57.6 does not).
* **The weights are not on disk.** `models/Qwen3.8-27B`, `-L8` and `-L16` each
  contain config + tokenizer + `model.safetensors.index.json` and **zero**
  `.safetensors` shards; the index expects 51.7 GiB across 18 shards. Download
  them first.

When they land, note the model is multimodal and hybrid — most layers are
`linear_attention` (Mamba-style: `A_log`, `conv1d`, `dt_bias`) with
`full_attention` every 4th. `swap_linears` only touches `nn.Linear` in the
decoder blocks and `_decoder_blocks` already knows the
`model.language_model.layers` nesting these checkpoints use, but the linear-
attention projections have not been looked at. Run `tests/test_tiles.py` with
their shapes before trusting anything.

## Layout, and what is shared

```
kernels/ans_fixed_decode_gemm.cpp -> snapshot of ../../awork/csrc/kernels/
codec/fixed_codec.py              -> snapshot of ../../awork/tools/ans/
bind/binding.cpp                     GENERATED each build from awork's
                                     csrc/torch_binding.cpp, by line span
ans_linear.py                        standalone nn.Module (no vllm-ascend dep)
```

The kernel and codec are **snapshots, not the authority** — `awork/csrc/kernels/`
and `awork/tools/ans/` stay the live copies and the ones the shipped operator
uses, so they are never allowed to drift apart. They are checked in as real
files (byte-identical to `awork` at the time of writing) instead of as symlinks
into `awork`, because git records a symlink as nothing but its target path: a
symlink here would upload no code at all and dangle on every other checkout.
**awork changed -> re-copy; this changed -> port it back.**

`ans_linear.py` is the second deliberate duplicate — it mirrors
`_encode_fixed_gemm_layer` / `_fused_fixed_linear` in
`awork/vllm_ascend/quantization/methods/ans_bf16.py`, which is bound to
vllm-ascend's `AscendLinearScheme` and cannot be imported here. **Change one,
change the other.**

To cut the tie to `awork` entirely, point `scripts/gen_binding.sh` at a vendored
copy of `torch_binding.cpp`.

## Build note that will bite you

The include order here is the **opposite** of the `.venv` (torch 2.8) build.
torch_npu 2.10 bundles its own newer ACL headers under
`third_party/acl/inc` and includes them by explicit relative path; CANN 8.5's
`acl_base.h` otherwise wins the include guard and `aclmdlRITask` (declared in
torch_npu's `acl_base_rt.h`, absent from CANN 8.5) ends up undeclared. So
torch_npu's acl directory goes **first**. `scripts/build.sh` already does this.

Also: always pass `-DCMAKE_BUILD_TYPE`. Empty, CANN's `merge_mix_obj.sh` runs
`shift 2` with one argument left and dies under `set -e` printing nothing.
