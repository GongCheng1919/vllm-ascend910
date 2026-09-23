# rANS weight compression — checkpoint (2026-09-16)

**Start here if you opened a session in `PureInt8LLMPretraining/vllm/`.**
The work lives in `rANS/`; read `rANS/README.md` next. The deep history,
every measurement and the reasoning behind each constant are in the
vllm-ascend checkout at `../awork/tools/ans/CHECKPOINT.md` — go there before
re-deriving anything.

## The one-paragraph version

Lossless BF16 weight compression, decoded on the AIV directly into the Cube's
GEMM panel. **12.33 bits/weight, 1.30x smaller, bit-exact.** It is **~3.4x
slower** than a bf16 matmul on device time, which dilutes to **~1.6x per token**
end to end. It buys **capacity, not speed**: QwQ-32B in bf16 needs 65.5 GiB and
OOMs on one 910B4 (61 GiB usable); compressed it is 48.3 GiB and runs.

Despite the directory name the production codec is **fixed-rate, not rANS**.
Per-symbol rANS costs ~1.6 cyc/weight and was abandoned; it survives only as the
entropy-bound reference (10.54 bits/weight). Do not benchmark it expecting
anything good.

## State

Everything below was verified in `.venv-023` (torch 2.10.0, torch_npu
2.10.0.post4, vLLM 0.23.0, transformers 5.5.4) on 910B4 + CANN 8.5.0.

| | |
|---|---|
| `rANS/tests/test_tiles.py` | 14 shapes bit-exact, incl. 27648x5120, 5120x27648, 17408x5120, 5120x17408 |
| `rANS/tests/test_gemm.py` | 10 shapes, within one bf16 ULP of `torch.matmul` |
| `rANS/e2e.py` on Qwen3-1.7B | 196/196 weights bit-identical, tokens identical, 1.297x |
| `rANS/tests/bench_qwq_layer.py` | 3.5–3.6x per decoder layer at QwQ-32B shapes |
| `rANS/tests/bench_qwen38_layer.py` | 3.32x per decode step at Qwen3.8-27B shapes, synthetic weights |
| `rANS/e2e.py` on Qwen3.8-27B | 400/400 weights bit-identical, tokens identical, 51.0 -> 41.0 GiB |
| `rANS/e2e.py --head` | 401/401 bit-identical with lm_head compressed, 40.1 GiB |
| `rANS/tests/capacity_qwen38.py` | 77824 -> 161792 tokens of context on one 910B4, 2.08x |

```bash
bash rANS/scripts/build.sh                     # ~2 min
bash rANS/scripts/run.sh rANS/tests/test_tiles.py
.venv-023/bin/python rANS/e2e.py rANS/bind/librans.so --model <hf dir> --mode both
```

## Numbers you should not re-measure

msprof device time, 5120x5120, M=1, weights streaming (nothing L2-resident):

```
torch bf16 matmul     44.8 us    1.00x    (1156 GB/s -- at peak)
decode only          134.7 us    3.01x    vec 106.1 us, vec ratio 81%
fused decode + GEMM  151.7 us    3.39x    vec 106.1 us, vec ratio 71%
```

QwQ-32B end to end, one 910B4, truncated to 40 layers so bf16 fits:

```
bf16   37200 MiB   11.2 tok/s
ANS    28708 MiB    7.1 tok/s    1.58x slower, 1.296x smaller
```

Full 64 layers: bf16 **OOM**; ANS 48.3 GiB resident, 4.3 tok/s.

## Why there is no more performance to get

The kernel is **vector-issue bound** and the floor is 106.1 us (2.37x). Every
other pipe is already hidden — confirmed by switching each off via the `dbg`
field in the tiling head and re-measuring. Measured and worth **zero**:

* deeper AIV/Cube ping-pong (NSLOT 2 -> 4)
* deeper plane prefetch (NBUF_IN 2 -> 4)
* interleaving two quarters on separate scratch
* **shortening the quarter ops from 2048 to 512 elements** — ~98% of a vector
  op's cost is independent of its length, so batching tiles into longer vectors
  buys nothing

Worth ~10%: Level-0 ops with the vector mask hoisted (kept). The only remaining
lever is a codec needing fewer than ~48 vector instructions per 8192-weight
tile, and the sign/exponent/mantissa split is already close to minimal.

## Qwen3.8-27B — was blocked, and not on the vLLM version

vLLM 0.23 was only half the problem. The other half — the weights — is being
fixed as of 2026-09-17; see the synthetic-throughput section below.

* Architecture support is fine: `qwen3_5` is in transformers 5.5.4 (`.venv-023`).
  `.venv` at 4.57.6 does not have it.
* **The weights were not on disk.** `models/Qwen3.8-27B`, `-L8`, `-L16` each held
  config + tokenizer + `model.safetensors.index.json` and **zero** `.safetensors`
  shards. The index wants **51.7 GiB across 18 shards**, and the upstream repo is
  `Qwen/Qwen3.8-27B` (rev `1d4bf0f2ff60`) — identical config and shard list.
  `models/dl_qwen38.sh` pulls it from **hf-mirror.com with the proxy unset**;
  it retries in a loop, so re-run it to resume. Two things it has to get right:
  `HF_ENDPOINT=https://hf-mirror.com` and **`HF_HUB_DISABLE_XET=1`** — the mirror
  does not proxy Xet, so `hf download` otherwise reaches the real
  `cas-server.xethub.hf.co` and dies with a 401 on every shard.

Its projection shapes (hidden 5120, ffn 17408) are already covered bit-exactly
by `test_tiles.py`. But it is multimodal and hybrid — most layers are
`linear_attention` (Mamba-style: `A_log`, `conv1d`, `dt_bias`), `full_attention`
every 4th. `swap_linears` only touches `nn.Linear` and `_decoder_blocks` already
handles the `model.language_model.layers` nesting, but **the linear-attention
projections have not been looked at**. Check their shapes before trusting a run.

## Two build traps, both silent

1. **Always pass `-DCMAKE_BUILD_TYPE`.** Empty, CANN's `merge_mix_obj.sh` runs
   `shift 2` with one argument left and dies under `set -e` **printing nothing**
   — surfacing only as a bare `Error 1` on the `*_merge_obj` target.
2. **Include order here is the OPPOSITE of the `.venv` (torch 2.8) build.**
   torch_npu 2.10 bundles newer ACL headers under `third_party/acl/inc` and
   includes them by explicit relative path; CANN 8.5's `acl_base.h` otherwise
   wins the include guard and `aclmdlRITask` (declared in torch_npu's
   `acl_base_rt.h`, absent from CANN 8.5) ends up undeclared. torch_npu's acl
   directory goes **first**. `scripts/build.sh` does this already.

Also, on any per-layer custom op: **check the host path before the kernel.**
Three host bugs here each cost more than every kernel change combined —
`int(npu_tensor[i])` in the hot path (4 device syncs per call, ~0.48 ms per
linear), rebuilding the Cube tiling per call, and `at::zeros` where `at::empty`
is correct.

## What to do next

The engineering is finished; the honest next step is **writing it up as a
capacity result**, not more kernel work.

1. Check the prior art first — exponent-only coding is probably DFloat11 /
   NeuZip at a similar ~1.5x. What is ours is the accelerator-level budget
   (0.13 cyc/weight against a 0.25 floor), the tuple-amortised rANS, and the
   bit-band finding.
2. When the Qwen3.8-27B shards land, `rANS/e2e.py --mode ans` is ready.
3. Known and unfixed: `GemmPanel` re-reads the whole B panel for every 128-row
   M block, so prefill at large M is worse than it needs to be (M=4096 reads B
   32 times). Never measured; flagged only.

## Qwen3.8-27B — synthetic throughput, 2026-09-17

Weights are downloading (`models/dl_qwen38.sh`, hf-mirror, Xet off). Timing does
not need them: `rANS/tests/bench_qwen38_layer.py` runs the real per-layer shapes
(read out of the published index by HTTP range request on the safetensors
headers, no shards) against synthetic weights.

**Timing is independent of the weight values, and that is measured, not assumed**
(17408x5120, one shape, five distributions):

```
weights                      wide%   MB ans  bits/w   ans us
normal s=0.02 (weight-like)  0.058    131.0   12.33    492
normal s=1.0                 0.056    131.0   12.33    491
all zeros                    0.000    130.8   12.31    482
heavy tail (cauchy)          1.303    135.5   12.76    840
uniform random BITS        100.000    492.1   46.31  29987
```

The only data dependence is the **wide-group fraction**, and a Gaussian of any
scale lands on the same 0.058% that real weights give. So benchmark with
`rng.normal`, never with random *bits* — those are the pathological case the
encoder is meant to refuse, and they cost 61x. The bench prints `wide%` per
shape so a synthetic run says when it is lying.

### The step, M=1 (48 linear-attn + 16 full-attn layers + lm_head)

```
whole decode step   47.7 GiB -> 36.8 GiB   bf16 44.6 ms -> ans 163.4 ms   3.66x
                                           of which 69.3 ms is HOST launch
linears only tok/s  bf16 22.4 -> ans 6.1
```

3.66x matches QwQ-32B (`bench_qwq_layer.py` re-run today: 3.59x), so the hybrid
architecture costs nothing in itself. 1.297x smaller, 12.33 bits/weight, same as
everywhere else.

### The new finding: a ~140 us HOST floor per call

`ans_fixed_decode_gemm` costs **~140 us of host time per call regardless of
shape**, and below ~4096 rows that is the whole cost — the device finishes
first. Measured by timing the launches before any sync:

```
N=   64 K=5120   enqueue 124.6 us   total 126.2 us   host-bound
N= 1024 K=5120   enqueue 123.4 us   total 125.0 us   host-bound
N=17408 K=5120   enqueue 133.7 us   total 493.4 us   device-bound
```

Qwen3.8-27B issues **497 linear calls per decode step** (QwQ issues 448 but they
are all large), so this is 69 of the 163 ms. It overlaps device work on the big
projections, so removing it entirely is worth ~10% overall — but on the small
ones it is 100% of the cost:

* `linear_attn.in_proj_a` / `in_proj_b` are **48x5120**. Compressing them saves
  45 MB out of 47.7 GiB and costs 96 x 142 us = **13.6 ms per step, 8%**.
  `swap_linears` has no size threshold; it should skip anything under ~8 MiB.
* `self_attn.k_proj` / `v_proj` (1024x5120) are host-bound too, 4.4 ms for
  74 MB saved. Borderline.

This is the largest remaining lever for this model and it is host-side, not
kernel-side — the same lesson as the three host bugs above, one level down.

**Fixed**: `swap_linears` now takes `min_bytes` (default `MIN_BYTES` = 4 MiB) and
`ans_bf16.py` takes `VLLM_ANS_MIN_MIB` (default 4, 0 disables). Below that the
weight stays bf16. Re-measured with the threshold on:

```
                     compressed   decode step    ratio
no threshold         45.36 GiB    163.4 ms       3.66x
4 MiB threshold      45.31 GiB    152.0 ms       3.32x
```

7% of the step for 45 MB of the 45 GiB. It excludes exactly the 96
`linear_attn.in_proj_a` / `in_proj_b` layers and nothing else.

### What actually gets compressed — verified, not assumed

Built from the config on the **meta device** (no weights, no HBM) and walked the
real `swap_linears` loop:

```
nn.Linear in the whole model: 607   of which under model.visual: 110
swap_linears replaces:        400,  45.31 of 51.7 GiB
```

**The vision tower was already excluded and always has been** — `_decoder_blocks`
returns `model.language_model.layers`, so the 110 visual linears, the merger
(40.5 + 45.0 MB), `lm_head` and `embed_tokens` are never reached. No change was
needed to keep the vision encoder in float; it never was anything else.

Closing the open question from above: the linear-attn projections are ordinary
`nn.Linear` (`in_proj_qkv` 10240x5120, `in_proj_z` 6144x5120, `out_proj`
5120x6144, `in_proj_a`/`b` 48x5120) and all pad cleanly to (64, 128) tiles.
`conv1d` is (10240, 1, 4), an `nn.Conv1d`, so it is left alone — correct.
`A_log`/`dt_bias` are 1-D.

`lm_head` is 248320x5120 = 2.4 GiB and `SKIP_NAMES` drops it. At M=1 it takes
6.5 ms either way (ANS and bf16 tie exactly), so compressing it would buy
0.55 GiB for free — the one place the skip list is leaving something on the
table. The bench counts it as compressed; the shipping code does not.

### End to end on the real weights — PASS, 2026-09-17

`e2e.py --mode both`, one 910B4, text-only prompt, 64 layers, nothing truncated:

```
                       bf16        ANS
HBM resident          51.0 GiB    41.0 GiB     (45.3 -> 34.9 GiB of linears)
CPU -> device            78 s        17 s
forward 1x1            975.7 ms   1117.2 ms    1.15x
forward 1x128         1038.6 ms   1104.9 ms    1.06x
forward 1x512         1243.0 ms   1291.6 ms    1.04x
generate 16 tok         3.4 tok/s   2.5 tok/s
```

```
decoded weights bit-identical: 400/400 checked
tokens identical:              True
logits max|delta|:             0.15625  (8.5e-03 of max|logit|, one bf16 ULP)
[status] PASS
```

CPU-side compression of the 400 linears took **544 s**; it happens before
anything reaches the device, so a model too big for HBM still loads.

**Neither end-to-end ratio here measures the codec.** The whole decode step is
host-bound and the NPU is idle for most of it, on both sides. Measured:

```
decode step:  enqueue 207.7 ms   total 207.9 ms    <- 99.9% host
memory-bound floor, 51.0 GiB at 1.1 TB/s: 46.4 ms  (22 tok/s)
```

The host takes as long to *issue* the step as the step takes to finish. So
Qwen3.8-27B runs at **4.8 tok/s where the hardware floor is 22 tok/s**, and
that gap is python and aten dispatch, not bandwidth and not this codec.

One aten op costs **~21 us to issue** on this stack, whatever its size (a
1x5120 `add` is 21.1 us; a 5120x5120 `matmul` is 24.0 us). 207.7 ms of issue
time is therefore about **9,900 aten ops per decode step**, ~155 per layer.
Host time to issue each block type:

```
Qwen3_5GatedDeltaNet   48 calls   120.2 ms   2505 us/call   (~120 ops/layer)
Qwen3_5Attention       16 calls    29.0 ms   1810 us/call
Qwen3_5MLP             64 calls    15.9 ms    249 us/call   (3 linears, near floor)
```

**Correction to an earlier reading in this file.** The bf16->ANS `generate`
delta of 106 ms was noted as matching the shape bench's 106.2 ms of *device*
time. That agreement is a coincidence: the step never becomes device-bound, so
what the model actually pays for ANS is host time — ~140 us to issue the fused
op against ~24 us for a matmul, plus the four extra aten ops in
`AnsFusedLinear.forward` — roughly 400 x 116 us + 400 x 4 x 21 us = 80 ms.
Same order, different cause. **The codec's real cost is the 3.32x device-time
figure from `bench_qwen38_layer.py`; the capacity result is unaffected because
bytes are bytes.**

Quote the generate ratio (1.36x) only with that caveat attached, and never the
forward ratio at all — they measure different code paths and only one of them
is even self-consistent.

`flash-linear-attention` / `causal-conv1d` are not installed, so the 48
`linear_attention` layers run transformers' pure-torch fallback. **Installing
them cannot help on this hardware**: `is_flash_linear_attention_available()` and
`is_causal_conv1d_available()` both start with `is_torch_cuda_available()`, which
is False on NPU, and both packages are CUDA kernels anyway. The Ascend
equivalent exists only inside vllm-ascend (`vllm_ascend/ops/triton/fla/`,
`patch/worker/patch_qwen3_5.py`), i.e. under vLLM, not under HF transformers.

Measured per layer at the real config (`probe_gdn.py` pattern, random weights):

```
seq=  1  linear_attn (chunk path)    26.59 ms/layer   x48 = 1276.4 ms
seq=128  linear_attn (chunk path)    26.61 ms/layer   x48 = 1277.1 ms  <- length-independent

torch_chunk_gated_delta_rule,     seq=1   18.32 ms   x48 = 879.3 ms
torch_recurrent_gated_delta_rule, seq=1    1.21 ms   x48 =  58.2 ms
full_attention layer,             seq=1    3.01 ms   x16 =  48.2 ms
```

`torch_chunk_gated_delta_rule` runs a **63-iteration python loop** of small
tensor ops; seq=1 and seq=128 cost the same, so it is pure launch overhead.
That is the whole story of the forward column:

* `model(ids)` with **no cache** takes the CHUNK path -> 879 of the 975.7 ms
  "1x1 forward" is that loop. The 1.15x is measured against it. **Discard it.**
* `generate` takes the RECURRENT path (1.21 ms/layer). bf16 294 ms/token, ANS
  400 ms/token, **delta 106 ms** against the 106.2 ms the shape bench predicts
  for the linears alone. That agreement is the reason to trust both numbers.

So end to end is 1.36x, on a baseline where the linears are 45.8 of 294 ms
(16%). Fix the fallback and the ratio rises toward the 3.32x of
`bench_qwen38_layer.py`. Also note `generate` here is a single 16-token run with
no repeats — treat its tok/s as +/-10% (two runs of the same config gave 3.4 and
3.6 tok/s for bf16).

**The capacity result is the real one, and this model states it differently from
QwQ-32B.** QwQ was OOM-vs-runs. Qwen3.8-27B fits either way on a 61 GiB card,
so what compression buys is context. Measured, `tests/capacity_qwen38.py`,
prompt grown 2048 tokens at a time carrying the cache:

```
         weights     context     peak
bf16       51.0G       77824    58.2G
ans        40.6G      161792    55.2G
context 2.08x   whole model 1.255x smaller (linears alone are 1.297x)
```

**2.08x the context on the same card.** KV measures 61 KB/token, which is what
the config says (only 16 of 64 layers are full attention, 4 kv heads x 256).
Both modes died on a transient chunk-path activation rather than on the KV
itself, and the ANS run died with 8 GiB allocated but fragmented away
(`1.08 GiB free`, wanted 1.88), so 2.08x is if anything the conservative read.

Segments, not one long forward, on purpose: without the FLA fast path a single
long prefill materialises `[1, 48, seq/64, 64, 64]` float32 tensors and the run
would measure the fallback's activations instead of the model's KV.

### lm_head: free capacity, now verified

`swap_linears(include_head=True)` / `e2e.py --head` also compresses the
248320x5120 head. Measured: **401/401 bit-identical**, tokens identical,
48825 -> 37633 MiB, HBM 40.1 GiB. That is 0.54 GiB more than without it, about
9000 more tokens of context, and at M=1 the fused op and the bf16 matmul tie
exactly (6501 vs 6502 us) so it costs nothing.

Off by default for one reason: **a tied head must never be compressed** --
`embed_tokens` is read by index, not by GEMM, and they share a Parameter.
`_head_module` refuses a tied head. Note the trap in that check: **every meta
tensor reports `data_ptr() == 0`**, so a `data_ptr` comparison marks an untied
head on a meta build as tied. Compare the Parameter identity first (transformers
ties by assigning the same object) and only fall back to `data_ptr` on real
tensors.

### One more trap: the auto class

`AutoModelForCausalLM.from_pretrained` **fails** on this checkpoint — it picks
the text-only `Qwen3_5ForCausalLM`, hands it the composite config and dies with
`'Qwen3_5Config' object has no attribute 'vocab_size'`. It needs
`AutoModelForImageTextToText` (i.e. `Qwen3_5ForConditionalGeneration`);
text-only generation on it is fine. `e2e.py` now picks the class off
`config.architectures`.

## vLLM is blocked on CANN, not on us — 2026-09-17

The HF numbers above are dispatch-bound, so the fix is vLLM (graph capture plus
the Ascend GDN kernels). It does not run here. **`vllm_ascend` 0.23.0 needs a
newer CANN than the 8.5.0 installed**, and that is three separate gaps, found in
this order:

1. `libvllm_ascend_kernels.so` will not `dlopen`: one undefined symbol,
   `rtFunctionGetMetaInfoSize` (the other 22 rt/acl imports resolve). Without
   it `torch.ops._C_ascend` never registers, and vllm_ascend calls into that
   namespace from **149 places with essentially no fallback**, so the engine
   dies on `npu_gemma_rms_norm`.
   *Worked around*: `rANS/scripts/rt_metainfo_shim.c`. Disassembly shows the
   symbol has exactly one caller, `vllm_ascend::mla_preprocess_impl` (DeepSeek
   MLA), which Qwen3.5 never reaches. The stub **aborts** if called rather than
   returning a fabricated size.
2. `aclnnAddRmsNormBias` is not in CANN 8.5.0's `libopapi.so`. Reached from the
   `norm_quant` graph-fusion pass.
   *Worked around*: `additional_config={"ascend_compilation_config":
   {"fuse_norm_quant": False}}` — note vllm-ascend reads its fusion switches
   from **its own** `additional_config`, not from vLLM's
   `compilation_config.pass_config`, and `custom_ops: ["none"]` does not
   disable `forward_oot` either. The pass only fuses a norm with a
   *quantisation*, so bf16 loses nothing.
3. **`aclnnCausalConv1d` does not exist in CANN 8.5.0 at all** — `libopapi.so`
   has no CausalConv1d op of any name. This is the GDN conv, on the model's
   core path, with no switch to route around it. **Stop here.**

So the route to a real end-to-end number is a **CANN upgrade**, and that has to
be weighed against re-validating the ANS kernel build against the new CANN.
The site-packages edit from (2) has been reverted; nothing outside this repo is
left modified. The shim is source in `rANS/scripts/` and only does anything
when `LD_PRELOAD`-ed.

Do not read the "custom fusions enabled" banner as proof the fusions ran, and
do not read `enable_custom_op() == True` as proof the kernels work — it only
means the library loaded.

## Layout

```
rANS/kernels/ans_fixed_decode_gemm.cpp   snapshot of ../awork/csrc/kernels/
                                         (a real file: git stores a symlink as
                                         its target path and nothing else, so a
                                         symlink here uploads no code)
rANS/codec/fixed_codec.py                snapshot of ../awork/tools/ans/
rANS/bind/binding.cpp                     generated each build from awork's
                                          csrc/torch_binding.cpp, by line span
rANS/ans_linear.py                        standalone nn.Module, MIRRORS
                                          awork/vllm_ascend/quantization/
                                          methods/ans_bf16.py -- change both
```

The awork copies stay the authority, and they are what the integrated operator
builds from, so the two must never drift: awork changed -> re-copy; this changed
-> port it back. Keeping these as snapshots (rather than symlinks into awork) is
what makes a clone of this repo self-contained.
