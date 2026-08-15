# fakequant_lab — P1 fake-quant harness for mid-group W4A8

Algorithm-side counterpart to `int4_cube_lab/` (which is the kernel side).
Implements `MID_GROUP_ROADMAP.md` §5.  Status and findings live in
`MGCKPT/P1_HARNESS.md`; this file is how to run it.

Environment: `export ASCEND_RT_VISIBLE_DEVICES=1` (device 0 is wedged).
All commands run from `vllm/`.

## What it does

Five configs over the same weights, in one pass over the checkpoint:

| config | W | A |
|---|---|---|
| `bf16` | — | — | *(control; runs the unpatched modules)* |
| `w8a8` | int8 per-channel | int8 per-token |
| `w8a8-mg{GK}` | int8 mid-group | int8 mid-group |
| `w4a8-mg{GK}` | **int4 mid-group** | **int8 mid-group** |
| `w4a8-pc` | int4 per-channel | int8 per-token |
| `w4a8-mg{GK}-asym` | int4 mid-group + zero point | int8 mid-group | *(opt-in, `--asym`, not the frozen spec)* |

Two arithmetic modes.  The *quantization algorithm* is identical in both — bit
width, group boundaries, **bf16** scales, zero point — which is what determines
model quality:

- **`dequant` (default)** — ordinary fake quant: dequantize to bf16, one bf16
  matmul.  Computationally a plain bf16 model (1.1–2.5x bf16), so it decodes at
  normal speed and is what task evaluation uses.
- **`kernel`** — reproduces the AscendC kernel's grouped integer arithmetic
  exactly (checked against a CPU int64 reference, `run_selftest.py` T4).  Costs
  8–19x bf16, so it is a verification tool, not the default.

The two agree to 0.04–0.17 percentage points of ΔPPL on the full WikiText-2 test
split, against effect sizes of 1–9 points — see `MGCKPT/P2_ACCURACY.md` §2.4.

## Commands

```bash
export ASCEND_RT_VISIBLE_DEVICES=1

# primitives: quantizer, MSD split, GEMM exactness, export layout   (~30 s)
python -m fakequant_lab.run_selftest

# engine: streaming bf16 must be bit-exact with Qwen2ForCausalLM     (~1 min)
python -m fakequant_lab.run_selftest_model

# Tier 1 (single tensor) + Tier 2 (single GEMM) SNR, per layer per Linear
python -m fakequant_lab.run_tiers --gk 1024 --windows 4 \
    --out fakequant_lab/results/tier12_gk1024.csv

# WikiText-2 perplexity, all configs in one weight pass
python -m fakequant_lab.run_ppl --gk 1024 --windows 0 \
    --out fakequant_lab/results/ppl_gk1024.json
```

```bash
# GPTQ calibration (C4, 128 windows); resumable, ~3-4 h for QwQ-32B
python -m fakequant_lab.run_gptq --gk 1024 --asym --calib c4 --nsamples 128 \
    --out fakequant_lab/gptq_gk1024_asym

# evaluate calibrated weights
python -m fakequant_lab.run_ppl --windows 0 \
    --gptq w4a8-mg1024-asym-gptq=fakequant_lab/gptq_gk1024_asym \
    --out fakequant_lab/results/ppl_gptq_full.json

# downstream tasks via lm_eval (needs >= 2 cards: 62 GB resident)
ASCEND_RT_VISIBLE_DEVICES=2,3 python -m fakequant_lab.run_tasks \
    --config w4a8-mg1024-asym --tasks glue --limit 200 \
    --out fakequant_lab/results/tasks_asym.json
```

Useful flags: `--layers N` (first N layers only, for smoke runs), `--windows N`
(0 = whole split), `--mode kernel` (verification arithmetic, 8–19x slower),
`--states-device cpu` (when hidden states for all configs will not fit in HBM),
`--asym`, `--gk-sweep 128,512,1024`.

## Metrics, and which ones to trust

`run_ppl` reports PPL, **KL(bf16 ‖ config)**, logit SNR, and per-layer hidden-state
SNR.  Gate on **PPL and KL**.  Logit SNR and per-layer SNR read far worse than the
model behaves — `w8a8` scores 13.3 dB on logits while its PPL is unchanged, because
most of the error is a common-mode logit shift that softmax cancels.  The absolute
32/35 dB Tier thresholds inherited from the INT8 study do not transfer to an
off-the-shelf QwQ-32B either; see `reports/midgroup/P1_harness.md` §3.4.  Tier 1/2
remain useful for *locating* the problem layer, not for passing or failing one.

## Files

| file | what |
|---|---|
| `quant.py` | grouped symmetric/asymmetric quantizer, bf16 scales, `w_ksum`, MSD split, group-major export |
| `fake_linear.py` | `FakeQuantLinear` + the config table + `PatchedLayer` context manager |
| `model_stream.py` | one-decoder-layer-at-a-time loading of a sharded Qwen2 checkpoint |
| `runner.py` | advances all configs in lockstep; PPL + per-layer (Tier 3) SNR |
| `run_tiers.py` | Tier 1 / Tier 2 SNR export to CSV |
| `run_ppl.py` | WikiText-2 perplexity to JSON |
| `gptq.py` | GPTQ: inverse-Hessian error compensation, shared `group_params` so the bf16-scale rule is the same |
| `run_gptq.py` | sequential GPTQ over the streamed model; C4 calibration; resumable |
| `resident.py` | whole model resident across cards, patched in place — an ordinary `Qwen2ForCausalLM` you can `generate()` with |
| `run_tasks.py` | lm_eval driver (GLUE / GSM8K / C-Eval / CMMLU) |
| `run_awq_probe.py` | per-input-channel scaling probe (P1 §4b: worth only 0.3–0.5 dB here) |
| `run_selftest*.py` | the self-tests above |

## Why layer streaming instead of `device_map`

QwQ-32B is 62 GB in bf16 and we need five configs over it.  One decoder layer
on device costs ~1 GB, so all five configs' hidden states share one card and one
pass over the weights.  It also yields per-layer SNR for free and avoids relying
on accelerate's NPU placement.
