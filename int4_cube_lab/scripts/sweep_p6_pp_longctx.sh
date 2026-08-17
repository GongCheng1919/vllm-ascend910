#!/usr/bin/env bash
# P6 · long context under PP, on the REAL model (PP=4, 64 layers, 16/rank).
#
# WHY A SEPARATE SWEEP.  Everything else in P6 ran at prompt_len=128, which says
# nothing about long context -- and long context splits into two regimes that
# move in OPPOSITE directions for us:
#
#   decode with a long KV : M stays = batch (tiny).  The GEMM cost is unchanged,
#                           but attention grows with context, so our GEMM win
#                           gets DILUTED -- the ratio should sag toward 1.0.
#   prefill of a long prompt : M = the whole prompt (thousands).  That is the
#                           large-M regime where mid-group pays its structural
#                           flush tax (P5) and loses to W8A8.  Expect us to LOSE.
#
# Reporting one blended number would hide both.  So this sweep reports them
# separately, and the bench is built to keep them apart:
#   * `step_us` warms up on the SAME prompts it then times, and prefix caching is
#     on, so the timed pass reuses the prefix KV -- it is a DECODE measurement
#     even at plen=8192.  That is deliberate, not an oversight.
#   * `prefill_ms` (--prefill-probe) is a separate max_tokens=1 pass over UNSEEN
#     prompts, so it pays a real prefill.
#
# Device 0 is wedged, so ranks are 1..4.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${OUT:-int4_cube_lab/results/p6_pp_longctx.csv}
LOG=${LOG:-int4_cube_lab/results/p6_pp_longctx.log}
ARMS=${ARMS:-"bf16 w8a8 w4a8 w8a8-native w4a8-native"}
PLENS=${PLENS:-"128 1024 4096 8192"}
BATCHES=${BATCHES:-"1 8"}
PP=${PP:-4}
LAYERS=${LAYERS:-64}

: > "$LOG"
devs=$(seq 1 "$PP" | paste -sd,)
echo "[sweep] out=$OUT arms='$ARMS' plens='$PLENS' batches='$BATCHES' pp=$PP L=$LAYERS devs=$devs"

for arm in $ARMS; do
  echo "[sweep] === arm=$arm pp=$PP layers=$LAYERS plens='$PLENS' ==="
  echo "[sweep] === arm=$arm pp=$PP layers=$LAYERS plens='$PLENS' ===" >> "$LOG"
  # All prompt lengths sweep inside ONE engine (max_model_len is sized from the
  # largest), so this costs 5 engine starts, not 40.
  ASCEND_RT_VISIBLE_DEVICES="$devs" .venv/bin/python \
    npu_ops/python/bench_engine_decode.py \
    --arm "$arm" --pp "$PP" --layers "$LAYERS" \
    --prompt-len $PLENS --batch $BATCHES --out-len 64 --prefill-probe \
    --cudagraph-mode FULL_DECODE_ONLY --out "$OUT" >> "$LOG" 2>&1 || {
      echo "[sweep] FAILED: arm=$arm (see $LOG)"
      exit 1
    }
  grep -E "^\[bench\] (rank|plen)" "$LOG" | tail -n $(( $(echo "$PLENS" | wc -w) * $(echo "$BATCHES" | wc -w) + PP ))
done
echo "[sweep] done -> $OUT"
