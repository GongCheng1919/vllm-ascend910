#!/usr/bin/env bash
# P6 · vLLM pipeline parallelism (PP) for our W8A8 / W4A8 arms.
#
# WHY PP AND NOT TP.  Our GEMM quantises along K at GK=1024.  TP shards the K of
# the row-parallel projections (o_proj K=5120, down_proj K=27648) and neither
# stays a multiple of 1024 at any TP>1, so mid-group needs a re-quantised
# checkpoint per TP -- that is P6.5's problem.  PP shards by LAYER: every rank
# keeps a whole K, GK=1024 stays aligned, and the kernels run unchanged.  This
# is the parallelism we can deploy today.
#
# HOW TO READ THE RESULT.  PP does NOT cut single-request step latency: a token
# still traverses all L layers, just spread over ranks, plus inter-rank
# transfer.  The two questions this sweep answers are
#   1. does the per-layer slope survive sharding (i.e. do our kernels and the
#      W4A8/W8A8 ratio hold up), and
#   2. what does throughput do once there are enough concurrent sequences for
#      the pipeline to actually fill.
# A "PP=4 is 4x" reading would be wrong; do not write one.
#
# Slope = (step_us@16layers - step_us@4layers) / 12, which also removes the
# ~4 ms/step fixed cost (scheduler, sampling, the unconverted 1.56 GB lm_head).
#
# Device 0 is wedged (hangs every launch), so ranks start at 1.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${OUT:-int4_cube_lab/results/p6_pp_engine.csv}
LOG=${LOG:-int4_cube_lab/results/p6_pp_engine.log}
BATCHES=${BATCHES:-"1 16 64 128"}
ARMS=${ARMS:-"bf16 w8a8 w4a8"}
PPS=${PPS:-"1 2 4"}

: > "$LOG"
echo "[sweep] out=$OUT log=$LOG arms='$ARMS' pps='$PPS' batches='$BATCHES'"

for pp in $PPS; do
  # ranks 1..pp, skipping the wedged device 0
  devs=$(seq 1 "$pp" | paste -sd,)
  for arm in $ARMS; do
    for layers in 4 16; do
      echo "[sweep] === arm=$arm pp=$pp layers=$layers devs=$devs ==="
      echo "[sweep] === arm=$arm pp=$pp layers=$layers devs=$devs ===" >> "$LOG"
      # `--cudagraph-mode` is mandatory: the vllm-ascend default PIECEWISE costs
      # ~390 us/layer of device idle and makes every arm unreadable (P4 D10).
      ASCEND_RT_VISIBLE_DEVICES="$devs" .venv/bin/python \
        npu_ops/python/bench_engine_decode.py \
        --arm "$arm" --pp "$pp" --layers "$layers" \
        --batch $BATCHES --out-len 64 \
        --cudagraph-mode FULL_DECODE_ONLY --out "$OUT" >> "$LOG" 2>&1 || {
          echo "[sweep] FAILED: arm=$arm pp=$pp layers=$layers (see $LOG)"
          exit 1
        }
      # The bench exits 2 if any rank converted nothing, so reaching here means
      # every rank really is running the arm -- see its `rank N: converted=` lines.
      grep -E "^\[bench\] (rank|batch)" "$LOG" | tail -n $(( $(echo "$BATCHES" | wc -w) + pp ))
    done
  done
done
echo "[sweep] done -> $OUT"
