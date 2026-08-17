#!/usr/bin/env bash
# P6 · PP at CONSTANT LAYERS PER RANK.  Companion to sweep_p6_pp.sh, which holds
# the TOTAL layer count fixed instead -- read this header before comparing them.
#
# WHY THIS EXISTS.  `sweep_p6_pp.sh` ran every PP at 4 and 16 total layers, so
# PP=4 got 1 and 4 layers per rank.  At that ratio a decode step is dominated by
# pipeline overhead (inter-rank transfer, bubbles, the ~4 ms/step scheduler and
# sampling cost) and the GEMM -- the only thing our arms change -- is a small
# slice.  All three arms duly converged at PP=4 (BF16 43 / W8A8 41 / W4A8 44
# tok/s @16 layers, batch 1).  That is a property of the CONFIGURATION, not of
# the kernels: the real QwQ-32B has 64 layers, so PP=4 means 16 layers per rank.
#
# It also broke the slope method.  `sweep_p6_pp.sh` reported a BF16 PP=4 slope of
# 310 us/layer against 1048 at PP=1 -- "PP made a layer 3.4x cheaper", which is
# structurally impossible because PP shards BY layer and never inside one.  The
# two-point slope assumes step(L) is affine in L with a PP-independent intercept;
# under PP the intercept moves with how the layers distribute, so slopes taken
# across a changing layers-per-rank are not per-layer costs.  Do not quote them.
#
# THE DESIGN.  Every rank holds the same number of layers in every config, so the
# per-rank weight footprint is identical (~14 GB at 16 layers/rank) and the
# pipeline is loaded the way a real deployment loads it:
#
#     PP=1 -> L= 8, 16      PP=2 -> L=16, 32      PP=4 -> L=32, 64
#              8   16/rank            8   16/rank           8   16/rank
#
# `PP=4, L=64` is the REAL model: full 64-layer QwQ-32B on four cards.
#
# HOW TO READ IT.  Compare arms WITHIN a PP -- does W4A8 keep its lead over our
# W8A8 and over BF16 once the model is sharded.  Do NOT compare tok/s ACROSS PP:
# those rows are different-sized models (a 64-layer model does 4x the work per
# token as a 16-layer one), so a lower number at PP=4 is arithmetic, not a
# regression.  Slope within a fixed PP is (step@16/rank - step@8/rank)/(8*PP).
#
# Device 0 is wedged (hangs every launch), so ranks start at 1.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${OUT:-int4_cube_lab/results/p6_pp_isorank.csv}
LOG=${LOG:-int4_cube_lab/results/p6_pp_isorank.log}
BATCHES=${BATCHES:-"1 16 64 128"}
ARMS=${ARMS:-"bf16 w8a8 w4a8"}
PPS=${PPS:-"1 2 4"}
LPRS=${LPRS:-"8 16"}          # layers per rank

: > "$LOG"
echo "[sweep] out=$OUT log=$LOG arms='$ARMS' pps='$PPS' lprs='$LPRS'"

for pp in $PPS; do
  devs=$(seq 1 "$pp" | paste -sd,)
  for arm in $ARMS; do
    for lpr in $LPRS; do
      layers=$(( lpr * pp ))
      echo "[sweep] === arm=$arm pp=$pp layers=$layers (${lpr}/rank) devs=$devs ==="
      echo "[sweep] === arm=$arm pp=$pp layers=$layers (${lpr}/rank) devs=$devs ===" >> "$LOG"
      ASCEND_RT_VISIBLE_DEVICES="$devs" .venv/bin/python \
        npu_ops/python/bench_engine_decode.py \
        --arm "$arm" --pp "$pp" --layers "$layers" \
        --batch $BATCHES --out-len 64 \
        --cudagraph-mode FULL_DECODE_ONLY --out "$OUT" >> "$LOG" 2>&1 || {
          echo "[sweep] FAILED: arm=$arm pp=$pp layers=$layers (see $LOG)"
          exit 1
        }
      # Exit 2 from the bench means some rank converted nothing, so reaching
      # here proves every rank really runs the arm.
      grep -E "^\[bench\] (rank|batch)" "$LOG" | tail -n $(( $(echo "$BATCHES" | wc -w) + pp ))
    done
  done
done
echo "[sweep] done -> $OUT"
