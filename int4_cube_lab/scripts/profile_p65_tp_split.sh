#!/usr/bin/env bash
# P6.5 · re-measure the TP=4 step SPLIT on the REAL 64-layer model.
#
# WHY.  D9's account -- Computing 6751->3275, Communication(Not Overlapped) 3632,
# Overlapped 0, Free 5302 -- was measured at L=16, plen=128.  The delivery matrix
# runs at L=64, plen=1024, where the comm term should scale with LAYER COUNT
# (one all-reduce per o_proj and one per down_proj, per layer) while the ~5 ms of
# host idle should NOT.  If comm really lands near 14 ms it is roughly HALF of
# the BF16 step, which caps what ANY arm's ratio can be at TP=4 -- and that
# ceiling is the single most load-bearing number for reading the delivery table.
#
# Extrapolating it would be guessing, and P6.5 has already been burned once by a
# per-layer extrapolation (D1: the GK cost curve was measured, not inferred).
# So: measure it, for bf16 AND w4a8, at the delivery axis's own shape.
#
# Absolute values run HIGH under profiling (D9 notes this); the SPLIT is what
# transfers, not the total.
#
# usage: int4_cube_lab/scripts/profile_p65_tp_split.sh [arm ...]
set -uo pipefail
cd "$(dirname "$0")/../.."

TP=${TP:-4}
LAYERS=${LAYERS:-64}
PLEN=${PLEN:-1024}
BATCH=${BATCH:-1}
ARMS=${*:-"bf16 w4a8"}
ROOT=${ROOT:-int4_cube_lab/results/p65_tp_split}
LOG=${LOG:-int4_cube_lab/results/p65_tp_split.log}
PY=${PY:-.venv/bin/python}

devs=$(seq 1 "$TP" | paste -sd,)
mkdir -p "$ROOT"
: > "$LOG"

for arm in $ARMS; do
    dir="$ROOT/$arm"
    rm -rf "$dir"; mkdir -p "$dir"
    echo "[split] === arm=$arm tp=$TP L=$LAYERS plen=$PLEN batch=$BATCH -> $dir"
    ASCEND_RT_VISIBLE_DEVICES="$devs" "$PY" \
        npu_ops/python/bench_engine_decode.py \
        --arm "$arm" --tp "$TP" --layers "$LAYERS" \
        --prompt-len "$PLEN" --batch "$BATCH" --out-len 32 \
        --cudagraph-mode FULL_DECODE_ONLY --async-scheduling \
        --profile-dir "$dir" >> "$LOG" 2>&1 \
        || { echo "[split] FAILED arm=$arm -- see $LOG"; continue; }
    sleep 5
done

# Multi-process profiling leaves each rank's raw dir UNANALYSED; without this the
# ASCEND_PROFILER_OUTPUT/ (and step_trace_time.csv) never appears.  D9 hit this.
echo "[split] analysing rank dirs (this is slow)..."
"$PY" - "$ROOT" >> "$LOG" 2>&1 <<'PY'
import glob, os, sys
import torch_npu
root = sys.argv[1]
for d in sorted(glob.glob(os.path.join(root, "*", "*_ascend_pt"))):
    if os.path.isdir(os.path.join(d, "ASCEND_PROFILER_OUTPUT")):
        continue
    print("analyse", d, flush=True)
    try:
        torch_npu.profiler.profiler.analyse(d)
    except Exception as e:                                  # noqa: BLE001
        print("  FAILED", repr(e), flush=True)
PY

echo "[split] step_trace_time.csv, rank 0 of each arm:"
for arm in $ARMS; do
    f=$(ls "$ROOT/$arm"/*_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        echo "--- $arm  ($f)"
        cat "$f"
    else
        echo "--- $arm  NO step_trace_time.csv (see $LOG)"
    fi
done
