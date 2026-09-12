#!/usr/bin/env bash
# P6.5 · the TP=4 DELIVERY matrix on the REAL 64-layer QwQ-32B.
#
# WHY THIS EXISTS.  Everything TP in P6.5 so far was DIAGNOSTIC, not deliverable:
# L=16, plen=128, batch in {1,16,64}, run to chase the 3632 us of unoverlapped
# comm (D9) and the 5.3 ms of device idle (D12).  The roadmap's actual P4 exit
# criterion is a different thing entirely -- MID_GROUP_ROADMAP.md Section 8 item 4,
# "rerun the frozen TP4 matrix, report BF16 / W8A8 / W4A8-mg three ways", with
# the decode-dense point at 1024 in / 256 out and concurrency 1-64.  P6 produced
# exactly that under PP; under TP the concurrency axis is thin and the
# long-context and prefill axes DO NOT EXIST AT ALL.  This sweep fills them.
#
# WHAT TO EXPECT, so a surprise is legible as a surprise:
#   * TP=4 will NOT beat TP=1 on latency.  That is settled (D13: 8217 vs 8182 us,
#     0.4%), and the account balances (D9).  This sweep is not re-asking it.
#   * The 3632 us/step of HCCL all-reduce is ARM-INDEPENDENT, so it is a fixed
#     tax added to every arm's step.  Our ratio over BF16 is therefore DILUTED at
#     TP=4 relative to the 2.28x measured at TP=1 (D12).  That dilution is a real
#     property of TP, not a measurement error, and the report must say so.  The
#     self-written all-reduce (D16-D18, 15x vs HCCL) is what would undo ~86% of
#     it; it is priced and shelved, NOT in this path -- the engine here calls
#     vLLM's own `tensor_model_parallel_all_reduce` like any other model.
#   * Long context, decode: M stays = batch, attention grows => our GEMM win gets
#     diluted, ratio sags toward 1.0 (same mechanism as P6's PP long-ctx sweep).
#   * Long context, PREFILL: M = the whole prompt, which is the large-M regime
#     where mid-group pays the structural flush tax (P5).  Expect to LOSE, as PP
#     did (0.64-0.69x, P6 D13).  Reported separately, never blended.
#
# THE TWO AXES ARE NOT A CROSS PRODUCT.  batch=256 x plen=8192 is 2M tokens of
# KV and would OOM, and it answers no question anyone asked.  Concurrency and
# context are swept as two separate lines through the space:
#   AXIS=conc : plen=1024, out=256, batch 1..256   <- the roadmap's own point
#   AXIS=ctx  : plen 128..8192, batch 1,8, out=64  <- P6's long-ctx design
#   AXIS=async: the conc axis again with async_scheduling OFF, bf16+w4a8 only.
#               D12 measured 1.23-1.29x for that flag at L=16, where idle was 44%
#               of the step; at L=64 idle should fall to ~17%, so the flag should
#               be worth LESS here.  That prediction has never been tested.
#
# ASYNC SCHEDULING IS ON for the two delivery axes, uniformly across all five
# arms.  It is one flag, it costs nothing, and with it off the 5.3 ms of host
# stall floods the table (D12).  It is unsupported with PP, which is why P6's
# numbers do not have it -- so do NOT compare a step_us from here against one
# from p6_pp_engine.csv without accounting for it.  The CSV carries an
# `async_sched` column for exactly that reason.
#
# Engine reproducibility is +-15% (P6 D11) and these are single runs.  A
# difference under ~5% is not a claim.  Ratios do NOT cancel the drift.
#
# Device 0 is wedged (every launch hangs there), so ranks are 1..4.
#
# usage:  [AXIS=conc|ctx|async|all] [ARMS=...] int4_cube_lab/scripts/sweep_p65_tp_e2e.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

AXIS=${AXIS:-all}
TP=${TP:-4}
LAYERS=${LAYERS:-64}
ARMS=${ARMS:-"bf16 w8a8 w4a8 w8a8-native w4a8-native"}
ASYNC_ARMS=${ASYNC_ARMS:-"bf16 w4a8"}
OUT=${OUT:-int4_cube_lab/results/p65_tp_e2e.csv}
LOG=${LOG:-int4_cube_lab/results/p65_tp_e2e.log}
PY=${PY:-.venv/bin/python}

CONC_PLEN=${CONC_PLEN:-1024}
CONC_BATCH=${CONC_BATCH:-"1 16 64 128 256"}
CONC_OUT=${CONC_OUT:-256}
CTX_PLEN=${CTX_PLEN:-"128 1024 4096 8192"}
CTX_BATCH=${CTX_BATCH:-"1 8"}
CTX_OUT=${CTX_OUT:-64}

devs=$(seq 1 "$TP" | paste -sd,)

# A killed vLLM worker keeps its card and every later run dies at
# hcclCommInitRootInfoConfig error 7.  Refuse to start on top of one rather than
# spend an hour producing FAILED rows.
# Ask the DRIVER who holds a card, not `pgrep` -- a pgrep pattern also matches
# the shell that is running the pattern, which false-positived on the first try.
orphans=$(npu-smi info 2>/dev/null | grep -E '^\| [0-9]+ +[0-9]+ +' || true)
if [ -n "$orphans" ]; then
    echo "[sweep] REFUSING: processes still hold NPU cards --"
    echo "$orphans"
    echo "[sweep] kill them (see /proc/*/fd for davinci handles), then rerun."
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
: > "$LOG"
echo "[sweep] axis=$AXIS tp=$TP layers=$LAYERS devs=$devs arms='$ARMS'"
echo "[sweep] out=$OUT log=$LOG"
echo "[sweep] started $(date -Is)" >> "$LOG"

run_one() {           # run_one <tag> <arm> <async 0|1> <out_len> <plens> <batches>
    local tag=$1 arm=$2 async=$3 olen=$4 plens=$5 batches=$6
    local extra=""
    [ "$async" = "1" ] && extra="--async-scheduling"
    echo "[sweep] === $tag arm=$arm async=$async plen='$plens' batch='$batches' out=$olen"
    {
        echo "=== $tag arm=$arm async=$async plen='$plens' batch='$batches' out=$olen"
        echo "=== $(date -Is)"
    } >> "$LOG"
    local t0=$SECONDS
    ASCEND_RT_VISIBLE_DEVICES="$devs" "$PY" \
        npu_ops/python/bench_engine_decode.py \
        --arm "$arm" --tp "$TP" --layers "$LAYERS" \
        --prompt-len $plens --batch $batches --out-len "$olen" \
        --prefill-probe --cudagraph-mode FULL_DECODE_ONLY \
        $extra --out "$OUT" >> "$LOG" 2>&1
    local rc=$?
    local dt=$(( SECONDS - t0 ))
    if [ $rc -ne 0 ]; then
        # Do NOT abort the sweep: one arm failing (an OOM at batch=256, a vendor
        # checkpoint that will not load) must not cost the other four.
        echo "[sweep] FAILED rc=$rc  $tag arm=$arm  (${dt}s) -- see $LOG"
        echo "=== FAILED rc=$rc after ${dt}s" >> "$LOG"
    else
        echo "[sweep] ok $tag arm=$arm (${dt}s)"
        grep -E "^\[bench\] plen=" "$LOG" | tail -n "$(( $(echo $plens | wc -w) * $(echo $batches | wc -w) ))"
    fi
    # Let the driver reclaim the cards before the next engine start.
    sleep 5
}

if [ "$AXIS" = "conc" ] || [ "$AXIS" = "all" ]; then
    for arm in $ARMS; do
        run_one conc "$arm" 1 "$CONC_OUT" "$CONC_PLEN" "$CONC_BATCH"
    done
fi

if [ "$AXIS" = "ctx" ] || [ "$AXIS" = "all" ]; then
    for arm in $ARMS; do
        run_one ctx "$arm" 1 "$CTX_OUT" "$CTX_PLEN" "$CTX_BATCH"
    done
fi

if [ "$AXIS" = "async" ] || [ "$AXIS" = "all" ]; then
    for arm in $ASYNC_ARMS; do
        run_one async-off "$arm" 0 "$CONC_OUT" "$CONC_PLEN" "$CONC_BATCH"
    done
fi

echo "[sweep] done $(date -Is).  rows:"
wc -l "$OUT"
