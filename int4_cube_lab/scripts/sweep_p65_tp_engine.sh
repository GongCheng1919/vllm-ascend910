#!/bin/bash
# P6.5: engine-level TP.  Does a token actually get faster, and does our W4A8 keep
# its lead over our W8A8 once K is sharded?
#
# WHY THE DESIGN DIFFERS FROM sweep_p6_pp_isorank.sh.  PP shards BY layer, so that
# sweep had to hold LAYERS PER RANK constant or the per-layer slope became
# meaningless (P6 D7).  TP shards WITHIN a layer: every rank holds all L layers, so
# there is no layers-per-rank knob at all.  The right design is the opposite one --
# hold the MODEL fixed and vary TP -- and the claim under test is that step_us goes
# DOWN, which is exactly what PP cannot do (P6 D10).
#
# L=16 (a quarter of QwQ-32B) so BF16 still fits one card at TP=1; the arms are
# compared WITHIN a TP, never across L.
#
# Engine reproducibility is +-15% (P6 D11) and these are single runs, so a
# difference under ~5% is not a claim.  Ratios do NOT cancel the drift.
#
# usage: sweep_p65_tp_engine.sh [out.csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="$(cd "${LAB_ROOT}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/p65_tp_engine.csv}"
LOG="${OUT%.csv}.log"
PY="${REPO}/.venv/bin/python"
LAYERS="${LAYERS:-16}"
BATCHES="${BATCHES:-1 16 64}"
ARMS="${ARMS:-bf16 w8a8 w4a8}"
TPS="${TPS:-1 2 4}"
# Device 0 is wedged (every launch hangs there), so ranks start at 1.
DEVS="1,2,3,4"

mkdir -p "$(dirname "${OUT}")"
: > "${LOG}"
echo "arm,layers,tp,batch,step_us,tok_s" > "${OUT}"

for tp in ${TPS}; do
    devs=$(echo "${DEVS}" | cut -d, -f1-"${tp}")
    for arm in ${ARMS}; do
        echo "=== arm=${arm} tp=${tp} L=${LAYERS} devs=${devs}" | tee -a "${LOG}"
        ASCEND_RT_VISIBLE_DEVICES="${devs}" "${PY}" \
            "${REPO}/npu_ops/python/bench_engine_decode.py" \
            --arm "${arm}" --tp "${tp}" --layers "${LAYERS}" \
            --batch ${BATCHES} --cudagraph-mode FULL_DECODE_ONLY \
            >> "${LOG}" 2>&1
        rc=$?
        if [ "${rc}" -ne 0 ]; then
            echo "  FAILED rc=${rc} -- see ${LOG}" | tee -a "${LOG}"
            for b in ${BATCHES}; do
                echo "${arm},${LAYERS},${tp},${b},FAILED,FAILED" >> "${OUT}"
            done
            continue
        fi
        # The bench prints one line per (plen, batch); pull step_us and tok/s.
        grep -E "^\[bench\] plen=" "${LOG}" | tail -n "$(echo ${BATCHES} | wc -w)" \
        | while read -r line; do
            b=$(echo "${line}"  | sed -E 's/.*batch=[[:space:]]*([0-9]+).*/\1/')
            st=$(echo "${line}" | sed -E 's/.*step=[[:space:]]*([0-9.]+).*/\1/')
            tk=$(echo "${line}" | sed -E 's/.*throughput=[[:space:]]*([0-9.]+).*/\1/')
            echo "${arm},${LAYERS},${tp},${b},${st},${tk}" | tee -a "${OUT}"
        done
    done
done

echo
echo "wrote ${OUT} (log: ${LOG})"
