#!/bin/bash
# Throughput sweep for the hand-written INT4 / INT8 cube kernels.
#   bash scripts/sweep.sh [output.csv]
# Device is picked with ASCEND_RT_VISIBLE_DEVICES (default 1 — device 0 has been
# observed wedged; check with the reference kernel before trusting a device).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/custom_kernels.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

SIZES="${SIZES:-512 1024 2048 4096 8192}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-20}"

mkdir -p "$(dirname "${OUT}")"
echo "kernel,M,N,K,avg_us,tflops" > "${OUT}"

for op in perchannel_int4_gemm perchannel_int8_gemm; do
    for s in ${SIZES}; do
        line=$("${LAB_ROOT}/build/${op}_test" \
                 --rows "${s}" --cols "${s}" --k "${s}" \
                 --profile --warmup "${WARMUP}" --repeat "${REPEAT}" 2>/dev/null \
               | grep '^\[csv\]' | sed 's/^\[csv\][[:space:]]*//')
        if [ -n "${line}" ]; then
            echo "${line}" | tee -a "${OUT}"
        else
            echo "${op},${s},${s},${s},FAILED,FAILED" | tee -a "${OUT}"
        fi
    done
done

echo
echo "wrote ${OUT}"
