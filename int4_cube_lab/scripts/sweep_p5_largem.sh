#!/bin/bash
# P5 (2026-08-14 reopen, D4): the ablation ladder at LARGE M (TILE_M=128).
#
# The M=16 ladder (P5_KERNEL_OPT.md §3'.1) found this kernel pure GM->L1 DMA
# bound -- but that only holds where the weight stream dominates.  At M>=512
# the mid-group tax against per-channel W4A8 is 1.34-1.47x, and the workspace
# round trip is O(M*N*G) while the weight stream is O(N*K/2), so the balance
# must flip.  This ladder says WHICH of the two flush costs it is:
#
#   pc        per-channel W4A8, same MSD, G=1 flush  -> the floor
#   full      mid-group, G flushes per output tile
#   noaiv     AIC unchanged (still Fixpipes to GM), AIV handshake-only
#             -> removes the workspace READ + all vector rescale work
#   nofix     no Fixpipe                -> also removes the workspace WRITE
#   dmaonly   GM->L1 only               -> the weight-stream floor
#
# usage: sweep_p5_largem.sh [out.csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/p5_largem_ablation.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
MS="${MS:-128 512 1024}"
PROJS="${PROJS:-qkv:7168:5120 o:5120:5120 gate_up:55296:5120 down:5120:27648}"

bench() {  # binary M N K -> median-of-3 us
    local bin="$1" m="$2" n="$3" k="$4" times="" row
    [ -x "${bin}" ] || { echo ""; return; }
    for _t in 1 2 3; do
        row=$("${bin}" --rows "${m}" --cols "${n}" --k "${k}" --profile --nosync --cold \
                --warmup "${WARMUP}" --repeat "${REPEAT}" 2>/dev/null \
              | grep '^\[csv\]' | sed 's/^\[csv\][[:space:]]*//')
        [ -n "${row}" ] && times="${times} $(echo "${row}" | cut -d, -f5)"
    done
    echo ${times} | tr ' ' '\n' | sort -g | awk 'NR==2'
}

mkdir -p "$(dirname "${OUT}")"
echo "proj,variant,M,N,K,tileM,avg_us" > "${OUT}"

MG="${LAB_ROOT}/build/midgroup_w4a8_gemm_m128_g1024_asym"
for proj in ${PROJS}; do
    IFS=: read -r pname N K <<< "${proj}"
    for m in ${MS}; do
        for v in pc:${LAB_ROOT}/build/perchannel_w4a8_gemm_test \
                 full:${MG}_test \
                 noaiv:${MG}_noaiv_test \
                 nofix:${MG}_nofix_test \
                 dmaonly:${MG}_dmaonly_test; do
            IFS=: read -r vname bin <<< "${v}"
            us=$(bench "${bin}" "${m}" "${N}" "${K}")
            echo "${pname},${vname},${m},${N},${K},128,${us:-FAILED}" | tee -a "${OUT}"
        done
    done
done
echo; echo "wrote ${OUT}"
