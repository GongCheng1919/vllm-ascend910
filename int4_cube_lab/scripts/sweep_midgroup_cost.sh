#!/bin/bash
# P0: what does mid-group cost, per GK?
#
# Compares the mid-group W4A8 probe (GK = 256/512/1024) against the per-channel
# W4A8 kernel (GK = inf, one Fixpipe per output tile) and the W8A8 baseline, on
# QwQ-32B Linear shapes, cold.  The probe uses degenerate per-group scales so
# the math matches per-channel exactly -- only the pipeline differs.
#
# usage: sweep_midgroup_cost.sh [out.csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/midgroup_cost.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
MS="${MS:-1 16 64 512 1024}"

# QwQ-32B: hidden=5120, intermediate=27648, 64 layers, GQA 40/8 heads x 128.
PROJS="${PROJS:-qkv:7168:5120 o:5120:5120 gate_up:55296:5120 down:5120:27648}"

tile_for() {
    if   [ "$1" -le 16 ]; then echo 16
    elif [ "$1" -le 64 ]; then echo 64
    else                       echo 128
    fi
}

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
echo "proj,variant,gk,M,N,K,tileM,avg_us" > "${OUT}"

for proj in ${PROJS}; do
    IFS=: read -r pname N K <<< "${proj}"
    for m in ${MS}; do
        tm=$(tile_for "${m}")
        # per-channel W4A8 (GK = inf) and the W8A8 baseline use the older naming
        case "${tm}" in
            16)  pc_sfx="_m16";  i8_sfx="_m16" ;;
            64)  pc_sfx="_m64";  i8_sfx="_m64" ;;
            128) pc_sfx="";      i8_sfx=""     ;;
        esac

        us=$(bench "${LAB_ROOT}/build/perchannel_int8_gemm${i8_sfx}_test" "${m}" "${N}" "${K}")
        echo "${pname},w8a8,-,${m},${N},${K},${tm},${us:-FAILED}" | tee -a "${OUT}"

        us=$(bench "${LAB_ROOT}/build/perchannel_w4a8_gemm${pc_sfx}_test" "${m}" "${N}" "${K}")
        echo "${pname},w4a8_pc,inf,${m},${N},${K},${tm},${us:-FAILED}" | tee -a "${OUT}"

        for gk in 1024 512 256; do
            us=$(bench "${LAB_ROOT}/build/midgroup_w4a8_gemm_m${tm}_g${gk}_test" "${m}" "${N}" "${K}")
            echo "${pname},w4a8_mg,${gk},${m},${N},${K},${tm},${us:-FAILED}" | tee -a "${OUT}"
        done
    done
done

echo
echo "wrote ${OUT}"
