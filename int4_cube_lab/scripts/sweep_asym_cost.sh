#!/bin/bash
# P3: what does the ASYMMETRIC weight cost, on top of mid-group GK=1024?
#
# Same protocol and same shapes as sweep_midgroup_cost.sh (which produced
# midgroup_cost_v4.csv), so the sym columns here should reproduce v4 -- that
# reproduction is the control.  The asym build feeds w_zero = 0 and a real
# a_ksum, so the extra MulAddDst executes on representative operands every
# group while contributing exactly 0.0 to the result (see mg_harness.h).
#
# usage: sweep_asym_cost.sh [out.csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/midgroup_cost_v5_asym.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
MS="${MS:-1 16 64 512 1024}"
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
        case "${tm}" in
            16)  pc_sfx="_m16";  i8_sfx="_m16" ;;
            64)  pc_sfx="_m64";  i8_sfx="_m64" ;;
            128) pc_sfx="";      i8_sfx=""     ;;
        esac

        us=$(bench "${LAB_ROOT}/build/perchannel_int8_gemm${i8_sfx}_test" "${m}" "${N}" "${K}")
        echo "${pname},w8a8,-,${m},${N},${K},${tm},${us:-FAILED}" | tee -a "${OUT}"

        us=$(bench "${LAB_ROOT}/build/perchannel_w4a8_gemm${pc_sfx}_test" "${m}" "${N}" "${K}")
        echo "${pname},w4a8_pc,inf,${m},${N},${K},${tm},${us:-FAILED}" | tee -a "${OUT}"

        us=$(bench "${LAB_ROOT}/build/midgroup_w4a8_gemm_m${tm}_g1024_test" "${m}" "${N}" "${K}")
        echo "${pname},w4a8_mg_sym,1024,${m},${N},${K},${tm},${us:-FAILED}" | tee -a "${OUT}"

        us=$(bench "${LAB_ROOT}/build/midgroup_w4a8_gemm_m${tm}_g1024_asym_test" "${m}" "${N}" "${K}")
        echo "${pname},w4a8_mg_asym,1024,${m},${N},${K},${tm},${us:-FAILED}" | tee -a "${OUT}"
    done
done

echo
echo "wrote ${OUT}"
