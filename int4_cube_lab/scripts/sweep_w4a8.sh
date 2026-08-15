#!/bin/bash
# W4A8 vs W8A8 vs W4A4, same pipeline / same tile geometry for all three.
#
#   square   M=N=K sweep, hot   -- the cube-bound end: does the MSD split cost
#                                  anything once the cube is the bottleneck?
#   qwen3    Qwen3-32B MLP shapes, cold -- the memory-bound end (decode), which
#                                  is where halving the weight is supposed to pay.
#
# usage: sweep_w4a8.sh [square|qwen3|all] [out.csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-all}"
OUT="${2:-${LAB_ROOT}/results/w4a8_${MODE}.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
DTYPES="${DTYPES:-int8 w4a8 int4}"

# suffix of the build whose TILE_M covers this M without over-padding
variant_for() {
    if   [ "$1" -le 16 ]; then echo "_m16"
    elif [ "$1" -le 64 ]; then echo "_m64"
    else                       echo ""
    fi
}

# median-of-3 microseconds for one (binary, M, N, K); clocks drift ~14% run to run.
bench() {
    local bin="$1" m="$2" n="$3" k="$4" cold="$5"
    local times="" row
    for _t in 1 2 3; do
        row=$("${bin}" --rows "${m}" --cols "${n}" --k "${k}" \
                --profile --nosync ${cold} --warmup "${WARMUP}" --repeat "${REPEAT}" 2>/dev/null \
              | grep '^\[csv\]' | sed 's/^\[csv\][[:space:]]*//')
        [ -n "${row}" ] && times="${times} $(echo "${row}" | cut -d, -f5)"
    done
    echo ${times} | tr ' ' '\n' | sort -g | awk 'NR==2'
}

emit() {  # tag M N K us
    if [ -n "$5" ]; then
        awk -v t="$1" -v d="$2" -v m="$3" -v n="$4" -v k="$5" -v u="$6" \
            'BEGIN{printf "%s,%s,%d,%d,%d,%s,%.2f\n", t,d,m,n,k,u, 2.0*m*n*k/u/1e6}'
    fi
}

mkdir -p "$(dirname "${OUT}")"
echo "shape,dtype,M,N,K,avg_us,tflops" > "${OUT}"

if [ "${MODE}" = "square" ] || [ "${MODE}" = "all" ]; then
    for s in 512 1024 2048 4096 8192; do
        for dt in ${DTYPES}; do
            bin="${LAB_ROOT}/build/perchannel_${dt}_gemm_test"
            us=$(bench "${bin}" "${s}" "${s}" "${s}" "")
            [ -z "${us}" ] && us="FAILED"
            if [ "${us}" = "FAILED" ]; then
                echo "square,${dt},${s},${s},${s},FAILED,FAILED" | tee -a "${OUT}"
            else
                emit "square" "${dt}" "${s}" "${s}" "${s}" "${us}" | tee -a "${OUT}"
            fi
        done
    done
fi

if [ "${MODE}" = "qwen3" ] || [ "${MODE}" = "all" ]; then
    MS="${MS:-1 4 16 64 128 512 1024}"
    for proj in "gate_up:51200:5120" "down:5120:25600"; do
        IFS=: read -r pname N K <<< "${proj}"
        for m in ${MS}; do
            sfx=$(variant_for "${m}")
            for dt in ${DTYPES}; do
                bin="${LAB_ROOT}/build/perchannel_${dt}_gemm${sfx}_test"
                us=$(bench "${bin}" "${m}" "${N}" "${K}" "--cold")
                if [ -z "${us}" ]; then
                    echo "${pname},${dt},${m},${N},${K},FAILED,FAILED" | tee -a "${OUT}"
                else
                    emit "${pname}" "${dt}" "${m}" "${N}" "${K}" "${us}" | tee -a "${OUT}"
                fi
            done
        done
    done
fi

echo
echo "wrote ${OUT}"
