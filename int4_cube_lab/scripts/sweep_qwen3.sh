#!/bin/bash
# Qwen3-32B MLP shapes (hidden=5120, intermediate=25600), TP=1:
#   gate_up_proj  N = 2*25600 = 51200,  K = 5120
#   down_proj     N = 5120,             K = 25600
# Picks the TILE_M build that fits M, so decode-shaped M is not padded to 128.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/qwen3_custom.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

MS="${MS:-1 4 16 64 128 512 1024}"
COLD_FLAG=""
[ "${COLD:-0}" = "1" ] && COLD_FLAG="--cold"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"

# suffix of the build whose TILE_M covers this M without over-padding
variant_for() {
    if   [ "$1" -le 16 ]; then echo "_m16"
    elif [ "$1" -le 64 ]; then echo "_m64"
    else                       echo ""
    fi
}

mkdir -p "$(dirname "${OUT}")"
echo "proj,dtype,M,N,K,avg_us,tflops" > "${OUT}"

for proj in "gate_up:51200:5120" "down:5120:25600"; do
    IFS=: read -r pname N K <<< "${proj}"
    for m in ${MS}; do
        sfx=$(variant_for "${m}")
        for dt in int4 int8; do
            bin="${LAB_ROOT}/build/perchannel_${dt}_gemm${sfx}_test"
            # Clocks drift ~14% run to run on this part, so take a median of 3.
            times=""
            for _t in 1 2 3; do
                row=$("${bin}" --rows "${m}" --cols "${N}" --k "${K}" \
                        --profile --nosync ${COLD_FLAG} --warmup "${WARMUP}" --repeat "${REPEAT}" 2>/dev/null \
                      | grep '^\[csv\]' | sed 's/^\[csv\][[:space:]]*//')
                [ -n "${row}" ] && times="${times} $(echo "${row}" | cut -d, -f5)"
            done
            us=$(echo ${times} | tr ' ' '\n' | sort -g | awk 'NR==2')
            if [ -n "${us}" ]; then
                tf=$(awk -v m="${m}" -v n="${N}" -v k="${K}" -v u="${us}" \
                     'BEGIN{printf "%.2f", 2.0*m*n*k/u/1e6}')
                echo "${pname},${dt},${m},${N},${K},${us},${tf}" | tee -a "${OUT}"
            else
                echo "${pname},${dt},${m},${N},${K},FAILED,FAILED" | tee -a "${OUT}"
            fi
        done
    done
done

echo
echo "wrote ${OUT}"
