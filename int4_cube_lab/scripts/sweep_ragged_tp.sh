#!/bin/bash
# P6.5: does the ragged-K route keep the algorithm's performance?
#
# TWO QUESTIONS, and the first one is the load-bearing one:
#
#  A. REGRESSION on ALIGNED shapes.  The ragged change moved `mm.k` and
#     `ldA/ldB.repeatTimes` from compile-time constants to runtime values and put
#     a SliceKB() call in the AIC's inner slice loop.  If that cost anything, every
#     number P0..P6 published stops being comparable -- which matters more than
#     any TP result.  Compared against results/p6_tp_shapes.csv (measured on
#     2026-08-17 with the PRE-ragged binary) at the same shapes and the same
#     protocol.  The lab's noise floor is +-2.6% (P6 D4), so anything inside that
#     is not a claim in either direction.
#
#  B. The ragged route vs the route it replaces, MEASURED IN THE SAME RUN:
#       mg_new = GK 1024 at K_rank              (ragged: 1..3 short groups)
#       mg_old = GK 1024/TP at K_rank           (the P6 route, exact groups)
#       w8     = per-channel int8               (the denominator; P6 D6 -- comparing
#                                                mid-group to itself decides nothing)
#     Same binaries, same session, so cross-session drift cannot explain a gap.
#
# Protocol is sweep_p6_tp.sh's verbatim: --cold --profile --nosync, median-of-3,
# warmup 10 / repeat 50.  --cold matters: 71 MB of int4 weights fit the 168 MB L2
# and a hot loop reads 1949 GB/s that no real layer ever sees.
#
# usage: sweep_ragged_tp.sh [out.csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/p65_ragged_tp.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
MS="${MS:-16 128 512}"

bench() {  # binary M N K -> median-of-3 us ("" if the binary rejects the shape)
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

tile_for() { if [ "$1" -le 16 ]; then echo 16; elif [ "$1" -le 64 ]; then echo 64; else echo 128; fi; }
int8_bin() {  # TILE_M -> the per-channel int8 binary
    case "$1" in
        16) echo "${LAB_ROOT}/build/perchannel_int8_gemm_m16_test";;
        64) echo "${LAB_ROOT}/build/perchannel_int8_gemm_m64_test";;
        *)  echo "${LAB_ROOT}/build/perchannel_int8_gemm_test";;
    esac
}
div() { awk -v a="$1" -v b="$2" 'BEGIN{if(b+0==0||a==""||b=="")print"";else printf "%.3f", a/b}'; }

mkdir -p "$(dirname "${OUT}")"
echo "proj,tp,M,N,K,tileM,gk_old,mg_new_us,mg_old_us,w8_us,new_vs_old,new_vs_w8" > "${OUT}"

row() {  # proj tp N K gk_old M
    local proj="$1" tp="$2" N="$3" K="$4" gko="$5" m="$6"
    local tm; tm=$(tile_for "${m}")
    # MUST be initialised: with `set -u`, a TP whose old route does not exist
    # (gk_old = "-") would reference an unset `old` and abort the row.
    local new="" old="" w8=""
    new=$(bench "${LAB_ROOT}/build/midgroup_w4a8_gemm_m${tm}_g1024_asym_test" "${m}" "${N}" "${K}")
    if [ "${gko}" != "-" ]; then
        old=$(bench "${LAB_ROOT}/build/midgroup_w4a8_gemm_m${tm}_g${gko}_asym_test" "${m}" "${N}" "${K}")
    fi
    w8=$(bench "$(int8_bin "${tm}")" "${m}" "${N}" "${K}")
    echo "${proj},${tp},${m},${N},${K},${tm},${gko},${new:-NA},${old:-NA},${w8:-NA},$(div "${new}" "${old}"),$(div "${new}" "${w8}")" \
        | tee -a "${OUT}"
}

echo "=== A+B: o_proj (N=5120) and down_proj (N=5120), K_rank = K/TP ==="
for m in ${MS}; do
    row o    1 5120  5120  -    "${m}"     # aligned: the regression reference
    row o    2 5120  2560  512  "${m}"
    row o    4 5120  1280  256  "${m}"
    row o    8 5120   640  -    "${m}"     # GK=128 is below the old floor: new route only
    row down 1 5120 27648  -    "${m}"     # aligned: the regression reference
    row down 2 5120 13824  512  "${m}"
    row down 4 5120  6912  256  "${m}"
    row down 8 5120  3456  -    "${m}"
done

echo
echo "wrote ${OUT}"
