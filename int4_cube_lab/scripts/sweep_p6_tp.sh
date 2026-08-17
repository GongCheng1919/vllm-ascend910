#!/bin/bash
# P6 B: what does tensor parallelism actually cost the mid-group kernel?
#
# vLLM splits K on the ROW-parallel projections (o_proj, down_proj) and N on the
# COLUMN-parallel ones (qkv, gate_up).  Only the row-parallel side interacts with
# the quant group.
#
# QwQ-32B has K = 2^10 * odd on both row-parallel projections
# (5120 = 1024*5, 27648 = 1024*27), so K_rank = K/TP is divisible by
# GK = 1024/TP at every TP -- the group boundary stays uniform and NO ragged
# group is needed.  TP=8 would want GK=128, which is below the kernel's floor
# (GK must be 256/512/1024 and K % INNER_K == 0), so this sweep stops at TP=4.
#
# The number we are after: the group flush is charged per flush COUNT (P5 D7),
# and under TP the per-rank flush count is
#     K_rank / GK = (K/TP) / (1024/TP) = K/1024      -- INVARIANT in TP
# while the MACs and the weight stream both drop by TP.  So the flush tax as a
# FRACTION of the layer should grow with TP.  `nofix` (flush deleted) gives the
# floor at each shape, so tax = full - nofix is directly readable per TP.
#
# usage: sweep_p6_tp.sh [out.csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/p6_tp_shapes.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
MS="${MS:-16 64 128 512 1024}"

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

tile_for() {  # M -> TILE_M, same dispatch as npu_ops/host TileMFor
    if   [ "$1" -le 16 ]; then echo 16
    elif [ "$1" -le 64 ]; then echo 64
    else echo 128; fi
}

mkdir -p "$(dirname "${OUT}")"
echo "proj,split,tp,M,N,K,gk,tileM,full_us,nofix_us,tax_us,tax_pct" > "${OUT}"

run_row() {  # pname split tp N K gk M
    local pname="$1" split="$2" tp="$3" N="$4" K="$5" gk="$6" m="$7"
    local tm; tm=$(tile_for "${m}")
    local suffix; [ "${gk}" -eq 1024 ] && suffix="g1024_asym" || suffix="g${gk}_asym"
    local full="${LAB_ROOT}/build/midgroup_w4a8_gemm_m${tm}_${suffix}_test"
    local nofx="${LAB_ROOT}/build/midgroup_w4a8_gemm_m${tm}_${suffix}_nofix_test"
    local f n tax pct
    f=$(bench "${full}" "${m}" "${N}" "${K}")
    n=$(bench "${nofx}" "${m}" "${N}" "${K}")
    if [ -n "${f}" ] && [ -n "${n}" ]; then
        tax=$(awk -v a="${f}" -v b="${n}" 'BEGIN{printf "%.2f", a-b}')
        pct=$(awk -v a="${f}" -v b="${n}" 'BEGIN{printf "%.2f", (a-b)/a*100}')
    fi
    echo "${pname},${split},${tp},${m},${N},${K},${gk},${tm},${f:-FAILED},${n:-FAILED},${tax:-NA},${pct:-NA}" \
        | tee -a "${OUT}"
}

for m in ${MS}; do
    # --- ROW-parallel: K is split, GK must shrink with it ---
    run_row o    K 1 5120 5120  1024 "${m}"
    run_row o    K 2 5120 2560   512 "${m}"
    run_row o    K 4 5120 1280   256 "${m}"
    run_row down K 1 5120 27648 1024 "${m}"
    run_row down K 2 5120 13824  512 "${m}"
    run_row down K 4 5120  6912  256 "${m}"

    # --- COLUMN-parallel: N is split, K and GK untouched (control) ---
    run_row qkv      N 1  7168 5120 1024 "${m}"
    run_row qkv      N 2  3584 5120 1024 "${m}"
    run_row qkv      N 4  1792 5120 1024 "${m}"
    run_row gate_up  N 1 55296 5120 1024 "${m}"
    run_row gate_up  N 2 27648 5120 1024 "${m}"
    run_row gate_up  N 4 13824 5120 1024 "${m}"
done
echo; echo "wrote ${OUT}"
