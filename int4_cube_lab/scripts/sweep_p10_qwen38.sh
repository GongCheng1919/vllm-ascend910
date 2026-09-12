#!/bin/bash
# P10 Phase A -- does mid-group W4A8 still win on Qwen3.8-27B's shapes?
#
# WHY THIS RUNS BEFORE ANY ENVIRONMENT UPGRADE.  Qwen3.8-27B is `model_type:
# qwen3_5`; it does not exist in vllm 0.13.0 / transformers 4.57.6, and the only
# vllm-ascend that has it (v0.23.0) drags in torch_npu 2.10 + CANN 9.0.1 -- a CANN
# MAJOR bump that lands on this kernel, not on the python glue.  `int4_cube_lab`
# imports no vLLM at all, so the operator-level answer is available TODAY on
# CANN 8.5.  This sweep is what decides whether that upgrade is worth buying.
#
# THE SHAPES (Qwen/Qwen3.8-27B config.json, text_config):
#   hidden 5120 · intermediate 17408 · 64 layers
#   layer_types: 16 x full_attention + 48 x linear_attention (Gated DeltaNet)
#   full attn  : 24 Q heads x head_dim 256 (attn_output_gate) · 4 KV heads
#   linear attn: 16 key heads x 128 · 48 value heads x 128
#
#   proj          layers  parallel  N        K        split
#   qkv               16  column    14336    5120     N/TP
#   o                 16  row        5120    6144     K/TP
#   gdn_in_qkvz       48  column    16384    5120     N/TP
#   gdn_out           48  row        5120    6144     K/TP   (same shape as o)
#   gate_up           64  column    34816    5120     N/TP
#   down              64  row        5120   17408     K/TP
#
#   NOT MEASURED, deliberately: gdn in_proj_ba (N=96) and the whole vision tower
#   (K=1152, N=4304) fail `N % 128 == 0` and stay BF16.  ba is 0.49 M/layer and
#   sits on the recurrent decay gate, where quantisation error compounds along the
#   sequence -- leaving it in BF16 is right independently of alignment.
#
# QwQ-32B rows are measured IN THE SAME SESSION as the control.  Clocks drift
# ~14% run to run on this part (P6 D4), so a cross-session comparison against
# p65_ragged_tp.csv would not be readable.
#
# M covers the MTP draft head: Qwen3.8 ships a 1-layer MTP head, so batch=1
# decode verifies M=2..4, not M=1.  That is still inside the win region, but it
# has to be measured rather than assumed.
#
# Protocol is sweep_ragged_tp.sh's verbatim: --cold --profile --nosync,
# median-of-3, warmup 10 / repeat 50.  --cold is not optional: int4 weights fit
# the 168 MB L2 and a hot loop reads a bandwidth no real layer ever sees.
#
# usage: sweep_p10_qwen38.sh [out.csv]
set -uo pipefail

# Two of these running at once contend for the same device and silently inflate
# BOTH sets of timings while interleaving rows into one CSV -- it looks like data,
# not like a failure.  Cost me one full sweep on 2026-09-04.
#
# flock, not pgrep: a `pgrep -f sweep_p10_qwen38.sh` also matches the shell that
# launched us (its -c string contains the script name), so the pgrep version
# refused to run at all.
_LOCK=/tmp/.sweep_p10_qwen38.lock
exec 9>"${_LOCK}"
if ! flock -n 9; then
    echo "REFUSING: another sweep_p10_qwen38.sh holds ${_LOCK}." >&2
    echo "  kill it first, or the timings of both runs are contaminated." >&2
    exit 3
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/p10_qwen38_shapes.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
MS="${MS:-1 2 4 8 16 64}"
TPS="${TPS:-1 2 4}"

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
int8_bin() {
    case "$1" in
        16) echo "${LAB_ROOT}/build/perchannel_int8_gemm_m16_test";;
        64) echo "${LAB_ROOT}/build/perchannel_int8_gemm_m64_test";;
        *)  echo "${LAB_ROOT}/build/perchannel_int8_gemm_test";;
    esac
}
div() { awk -v a="$1" -v b="$2" 'BEGIN{if(b+0==0||a==""||b=="")print"";else printf "%.3f", a/b}'; }

mkdir -p "$(dirname "${OUT}")"
echo "model,proj,layers,par,tp,M,N,K,tileM,mg_us,w8_us,mg_vs_w8" > "${OUT}"

row() {  # model proj layers par tp N K M   (N,K already split for this TP)
    local model="$1" proj="$2" L="$3" par="$4" tp="$5" N="$6" K="$7" m="$8"
    local tm mg w8
    tm=$(tile_for "${m}")
    mg=$(bench "${LAB_ROOT}/build/midgroup_w4a8_gemm_m${tm}_g1024_asym_test" "${m}" "${N}" "${K}")
    w8=$(bench "$(int8_bin "${tm}")" "${m}" "${N}" "${K}")
    echo "${model},${proj},${L},${par},${tp},${m},${N},${K},${tm},${mg:-NA},${w8:-NA},$(div "${mg}" "${w8}")" \
        | tee -a "${OUT}"
}

# proj:layers:parallel:N_tp1:K_tp1
QWEN38="qkv:16:col:14336:5120 o:16:row:5120:6144 gdn_in_qkvz:48:col:16384:5120 gdn_out:48:row:5120:6144 gate_up:64:col:34816:5120 down:64:row:5120:17408"
QWQ32B="qkv:64:col:7680:5120 o:64:row:5120:5120 gate_up:64:col:55296:5120 down:64:row:5120:27648"

sweep() {  # model  "spec spec ..."
    local model="$1"; shift
    local spec pname L par N1 K1 N K
    for m in ${MS}; do
        for tp in ${TPS}; do
            for spec in "$@"; do
                IFS=: read -r pname L par N1 K1 <<< "${spec}"
                if [ "${par}" = "col" ]; then N=$((N1 / tp)); K=${K1}
                else                          N=${N1};        K=$((K1 / tp)); fi
                row "${model}" "${pname}" "${L}" "${par}" "${tp}" "${N}" "${K}" "${m}"
            done
        done
    done
}

echo "=== Qwen3.8-27B ==="
sweep qwen38 ${QWEN38}
echo
echo "=== QwQ-32B (same-session control) ==="
sweep qwq32b ${QWQ32B}

echo
echo "wrote ${OUT}"
