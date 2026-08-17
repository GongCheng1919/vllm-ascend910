#!/bin/bash
# P6 C: at the TP-split ROW-parallel shapes, is mid-group W4A8 still the right
# kernel at all?
#
# sweep_p6_tp.sh answers "what does TP cost mid-group" by comparing mid-group to
# ITSELF.  That cannot decide anything: the deployable question is whether
# mid-group still beats the alternatives once K has been cut by TP.
#
#   mg   mid-group W4A8, GK = 1024/TP   -- the candidate
#   pc   per-channel W4A8, same MSD     -- one scale over the whole K_rank.
#                                          NOTE this gets FINER as TP grows:
#                                          at K_rank=1280 a per-channel scale
#                                          covers 1280 elements vs g1024's 1024,
#                                          so the accuracy gap nearly closes --
#                                          this is the user's "just use PC when
#                                          the split is small" idea, measured.
#   w8   per-channel int8 W8A8          -- what we are trying to beat
#
# usage: sweep_p6_tp_alt.sh [out.csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/p6_tp_alternatives.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
REPS="${REPS:-3}"
MS="${MS:-16 64 128 512}"

bench() {
    local bin="$1" m="$2" n="$3" k="$4" times="" row
    [ -x "${bin}" ] || { echo ""; return; }
    for _t in $(seq "${REPS}"); do
        row=$("${bin}" --rows "${m}" --cols "${n}" --k "${k}" --profile --nosync --cold \
                --warmup "${WARMUP}" --repeat "${REPEAT}" 2>/dev/null \
              | grep '^\[csv\]' | sed 's/^\[csv\][[:space:]]*//')
        [ -n "${row}" ] && times="${times} $(echo "${row}" | cut -d, -f5)"
    done
    echo ${times} | tr ' ' '\n' | sort -g | awk -v n="${REPS}" 'NR==int((n+1)/2)'
}

suffix_for() {  # TILE_M -> the per-channel build suffix ("" means the 128 build)
    case "$1" in 16) echo "_m16";; 64) echo "_m64";; *) echo "";; esac
}
tile_for() { if [ "$1" -le 16 ]; then echo 16; elif [ "$1" -le 64 ]; then echo 64; else echo 128; fi; }

mkdir -p "$(dirname "${OUT}")"
echo "proj,tp,M,N,K,gk,tileM,mg_us,pc_us,w8_us,mg_vs_pc,mg_vs_w8" > "${OUT}"

row() {  # pname tp N K gk M
    local pname="$1" tp="$2" N="$3" K="$4" gk="$5" m="$6"
    local tm; tm=$(tile_for "${m}")
    local sfx; sfx=$(suffix_for "${tm}")
    local gsfx; [ "${gk}" -eq 1024 ] && gsfx="g1024_asym" || gsfx="g${gk}_asym"
    local mg pc w8 r1 r2
    mg=$(bench "${LAB_ROOT}/build/midgroup_w4a8_gemm_m${tm}_${gsfx}_test" "${m}" "${N}" "${K}")
    pc=$(bench "${LAB_ROOT}/build/perchannel_w4a8_gemm${sfx}_test"        "${m}" "${N}" "${K}")
    w8=$(bench "${LAB_ROOT}/build/perchannel_int8_gemm${sfx}_test"        "${m}" "${N}" "${K}")
    [ -n "${mg}" ] && [ -n "${pc}" ] && r1=$(awk -v a="${mg}" -v b="${pc}" 'BEGIN{printf "%.3f", a/b}')
    [ -n "${mg}" ] && [ -n "${w8}" ] && r2=$(awk -v a="${mg}" -v b="${w8}" 'BEGIN{printf "%.3f", a/b}')
    echo "${pname},${tp},${m},${N},${K},${gk},${tm},${mg:-F},${pc:-F},${w8:-F},${r1:-NA},${r2:-NA}" \
        | tee -a "${OUT}"
}

for m in ${MS}; do
    row o    1 5120 5120  1024 "${m}"
    row o    2 5120 2560   512 "${m}"
    row o    4 5120 1280   256 "${m}"
    row down 1 5120 27648 1024 "${m}"
    row down 2 5120 13824  512 "${m}"
    row down 4 5120  6912  256 "${m}"
done
echo; echo "wrote ${OUT}"
