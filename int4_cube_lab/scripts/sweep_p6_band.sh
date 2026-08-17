#!/bin/bash
# P6 A: is D9's L2 banding safe to make the DEFAULT, and where can it possibly help?
#
# READ THIS BEFORE CHANGING THE M LIST.
#
# Banding reorders (mTile, nTile) so a band of weights stays L2-resident across
# the M-sweep.  When there is only ONE M-tile there is no M-sweep, and TileToMN
# collapses to the flat mapping exactly:
#
#     numMTiles == 1  =>  fullBandTiles == superN
#                     =>  mTile = within/superN = 0,  nTile = tileId
#
# i.e. banding is a mathematical IDENTITY at numMTiles == 1, not merely a no-op
# in effect.  And the host dispatcher (TileMFor: M<=16 -> 16, M<=64 -> 64, else
# 128) makes Mp/TILE_M == 1 unavoidable for TILE_M=16 and TILE_M=64:
#
#     TILE_M=16  <= reached only when M <= 16   -> Mp = 16   -> 1 M-tile
#     TILE_M=64  <= reached only when M <= 64   -> Mp = 64   -> 1 M-tile
#     TILE_M=128 <= M > 64; Mp = ceil(M/128)*128 -> >1 tile only from M >= 256
#
# So banding can ONLY do anything at TILE_M=128 with M >= 256 -- prefill.  The
# whole decode range that P4 tuned (batch 1..128) is structurally out of reach.
# M=256/512/1024 below give 2/4/8 M-tiles.
#
# qkv and o are the built-in controls in TWO independent ways: their N*Kb is
# under the 32 MB band cap (18.4 / 13.1 MB) so superN degenerates, AND they are
# the same code path either way.  Their delta IS the noise floor of this
# harness -- read it before believing any small win on gate_up/down.
#
# usage: sweep_p6_band.sh [out.csv]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-${LAB_ROOT}/results/p6_band_prefill.csv}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
REPS="${REPS:-5}"            # median-of-5, not 3: the controls showed ~2.6% spread
MS="${MS:-256 512 1024}"
PROJS="${PROJS:-qkv:7168:5120 o:5120:5120 gate_up:55296:5120 down:5120:27648}"

bench() {  # binary M N K -> median-of-REPS us
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

mkdir -p "$(dirname "${OUT}")"
echo "proj,tileM,mtiles,M,N,K,base_us,band_us,delta_pct" > "${OUT}"

BASE="${LAB_ROOT}/build/midgroup_w4a8_gemm_m128_g1024_asym_test"
BAND="${LAB_ROOT}/build/midgroup_w4a8_gemm_m128_g1024_asym_band_test"
for m in ${MS}; do
    for proj in ${PROJS}; do
        IFS=: read -r pname N K <<< "${proj}"
        b=$(bench "${BASE}" "${m}" "${N}" "${K}")
        d=$(bench "${BAND}" "${m}" "${N}" "${K}")
        pct=""
        if [ -n "${b}" ] && [ -n "${d}" ]; then
            pct=$(awk -v a="${b}" -v c="${d}" 'BEGIN{printf "%.2f", (c-a)/a*100}')
        fi
        echo "${pname},128,$((m/128)),${m},${N},${K},${b:-FAILED},${d:-FAILED},${pct:-NA}" \
            | tee -a "${OUT}"
    done
done
echo; echo "wrote ${OUT}"
