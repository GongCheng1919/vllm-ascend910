#!/bin/bash
# P6 correctness gate.  Nothing gets measured, defaulted or synced to npu_ops
# until every new build passes --check --stress here (P5 §2 hard gate).
#
#   * the GK=512/256 ASYM builds are new (K,GK) configurations of an existing
#     template -- the zero-point term and the group-major scale layout have never
#     run at these group counts;
#   * the band builds at TILE_M=16/64 are a pure scheduling change and must be
#     BIT-EXACT against the unbanded build, not merely within SNR.
#
# N is held at 1280 (still a multiple of TILE_N=128) to keep the CPU reference
# affordable; K carries the real TP-split values, which is where the risk is.
#
# usage: check_p6.sh [stress]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
STRESS="${1:-10}"
N="${N:-1280}"

fail=0
check() {  # name M K [N]
    local name="$1" m="$2" k="$3" N="${4:-${N}}"
    local bin="${LAB_ROOT}/build/${name}_test"
    if [ ! -x "${bin}" ]; then
        echo "MISSING  ${name}"; fail=$((fail+1)); return
    fi
    local out
    out=$("${bin}" --rows "${m}" --cols "${N}" --k "${k}" --check --stress "${STRESS}" 2>&1)
    if echo "${out}" | grep -qiE "mismatches=0|PASS"; then
        local mm
        mm=$(echo "${out}" | grep -oE "mismatches=[0-9]+" | sort -u | tr '\n' ' ')
        if echo "${out}" | grep -qE "mismatches=[1-9]"; then
            echo "FAIL     ${name}  M=${m} K=${k}  ${mm}"; fail=$((fail+1))
        else
            echo "PASS     ${name}  M=${m} K=${k}  ${mm}"
        fi
    else
        echo "FAIL     ${name}  M=${m} K=${k}"
        echo "${out}" | tail -5 | sed 's/^/         | /'
        fail=$((fail+1))
    fi
}

echo "=== P6 B: asym GK=512 (TP=2 shapes) ==="
check midgroup_w4a8_gemm_m16_g512_asym    16  2560     # o_proj   TP=2
check midgroup_w4a8_gemm_m16_g512_asym    16  13824    # down     TP=2
check midgroup_w4a8_gemm_m64_g512_asym    64  2560
check midgroup_w4a8_gemm_m64_g512_asym    64  13824
check midgroup_w4a8_gemm_m128_g512_asym   512 2560
check midgroup_w4a8_gemm_m128_g512_asym   512 13824

echo
echo "=== P6 B: asym GK=256 (TP=4 shapes) ==="
check midgroup_w4a8_gemm_m16_g256_asym    16  1280     # o_proj   TP=4
check midgroup_w4a8_gemm_m16_g256_asym    16  6912     # down     TP=4
check midgroup_w4a8_gemm_m64_g256_asym    64  1280
check midgroup_w4a8_gemm_m64_g256_asym    64  6912
check midgroup_w4a8_gemm_m128_g256_asym   512 1280
check midgroup_w4a8_gemm_m128_g256_asym   512 6912

echo
echo "=== P6 A: banded traversal at decode TILE_M (must be bit-exact) ==="
# N MATTERS HERE.  Banding only engages when N*Kb > kBandCapBytes (32 MB); below
# that superN degenerates to the flat mapping and the banded build is byte-identical
# to the base one, so a small-N --check exercises NOTHING.  These four shapes are
# the real ones: gate_up N*Kb = 141.6 MB, down N*Kb = 70.8 MB.
check midgroup_w4a8_gemm_m16_g1024_asym_band  16  5120  55296   # gate_up, bands
check midgroup_w4a8_gemm_m16_g1024_asym_band  16  27648  5120   # down,    bands
check midgroup_w4a8_gemm_m64_g1024_asym_band  64  5120  55296
check midgroup_w4a8_gemm_m64_g1024_asym_band  64  27648  5120
# Controls: below the cap the mapping must degenerate and still pass.
check midgroup_w4a8_gemm_m16_g1024_asym_band  16  5120   7168   # qkv, no-op
check midgroup_w4a8_gemm_m64_g1024_asym_band  64  5120   5120   # o,   no-op

echo
if [ "${fail}" -eq 0 ]; then echo "ALL PASS"; else echo "${fail} FAILURE(S)"; fi
exit "${fail}"
