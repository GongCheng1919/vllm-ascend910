#!/bin/bash
# P6.5 RAGGED-K gate.  The op's contract is now "any K" (kernels/mg_kgeom.h): the
# misalignment is absorbed in the op's own packed layout by zero-padding each
# quant group up to one int4 cube fractal (64 elements).  Nothing above the op
# aligns anything.
#
# WHAT EACH BLOCK IS FOR -- the shapes are chosen to separate the two mechanisms,
# because a gate that only proves "the TP shapes work" would not tell you WHICH
# part of the change is wrong when something breaks:
#
#   1. ALIGNED regression.  Must still pass, and the pre-ragged packers must be
#      reproduced byte for byte (that part is `ragged_layout_test`).  Without this
#      every number P0..P6 measured stops being comparable.
#   2. Short GROUP, no short SLICE (K=2560/13824 at GK=1024: last group 512 elem
#      = exactly one 256 B slice).  Exercises the flush-cadence change alone.
#   3. Short GROUP *and* short SLICE (K=1280/6912): adds the runtime mm.k /
#      repeatTimes / Nd2Nz dValue path.
#   4. Real PADDING (K=1057/1025/777/63/1): gPad > gReal, so the zero-fill itself
#      is under test.  This is the block that caught the quantiser's isPad bug --
#      a DataCopyPad with a non-32 B blockLen and isPad=false pulls in the
#      FOLLOWING real elements, which leaves a_scale correct and everything else
#      wrong.
#   5. Odd K, so the last real element lands in a low nibble and its partner
#      nibble is padding.
#
# M is held at the build's TILE_M.  A ragged M is a SEPARATE axis (still open) and
# mixing it in here would confuse the two; see the note on ReferenceMidGroupGemm.
#
# usage: check_ragged.sh [stress]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# BUILD_DIR: grade a second CANN (P10 B0) without clobbering the CANN 8.5.0 tree.
BUILD_DIR="${BUILD_DIR:-${LAB_ROOT}/build}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
STRESS="${1:-10}"
N="${N:-256}"

fail=0

check() {  # binary M K [N]
    local name="$1" m="$2" k="$3" nn="${4:-${N}}"
    local bin="${BUILD_DIR}/${name}_test"
    if [ ! -x "${bin}" ]; then
        echo "MISSING  ${name}"; fail=$((fail+1)); return
    fi
    local out
    out=$("${bin}" --rows "${m}" --cols "${nn}" --k "${k}" --check --stress "${STRESS}" 2>&1)
    # REFUSING = the harness rejected the shape or the reference was degenerate.
    # Treat it as a failure, never as a pass: a vacuous comparison is exactly the
    # kind of "quietly did nothing" result P6 D8 was written about.
    if echo "${out}" | grep -q "REFUSING"; then
        echo "REFUSED  ${name}  M=${m} K=${k}"; fail=$((fail+1)); return
    fi
    if echo "${out}" | grep -qE "mismatches=[1-9]" || ! echo "${out}" | grep -q "PASS"; then
        echo "FAIL     ${name}  M=${m} K=${k} N=${nn}"
        echo "${out}" | tail -6 | sed 's/^/         | /'
        fail=$((fail+1))
    else
        local lg
        lg=$(echo "${out}" | grep -oE "groups=[0-9]+ lastGK=[0-9]+" | head -1)
        echo "PASS     ${name}  M=${m} K=${k} N=${nn}  ${lg}"
    fi
}

qcheck() {  # quantiser: M K -- BYTE-exact against the CPU model of quant.py
    local m="$1" k="$2"
    local bin="${BUILD_DIR}/midgroup_quant_a_g1024_test"
    if [ ! -x "${bin}" ]; then echo "MISSING  quant_a"; fail=$((fail+1)); return; fi
    local out
    out=$("${bin}" --rows "${m}" --k "${k}" --check 2>&1)
    if echo "${out}" | grep -qE "byte mismatches=[1-9]" || ! echo "${out}" | grep -q PASS; then
        echo "FAIL     quant_a  M=${m} K=${k}"
        echo "${out}" | grep -E "^\[check\]" | sed 's/^/         | /'
        fail=$((fail+1))
    else
        echo "PASS     quant_a  M=${m} K=${k}  $(echo "${out}" | grep -oE "lastGK=[0-9]+ Kpad/2=[0-9]+" | head -1)"
    fi
}

for build in m16:16 m64:64 m128:128; do
    tag="${build%%:*}"; m="${build##*:}"
    bin="midgroup_w4a8_gemm_${tag}_g1024_asym"
    echo "=== GEMM ${tag} (TILE_M=${m}) ==="
    echo "--- 1. aligned regression"
    check "${bin}" "${m}" 5120
    check "${bin}" "${m}" 27648
    echo "--- 2. short group, full slices  (QwQ TP=2)"
    check "${bin}" "${m}" 2560
    check "${bin}" "${m}" 13824
    echo "--- 3. short group + short slice (QwQ TP=4 / TP=8)"
    check "${bin}" "${m}" 1280
    check "${bin}" "${m}" 6912
    check "${bin}" "${m}" 640
    check "${bin}" "${m}" 3456
    echo "--- 4. real padding: gPad > gReal"
    check "${bin}" "${m}" 1057    # last group 33 -> 64
    check "${bin}" "${m}" 1025    # last group  1 -> 64
    check "${bin}" "${m}" 777
    check "${bin}" "${m}" 63
    check "${bin}" "${m}" 1       # a one-element K
    echo "--- 5. odd K"
    check "${bin}" "${m}" 1023
    check "${bin}" "${m}" 5121
    echo
done

echo "=== GEMM sym path (the zero-point term compiled out) ==="
check midgroup_w4a8_gemm_m16_g1024 16 1057
check midgroup_w4a8_gemm_m16_g1024 16 6912
echo
echo "=== GEMM at other GK: QGROUP_PAD_KB / SLICES_PER_QGROUP differ ==="
check midgroup_w4a8_gemm_m16_g512_asym 16 2560
check midgroup_w4a8_gemm_m16_g512_asym 16 1057
check midgroup_w4a8_gemm_m16_g256_asym 16 1280
check midgroup_w4a8_gemm_m16_g256_asym 16 1057
echo
echo "=== quantiser (byte-exact) ==="
for k in 5120 27648 2560 13824 1280 6912 640 3456 1057 1025 777 63 1 35 1023 5121; do
    qcheck 16 "${k}"
done

echo
if [ "${fail}" -eq 0 ]; then echo "ALL PASS"; else echo "${fail} FAILURE(S)"; fi
exit "${fail}"
