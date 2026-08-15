#!/bin/bash
# scripts/profile.sh <op_name> [args...]
# Wraps the test binary with msprof to capture cube/vector utilization.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <op_name> [args...]" >&2
    exit 1
fi

OP="$1"; shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${LAB_ROOT}/build/${OP}_test"

if [ ! -x "${BIN}" ]; then
    echo "ERROR: ${BIN} not found. Run 'bash scripts/build.sh ${OP}' first." >&2
    exit 1
fi

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-8.5.0}"
    # shellcheck disable=SC1091
    [ -f "${CANN_ROOT}/set_env.sh" ] && source "${CANN_ROOT}/set_env.sh"
fi

PROF_OUT="${LAB_ROOT}/prof_out_${OP}_$(date +%s)"
mkdir -p "${PROF_OUT}"

msprof \
    --application="${BIN} $*" \
    --output="${PROF_OUT}" \
    --aic-metrics=PipeUtilization,ArithmeticUtilization,Memory \
    --aicpu=on \
    --sys-hardware-mem=on

echo
echo "Profile artifacts in: ${PROF_OUT}"
echo "Key files:"
ls -1 "${PROF_OUT}" 2>/dev/null | head -20
