#!/bin/bash
# scripts/run.sh <op_name> [args...]
# Example: bash scripts/run.sh mid_group_gemm_fwd --rows 4 --cols 1024 --check
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

exec "${BIN}" "$@"
