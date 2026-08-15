#!/bin/bash
# scripts/build.sh — cmake + make for the entire lab
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LAB_ROOT}"

# Source CANN env if not already
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-8.5.0}"
    if [ -f "${CANN_ROOT}/set_env.sh" ]; then
        # shellcheck disable=SC1091
        source "${CANN_ROOT}/set_env.sh"
    else
        echo "ERROR: ASCEND_HOME_PATH not set and ${CANN_ROOT}/set_env.sh missing" >&2
        exit 1
    fi
fi

# Optional: target specific test
TARGET="${1:-}"

mkdir -p build
cd build
if [ ! -f Makefile ]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

if [ -n "${TARGET}" ]; then
    make -j"$(nproc)" "${TARGET}_test"
else
    make -j"$(nproc)"
fi

echo
echo "Built:"
ls -1 ./*_test 2>/dev/null || echo "  (no _test binaries — check build output)"
