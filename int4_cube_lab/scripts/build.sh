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

# `ascendc_library` does not track the .inc files its .cpp wrappers include, so
# editing a .inc rebuilds NOTHING and the next test run silently grades a STALE
# kernel -- the same shape of failure as P6 D8 (a path that does no work looks
# exactly like one that works).  Touch every wrapper whose .inc is newer.
for inc in kernels/*.inc kernels/*.h; do
    [ -f "${inc}" ] || continue
    base="$(basename "${inc}")"
    for cpp in kernels/*.cpp; do
        if grep -q "\"${base}\"" "${cpp}" && [ "${inc}" -nt "${cpp}" ]; then
            echo "  [dep] ${base} is newer than ${cpp} -- touching to force a rebuild"
            touch "${cpp}"
        fi
    done
done

# Optional: target specific test
TARGET="${1:-}"

# BUILD_DIR lets a second CANN (e.g. the 9.1.0 probe in $HOME/Ascend, P10 B0) be
# graded WITHOUT clobbering the CANN 8.5.0 build tree that every P0..P10-PhaseA
# number was measured with.  A cmake cache remembers its toolkit path, so one
# build/ cannot serve two CANNs.
BUILD_DIR="${BUILD_DIR:-build}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
if [ ! -f Makefile ]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

if [ -n "${TARGET}" ]; then
    make -j"${JOBS:-$(nproc)}" "${TARGET}_test"
else
    make -j"${JOBS:-$(nproc)}"
fi

echo
echo "Built:"
ls -1 ./*_test 2>/dev/null || echo "  (no _test binaries — check build output)"
