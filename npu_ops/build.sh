#!/bin/bash
# Build libvllm_w4a8_npu_ops.so -- registers torch.ops.npu.midgroup_{w4a8_gemm,quant_a}.
#
#   bash npu_ops/build.sh
#
# Build for a specific interpreter (vLLM lives in .venv with torch 2.8):
#   PYTHON="$PWD/.venv/bin/python" BUILD_DIR=build-venv bash npu_ops/build.sh
# PYTHON must be ABSOLUTE: this script cd's into npu_ops/, so a relative
# `.venv/bin/python` resolves to `npu_ops/.venv/...` and the build dies with a
# bare `EXIT=127` and an otherwise empty log.
# The ABI/torch version must match the interpreter that will load the .so.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-8.5.0}"
    [ -f "${CANN_ROOT}/set_env.sh" ] && source "${CANN_ROOT}/set_env.sh"
fi

PYTHON="${PYTHON:-python3}"
TORCH_PATH=$("${PYTHON}" -c "import os,torch; print(os.path.dirname(torch.__file__))" 2>/dev/null | tail -1)
TORCH_NPU_PATH=$("${PYTHON}" -c "import os,torch_npu; print(os.path.dirname(torch_npu.__file__))" 2>/dev/null | tail -1)
ABI=$("${PYTHON}" -c "import torch; print(int(torch.compiled_with_cxx11_abi()))" 2>/dev/null | tail -1)
echo "torch=${TORCH_PATH}"
echo "torch_npu=${TORCH_NPU_PATH}"
echo "cxx11_abi=${ABI}"

BUILD_DIR="${BUILD_DIR:-build}"
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
cmake .. \
    -DTORCH_PATH="${TORCH_PATH}" \
    -DTORCH_NPU_PATH="${TORCH_NPU_PATH}" \
    -DGLIBCXX_USE_CXX11_ABI="${ABI}" \
    -DCMAKE_BUILD_TYPE=Release
make -j"${MAX_JOBS:-$(nproc)}"

echo
echo "Built: $(ls -1 lib*.so 2>/dev/null || echo '(none -- check output)')"
