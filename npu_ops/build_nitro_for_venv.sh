#!/bin/bash
# Build nitro's AscendC kernels against vLLM's .venv torch, into OUR tree.
#
# nitro ships a prebuilt libnitro_npu_ops.so compiled for the SYSTEM torch
# (2.7.1).  Loading it from the .venv (torch 2.8) registers the ops under the
# wrong dispatch key -- the impls show up as `MAIA:` instead of `PrivateUse1:`
# and every npu tensor is rejected as a CPU tensor.
#
# nitro's own setup.py would write the rebuilt .so back to
# nitro/ops/ascend/lib/, clobbering the copy nitro itself uses under the system
# torch.  So we drive cmake directly with our own output directory and never
# touch nitro's tree.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NITRO_CSRC="${NITRO_CSRC:-/home/gongcheng/PureInt8LLMPretraining/nitro-workspace/nitro/nitro/csrc}"
PYTHON="${PYTHON:-${HERE}/../.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-${HERE}/build-nitro-venv}"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-8.5.0}"
    [ -f "${CANN_ROOT}/set_env.sh" ] && source "${CANN_ROOT}/set_env.sh"
fi

TORCH_PATH=$("${PYTHON}" -c "import os,torch;print(os.path.dirname(torch.__file__))" 2>/dev/null | tail -1)
TORCH_NPU_PATH=$("${PYTHON}" -c "import os,torch_npu;print(os.path.dirname(torch_npu.__file__))" 2>/dev/null | tail -1)
ABI=$("${PYTHON}" -c "import torch;print(int(torch.compiled_with_cxx11_abi()))" 2>/dev/null | tail -1)
echo "nitro csrc = ${NITRO_CSRC}"
echo "torch      = ${TORCH_PATH}"
echo "abi        = ${ABI}"

mkdir -p "${BUILD_DIR}/lib" && cd "${BUILD_DIR}"
cmake "${NITRO_CSRC}" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${BUILD_DIR}/lib" \
    -DTORCH_PATH="${TORCH_PATH}" \
    -DTORCH_NPU_PATH="${TORCH_NPU_PATH}" \
    -DGLIBCXX_USE_CXX11_ABI="${ABI}" >/dev/null
make -j"${MAX_JOBS:-$(nproc)}"

echo
echo "Built: $(ls -1 "${BUILD_DIR}/lib"/*.so 2>/dev/null || echo '(none)')"
