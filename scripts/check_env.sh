#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

BF16_MODEL="${PROJECT_ROOT}/models/QwQ-32B"
W8A8_MODEL="${PROJECT_ROOT}/models/QwQ-32B-W8A8"

echo "project_root=${PROJECT_ROOT}"
echo "cann_root=${CANN_ROOT}"
echo "atb_root=${ATB_ROOT}"
echo "visible_devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo "python=$("${VENV_DIR}/bin/python" --version 2>&1)"
echo
npu-smi info
echo

"${VENV_DIR}/bin/python" - "${BF16_MODEL}" "${W8A8_MODEL}" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import torch
import torch_npu

bf16 = Path(sys.argv[1])
w8a8 = Path(sys.argv[2])
packages = (
    "torch",
    "torch-npu",
    "vllm",
    "vllm-ascend",
    "triton-ascend",
    "transformers",
    "numpy",
)
for package in packages:
    print(f"{package}={metadata.version(package)}")

print(f"torch_file={torch.__file__}")
print(f"npu_available={torch.npu.is_available()}")
print(f"npu_count={torch.npu.device_count()}")

for path, expected_quant in ((bf16, None), (w8a8, "W8A8")):
    if not path.is_dir():
        raise SystemExit(f"missing model directory: {path}")
    config = json.loads((path / "config.json").read_text())
    print(
        f"model={path.name} layers={config['num_hidden_layers']} "
        f"native_max_len={config['max_position_embeddings']}"
    )
    if expected_quant:
        quant = json.loads(
            (path / "quant_model_description.json").read_text()
        )
        actual = quant.get("model_quant_type")
        if actual != expected_quant:
            raise SystemExit(
                f"{path.name}: expected {expected_quant}, found {actual}"
            )
        print(f"model={path.name} quantization={actual}")

if not torch.npu.is_available():
    raise SystemExit("NPU is not available")
x = torch.randn((128, 128), dtype=torch.bfloat16, device="npu:0")
y = x @ x
torch.npu.synchronize()
print(f"bf16_matmul=pass mean={float(y.float().mean().cpu()):.6f}")
PY

