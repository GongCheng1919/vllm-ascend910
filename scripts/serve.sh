#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

usage() {
  echo "Usage: $0 <bf16|w8a8|w4a8> <tp4|pp4> <max-model-len> [port] [extra vLLM args...]" >&2
}

if (( $# < 3 )); then
  usage
  exit 2
fi

PRECISION="$1"
PARALLELISM="$2"
MAX_MODEL_LEN="$3"
PORT="${4:-8000}"
if (( $# >= 4 )); then
  shift 4
else
  shift 3
fi
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

case "${PRECISION}" in
  bf16)
    MODEL_PATH="${PROJECT_ROOT}/models/QwQ-32B"
    SERVED_MODEL_NAME="qwq-32b-bf16"
    QUANT_ARGS=()
    ;;
  w8a8)
    MODEL_PATH="${PROJECT_ROOT}/models/QwQ-32B-W8A8"
    SERVED_MODEL_NAME="qwq-32b-w8a8"
    QUANT_ARGS=(--quantization ascend)
    ;;
  w4a8)
    MODEL_PATH="${PROJECT_ROOT}/models/QwQ-32B-W4A8-Random"
    SERVED_MODEL_NAME="qwq-32b-w4a8"
    QUANT_ARGS=(--quantization ascend)
    ;;
  *)
    usage
    exit 2
    ;;
esac

case "${PARALLELISM}" in
  tp4)
    TP_SIZE=4
    PP_SIZE=1
    ;;
  pp4)
    TP_SIZE=1
    PP_SIZE=4
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Incomplete model directory: ${MODEL_PATH}" >&2
  exit 1
fi

if (( MAX_MODEL_LEN > 40960 )); then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

EXECUTION_MODE="${EXECUTION_MODE:-graph}"
MODE_ARGS=()
case "${EXECUTION_MODE}" in
  graph) ;;
  eager) MODE_ARGS=(--enforce-eager) ;;
  *)
    echo "EXECUTION_MODE must be graph or eager, found: ${EXECUTION_MODE}" >&2
    exit 2
    ;;
esac

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
HF_OVERRIDE_ARGS=()
if [[ -n "${HF_OVERRIDES_JSON:-}" ]]; then
  HF_OVERRIDE_ARGS=(--hf-overrides "${HF_OVERRIDES_JSON}")
fi

COMMAND=(
  "${VENV_DIR}/bin/vllm" serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 127.0.0.1
  --port "${PORT}"
  --dtype bfloat16
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size "${PP_SIZE}"
  --distributed-executor-backend mp
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --no-enable-prefix-caching
  --enable-chunked-prefill
  --generation-config vllm
  --disable-uvicorn-access-log
  "${QUANT_ARGS[@]}"
  "${MODE_ARGS[@]}"
  "${HF_OVERRIDE_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

printf 'Launching:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
