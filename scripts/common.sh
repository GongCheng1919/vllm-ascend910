#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-8.5.0}"
ATB_ROOT="${ATB_ROOT:-/usr/local/Ascend/nnal/atb}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing virtual environment: ${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CANN_ROOT}/set_env.sh" ]]; then
  echo "Missing CANN environment script: ${CANN_ROOT}/set_env.sh" >&2
  exit 1
fi
if [[ ! -f "${ATB_ROOT}/set_env.sh" ]]; then
  echo "Missing ATB environment script: ${ATB_ROOT}/set_env.sh" >&2
  exit 1
fi

# Remove the host's application PYTHONPATH before adding the CANN runtime.
export PYTHONPATH=
set +o nounset
# shellcheck disable=SC1091
source "${CANN_ROOT}/set_env.sh"
# shellcheck disable=SC1091
source "${ATB_ROOT}/set_env.sh" --cxx_abi=1
set -o nounset

export PYTHONNOUSERSITE=1
export HOME="${VLLM_RUNTIME_HOME:-${PROJECT_ROOT}/.runtime/home}"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.runtime/cache"
export HF_HOME="${PROJECT_ROOT}/.runtime/huggingface"
export MODELSCOPE_CACHE="${PROJECT_ROOT}/.runtime/modelscope"
export MPLCONFIGDIR="${PROJECT_ROOT}/.runtime/matplotlib"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,4,5}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export VLLM_PLUGINS=ascend

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

mkdir -p \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${MODELSCOPE_CACHE}" \
  "${MPLCONFIGDIR}"
