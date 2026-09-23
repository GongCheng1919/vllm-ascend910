#!/bin/bash
R="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/home/gongcheng/PureInt8LLMPretraining/vllm/.venv-023}"
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1 || true
cd "$R" && exec "$VENV/bin/python" "$1" "$R/bind/librans.so" "${@:2}"
