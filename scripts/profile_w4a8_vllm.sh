#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_ROOT}/results/w4a8_vllm_profile}"
PORT="${PORT:-8001}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"

mkdir -p "${OUTPUT_DIR}/trace"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

export VLLM_TORCH_PROFILER_DIR="${OUTPUT_DIR}/trace"
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0
export ASCEND_GLOBAL_LOG_LEVEL=3

SERVER_PID=""
stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -9 -- "-${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap stop_server EXIT INT TERM

setsid "${PROJECT_ROOT}/scripts/serve.sh" w4a8 tp4 2048 "${PORT}" \
  >"${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" >"${OUTPUT_DIR}/server.pid"

elapsed=0
until curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 120 "${OUTPUT_DIR}/server.log" >&2 || true
    exit 1
  fi
  if (( elapsed >= STARTUP_TIMEOUT )); then
    echo "Profile server startup timed out after ${elapsed}s" >&2
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

curl --fail --silent --show-error --request POST \
  "http://127.0.0.1:${PORT}/start_profile" \
  >"${OUTPUT_DIR}/start_profile.json"

curl --fail --silent --show-error \
  "http://127.0.0.1:${PORT}/v1/completions" \
  --header "Content-Type: application/json" \
  --data '{
    "model": "qwq-32b-w4a8",
    "prompt": "Profile the quantized matrix multiply path.",
    "max_tokens": 8,
    "temperature": 0,
    "ignore_eos": true
  }' >"${OUTPUT_DIR}/completion.json"

curl --fail --silent --show-error --request POST \
  --max-time 1800 \
  "http://127.0.0.1:${PORT}/stop_profile" \
  >"${OUTPUT_DIR}/stop_profile.json"

find "${OUTPUT_DIR}/trace" -type f -printf '%P,%s\n' \
  | sort >"${OUTPUT_DIR}/trace_manifest.csv"
stop_server
SERVER_PID=""
echo "vLLM W4A8 profile complete: ${OUTPUT_DIR}"
