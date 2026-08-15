#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/common.sh"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_qwq32b_4npu}"
RUN_DIR="${PROJECT_ROOT}/results/${RUN_ID}"
PORT="${PORT:-8000}"
PRECISIONS="${PRECISIONS:-bf16 w8a8}"
PARALLELISMS="${PARALLELISMS:-tp4 pp4}"
SCENARIOS="${SCENARIOS:-concurrency long_context}"
REPEATS="${REPEATS:-1}"
SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-1800}"
LONG_CONTEXT_CONCURRENCY="${LONG_CONTEXT_CONCURRENCY:-4}"
CONCURRENCY_POINTS="${CONCURRENCY_POINTS:-1 4 8 16 32 64}"
LONG_CONTEXT_POINTS="${LONG_CONTEXT_POINTS:-4096 8192 16384 32768 40960 65536}"
HIGH_CONCURRENCY_MAX_NUM_BATCHED_TOKENS="${HIGH_CONCURRENCY_MAX_NUM_BATCHED_TOKENS:-8192}"
LONG_CONTEXT_MAX_NUM_BATCHED_TOKENS="${LONG_CONTEXT_MAX_NUM_BATCHED_TOKENS:-8192}"
DEFAULT_LONG_CONTEXT_HF_OVERRIDES='{"max_position_embeddings":131072,"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}'
LONG_CONTEXT_HF_OVERRIDES="${LONG_CONTEXT_HF_OVERRIDES-${DEFAULT_LONG_CONTEXT_HF_OVERRIDES}}"
READY_CHECK_TIMEOUT_SECONDS="${READY_CHECK_TIMEOUT_SECONDS:-60}"
EXECUTION_MODE="${EXECUTION_MODE:-graph}"
SELECTED_NPU_IDS="${SELECTED_NPU_IDS:-2 3 6 7}"

mkdir -p "${RUN_DIR}"

capture_npu_info() {
  local output_path="$1"
  local attempt
  for attempt in $(seq 1 5); do
    if npu-smi info >"${output_path}"; then
      return 0
    fi
    echo "npu-smi snapshot attempt ${attempt} failed; retrying" >&2
    sleep 5
  done
  echo "Unable to capture NPU state after 5 attempts" >&2
  return 1
}

cat >"${RUN_DIR}/run.env" <<EOF
run_id=${RUN_ID}
date_utc=$(date -u --iso-8601=seconds)
precisions=${PRECISIONS}
parallelisms=${PARALLELISMS}
scenarios=${SCENARIOS}
repeats=${REPEATS}
execution_mode=${EXECUTION_MODE}
prefix_caching=disabled
visible_devices=${ASCEND_RT_VISIBLE_DEVICES}
hccl_op_expansion_mode=${HCCL_OP_EXPANSION_MODE}
selected_npu_ids=${SELECTED_NPU_IDS}
high_concurrency_input_tokens=1024
high_concurrency_output_tokens=256
high_concurrency_points=${CONCURRENCY_POINTS// /,}
long_context_total_tokens=${LONG_CONTEXT_POINTS// /,}
long_context_output_tokens=128
long_context_concurrency=${LONG_CONTEXT_CONCURRENCY}
high_concurrency_max_num_batched_tokens=${HIGH_CONCURRENCY_MAX_NUM_BATCHED_TOKENS}
long_context_max_num_batched_tokens=${LONG_CONTEXT_MAX_NUM_BATCHED_TOKENS}
long_context_hf_overrides=${LONG_CONTEXT_HF_OVERRIDES:-none}
native_model_limit=40960
EOF

"${VENV_DIR}/bin/pip" freeze >"${RUN_DIR}/pip-freeze.txt"
capture_npu_info "${RUN_DIR}/npu-before.txt"

SERVER_PID=""
MONITOR_PID=""

stop_server() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  MONITOR_PID=""
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
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

wait_for_server() {
  local log_file="$1"
  local elapsed=0
  until curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Server exited before becoming healthy: ${log_file}" >&2
      tail -n 120 "${log_file}" >&2 || true
      return 1
    fi
    if (( elapsed >= SERVER_TIMEOUT_SECONDS )); then
      echo "Server startup timed out after ${elapsed}s: ${log_file}" >&2
      return 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
}

result_is_complete() {
  local result_path="$1"
  [[ -f "${result_path}" ]] || return 1
  "${VENV_DIR}/bin/python" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(not (d.get("completed", 0) > 0 and d.get("failed", 0) == 0))' \
    "${result_path}"
}

config_is_complete() {
  local config_dir="$1"
  local scenario="$2"
  local points
  if [[ "${scenario}" == "concurrency" ]]; then
    points="${CONCURRENCY_POINTS}"
  else
    points="${LONG_CONTEXT_POINTS}"
  fi
  for point in ${points}; do
    for repeat in $(seq 1 "${REPEATS}"); do
      if ! result_is_complete \
        "${config_dir}/client/${scenario}_${point}_r${repeat}.json"; then
        return 1
      fi
    done
  done
}

run_point() {
  local precision="$1"
  local parallelism="$2"
  local scenario="$3"
  local x_value="$4"
  local input_tokens="$5"
  local output_tokens="$6"
  local concurrency="$7"
  local num_prompts="$8"
  local repeat="$9"
  local config_dir="${10}"
  local served_model="qwq-32b-${precision}"
  local result_file="${scenario}_${x_value}_r${repeat}.json"
  local client_log="${config_dir}/client/${scenario}_${x_value}_r${repeat}.log"

  if result_is_complete "${config_dir}/client/${result_file}"; then
    echo "Skipping completed point: ${parallelism}/${precision}/${scenario}/${x_value}/r${repeat}"
    return
  fi

  "${VENV_DIR}/bin/vllm" bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${PORT}" \
    --endpoint /v1/completions \
    --model "${served_model}" \
    --tokenizer "${PROJECT_ROOT}/models/QwQ-32B" \
    --dataset-name random \
    --seed 20260729 \
    --num-prompts "${num_prompts}" \
    --random-input-len "${input_tokens}" \
    --random-output-len "${output_tokens}" \
    --random-range-ratio 0 \
    --request-rate inf \
    --max-concurrency "${concurrency}" \
    --ignore-eos \
    --num-warmups 1 \
    --ready-check-timeout-sec "${READY_CHECK_TIMEOUT_SECONDS}" \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --result-dir "${config_dir}/client" \
    --result-filename "${result_file}" \
    --metadata \
      "run_id=${RUN_ID}" \
      "precision=${precision}" \
      "parallelism=${parallelism}" \
      "scenario=${scenario}" \
      "x_value=${x_value}" \
      "input_tokens=${input_tokens}" \
      "output_tokens=${output_tokens}" \
      "repeat=${repeat}" \
      "execution_mode=${EXECUTION_MODE}" \
      "hf_overrides=${HF_OVERRIDES_JSON:-none}" \
    2>&1 | tee "${client_log}"

  if ! result_is_complete "${config_dir}/client/${result_file}"; then
    echo "Benchmark did not produce a complete result: ${result_file}" >&2
    return 1
  fi
}

for scenario in ${SCENARIOS}; do
  case "${scenario}" in
    concurrency)
      MAX_MODEL_LEN=2048
      MAX_NUM_BATCHED_TOKENS="${HIGH_CONCURRENCY_MAX_NUM_BATCHED_TOKENS}"
      HF_OVERRIDES_JSON=""
      ;;
    long_context)
      MAX_MODEL_LEN=65536
      MAX_NUM_BATCHED_TOKENS="${LONG_CONTEXT_MAX_NUM_BATCHED_TOKENS}"
      HF_OVERRIDES_JSON="${LONG_CONTEXT_HF_OVERRIDES}"
      ;;
    *)
      echo "Unknown scenario: ${scenario}" >&2
      exit 2
      ;;
  esac

  for parallelism in ${PARALLELISMS}; do
    for precision in ${PRECISIONS}; do
      config_name="${parallelism}_${precision}_${scenario}"
      config_dir="${RUN_DIR}/${config_name}"
      mkdir -p "${config_dir}/client"
      server_log="${config_dir}/server.log"

      if config_is_complete "${config_dir}" "${scenario}"; then
        echo "Skipping completed configuration: ${config_name}"
        printf 'completed\n' >"${config_dir}/status.txt"
        continue
      fi

      echo "Starting ${config_name}"
      export EXECUTION_MODE MAX_NUM_BATCHED_TOKENS HF_OVERRIDES_JSON
      setsid "${PROJECT_ROOT}/scripts/serve.sh" \
        "${precision}" "${parallelism}" "${MAX_MODEL_LEN}" "${PORT}" \
        >"${server_log}" 2>&1 &
      SERVER_PID=$!
      printf '%s\n' "${SERVER_PID}" >"${config_dir}/server.pid"

      "${VENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/monitor_npu.py" \
        --watch-pid "${SERVER_PID}" \
        --output "${config_dir}/npu-samples.csv" &
      MONITOR_PID=$!

      if ! wait_for_server "${server_log}"; then
        printf 'startup_failed\n' >"${config_dir}/status.txt"
        stop_server
        continue
      fi
      printf 'server_ready\n' >"${config_dir}/status.txt"
      curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" \
        >"${config_dir}/models.json"

      if [[ "${scenario}" == "concurrency" ]]; then
        for concurrency in ${CONCURRENCY_POINTS}; do
          num_prompts=$((concurrency * 4))
          if (( num_prompts < 8 )); then
            num_prompts=8
          fi
          for repeat in $(seq 1 "${REPEATS}"); do
            run_point \
              "${precision}" "${parallelism}" "${scenario}" \
              "${concurrency}" 1024 256 "${concurrency}" \
              "${num_prompts}" "${repeat}" "${config_dir}"
          done
        done
      else
        for total_tokens in ${LONG_CONTEXT_POINTS}; do
          input_tokens=$((total_tokens - 128))
          num_prompts=$((LONG_CONTEXT_CONCURRENCY * 2))
          for repeat in $(seq 1 "${REPEATS}"); do
            run_point \
              "${precision}" "${parallelism}" "${scenario}" \
              "${total_tokens}" "${input_tokens}" 128 \
              "${LONG_CONTEXT_CONCURRENCY}" "${num_prompts}" \
              "${repeat}" "${config_dir}"
          done
        done
      fi

      printf 'completed\n' >"${config_dir}/status.txt"
      stop_server
      sleep 10
    done
  done
done

capture_npu_info "${RUN_DIR}/npu-after.txt"
"${VENV_DIR}/bin/python" "${PROJECT_ROOT}/benchmarks/summarize.py" "${RUN_DIR}"
echo "Benchmark complete: ${RUN_DIR}"
