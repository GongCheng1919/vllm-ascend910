#!/bin/bash
# run.sh — run the raw GEMM harness.
#   ./run.sh --op s8s8 --m 128 --n 5120 --k 5120 --check --perf
#   ./run.sh --sweep --check --perf
set -e
cd "$(dirname "$0")/.."
source /usr/local/Ascend/cann-8.5.0/set_env.sh
exec ./build/gemm_test "$@"
