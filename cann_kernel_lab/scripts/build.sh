#!/bin/bash
# build.sh — build all three raw GEMM kernels + host harness.
set -e
cd "$(dirname "$0")/.."
source /usr/local/Ascend/cann-8.5.0/set_env.sh
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null
make -j"$(nproc)" gemm_test
echo "[build] OK -> build/gemm_test"
