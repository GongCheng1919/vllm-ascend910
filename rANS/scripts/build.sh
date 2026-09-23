#!/bin/bash
# Build the AscendC kernel and the torch binding for the vllm 0.23 venv.
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
AWORK="${AWORK:-/home/gongcheng/PureInt8LLMPretraining/awork}"
VENV="${VENV:-/home/gongcheng/PureInt8LLMPretraining/vllm/.venv-023}"
SP="$(echo "$VENV"/lib/python3*/site-packages)"
A=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1 || true

# 1) kernel .so -- no torch dependency, so it is torch-version agnostic.
cmake -S "$R" -B "$R/build" -DSOC_VERSION=ascend910b4 -DRUN_MODE=npu \
      -DCMAKE_BUILD_TYPE=Release > "$R/cfg.log" 2>&1 || { tail -30 "$R/cfg.log"; exit 1; }
cmake --build "$R/build" -j16 > "$R/build.log" 2>&1 || { tail -60 "$R/build.log"; exit 1; }

# 2) binding, generated fresh from the canonical source every build.
bash "$R/scripts/gen_binding.sh" "$R/bind/binding.cpp"

# INCLUDE ORDER, and it is the opposite of the .venv (torch 2.8) build:
# torch_npu 2.10 bundles its own NEWER acl headers under
# third_party/acl/inc and its AclInterface.h includes them by an explicit
# relative path. CANN 8.5's acl_base.h otherwise wins the include guard and
# aclmdlRITask (declared in torch_npu's acl_base_rt.h, absent from CANN 8.5)
# ends up undeclared. So torch_npu's acl dir goes FIRST here.
g++ -O2 -std=c++17 -w -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
  -o "$R/bind/librans.so" "$R/bind/binding.cpp" \
  -I "$SP/torch_npu/include/third_party/acl/inc" \
  -I "$A/include" -I "$A/aarch64-linux/include/experiment/platform" \
  -I "$AWORK/csrc" \
  -I "$SP/torch/include" -I "$SP/torch/include/torch/csrc/api/include" \
  -I "$SP/torch_npu/include" -I "$SP/pybind11/include" \
  -I /usr/local/python3.10.20/include/python3.10 \
  -L "$SP/torch/lib" -ltorch -ltorch_cpu -lc10 \
  -L "$SP/torch_npu/lib" -ltorch_npu \
  -L "$A/aarch64-linux/lib64" -lascendcl -lplatform -ltiling_api -lregister \
  -L "$R/build/lib" -lans_fixed_decode_gemm_kernel \
  -Wl,-rpath,"$R/build/lib" -Wl,-rpath,"$SP/torch_npu/lib" \
  > "$R/bind.log" 2>&1 || { tail -40 "$R/bind.log"; exit 1; }
echo "[build] OK  $R/bind/librans.so"
