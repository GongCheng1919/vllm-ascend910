// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef EXTENTION_CSRC_UTILS_H
#define EXTENTION_CSRC_UTILS_H
#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascendc_path {

#define DEVICE_TYPE c10::DeviceType::PrivateUse1

inline at::Tensor CopyTensorHostToDevice(const at::Tensor& cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    c10_npu::GetDevice(&deviceIndex);
    return cpuPinMemTensor.to(c10::Device(DEVICE_TYPE, deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar& cpu_scalar, at::ScalarType scalar_data_type)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

inline void *ConvertType(const at::Tensor &at_tensor)
{
    // data_ptr() = storage base + storage_offset * itemsize.  MUST use this, not
    // storage().data() (the storage BASE): FSDP hands params out as contiguous VIEWS
    // at a nonzero storage_offset into a flat buffer, so a base-only read fetches the
    // wrong bytes -> deterministic garbage (see n4_probe_storage_offset.py, was -82 dB).
    // Host-side pointer add only; no copy, no extra memory.
    return at_tensor.data_ptr();
}

template <typename T> T ConvertType(T value)
{
    return value;
}

template <typename... Ts> constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

// --- host-side alignment helpers ---------------------------------------------
// The mid-group kernels require 128-aligned M/N (and group-aligned K).  These
// pad/slice helpers keep that alignment a HOST concern (op-impl), so the Python
// op wrappers / model never see padding.  Zero-padding a quant input leaves every
// per-group absmax unchanged (zeros don't move the max) and contributes 0 to every
// GEMM group dot-product, so the sliced logical outputs are bit-exact.

// Round x up to a multiple of m (m > 0).
inline int64_t CeilTo(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

// Zero-pad (or `value`-pad) a 2-D tensor to [rows, cols] (>= current dims).
// constant_pad_nd order is {last_dim_left, last_dim_right, dim0_left, dim0_right}.
inline at::Tensor Pad2D(const at::Tensor& t, int64_t rows, int64_t cols, double value = 0.0)
{
    const int64_t pr = rows - t.size(0), pc = cols - t.size(1);
    if (pr == 0 && pc == 0) return t.contiguous();
    return at::constant_pad_nd(t, {0, pc, 0, pr}, value);
}

// Pad a 1-D tensor to length n (>= current).
inline at::Tensor Pad1D(const at::Tensor& t, int64_t n, double value = 0.0)
{
    const int64_t p = n - t.size(0);
    if (p == 0) return t.contiguous();
    return at::constant_pad_nd(t, {0, p}, value);
}

// Slice a 2-D tensor back to [rows, cols] (<= current dims) and make contiguous.
inline at::Tensor Slice2D(const at::Tensor& t, int64_t rows, int64_t cols)
{
    if (t.size(0) == rows && t.size(1) == cols) return t;
    return t.narrow(0, 0, rows).narrow(1, 0, cols).contiguous();
}


#define EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                                          \
    do {                                                                                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                      \
        auto converted_params = ConvertTypes(__VA_ARGS__);                                   \
        /* The lambda is declared `-> int`; falling off its end is UB.  At -O0 that  \
         * merely returns garbage, but at -O3 GCC treats the fall-through as        \
         * unreachable and the launch turns into a host-side SEGV with no ACL error \
         * and nothing in plog.  Return the launch status explicitly. */            \
        auto acl_call = [acl_stream, blockdim, converted_params]() -> int {                  \
            return std::apply([&](auto&&... params) -> int {                                 \
                return ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, acl_stream, params...);    \
            }, converted_params);                                                            \
        };                                                                                   \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);                         \
    } while (false)
} // namespace ascendc_path
#endif
