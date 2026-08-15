// gemm_s8s8.cpp — INT8 x INT8 -> INT32 raw GEMM (hardware s8s8 path)
#include "gemm_raw.h"

extern "C" __global__ __aicore__ void gemm_s8s8(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t N, uint32_t K)
{
    gemm_raw::GemmRawKernel<int8_t, int32_t>(a, b, c, M, N, K);
}
