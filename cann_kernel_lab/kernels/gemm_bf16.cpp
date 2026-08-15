// gemm_bf16.cpp — BF16 x BF16 -> FP32 raw GEMM (hardware bf16 path)
#include "gemm_raw.h"

extern "C" __global__ __aicore__ void gemm_bf16(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t N, uint32_t K)
{
    gemm_raw::GemmRawKernel<bfloat16_t, float>(a, b, c, M, N, K);
}
