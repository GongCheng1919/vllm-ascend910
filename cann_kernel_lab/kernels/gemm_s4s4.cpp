// gemm_s4s4.cpp — INT4 x INT4 -> INT32 raw GEMM (hardware mad_s4 path)
#include "gemm_raw.h"

extern "C" __global__ __aicore__ void gemm_s4s4(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t N, uint32_t K)
{
    gemm_raw::GemmRawKernel<int4b_t, int32_t>(a, b, c, M, N, K);
}
