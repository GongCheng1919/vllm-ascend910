// main_gemm_int8.cpp — INT8 x INT8 GEMM harness (int32 output).
#include "aclrtlaunch_gemm_int8.h"
#include "gemm_driver.h"

int main(int argc, char** argv)
{
    auto launch = [](uint32_t blockDim, aclrtStream stream, void* a, void* b, void* c,
                        void* workspace, uint32_t M, uint32_t N, uint32_t K) {
        ACL_CHECK(aclrtlaunch_gemm_int8(blockDim, stream, a, b, c, workspace, M, N, K));
    };
    return RunGemm<int8_t, int32_t, GemmDType::INT8, false>(argc, argv, "gemm_int8", launch, RefIntGemm);
}
