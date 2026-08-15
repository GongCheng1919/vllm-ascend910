// main_gemm_bf16.cpp — BF16 x BF16 GEMM harness.
#include "aclrtlaunch_gemm_bf16.h"
#include "gemm_driver.h"

int main(int argc, char** argv)
{
    auto launch = [](uint32_t blockDim, aclrtStream stream, void* a, void* b, void* c,
                        void* workspace, uint32_t M, uint32_t N, uint32_t K) {
        ACL_CHECK(aclrtlaunch_gemm_bf16(blockDim, stream, a, b, c, workspace, M, N, K));
    };
    return RunGemm<uint16_t, uint16_t, GemmDType::BF16, false>(argc, argv, "gemm_bf16", launch, RefBf16Gemm);
}
