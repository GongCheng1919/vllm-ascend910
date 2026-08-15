// main_gemm_int4.cpp — INT4 x INT4 GEMM harness (packed 2 nibbles per byte, int32 output).
#include "aclrtlaunch_gemm_int4.h"
#include "gemm_driver.h"

int main(int argc, char** argv)
{
    auto launch = [](uint32_t blockDim, aclrtStream stream, void* a, void* b, void* c,
                        void* workspace, uint32_t M, uint32_t N, uint32_t K) {
        ACL_CHECK(aclrtlaunch_gemm_int4(blockDim, stream, a, b, c, workspace, M, N, K));
    };
    return RunGemm<uint8_t, int32_t, GemmDType::INT4, true>(argc, argv, "gemm_int4", launch, RefInt4Gemm);
}
