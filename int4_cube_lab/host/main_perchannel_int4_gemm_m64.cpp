#include "aclrtlaunch_perchannel_int4_gemm_m64.h"
#include "gemm_harness.h"

int main(int argc, char** argv)
{
    return harness::Run(argc, argv, 4, 64, "perchannel_int4_gemm_m64",
        [](uint32_t blockDim, aclrtStream stream, void* a, void* as, void* w,
           void* ws, void* y, void* wsp, uint32_t M, uint32_t N, uint32_t K) {
            ACL_CHECK(aclrtlaunch_perchannel_int4_gemm_m64(
                blockDim, stream, a, as, w, ws, y, wsp, M, N, K));
        });
}
