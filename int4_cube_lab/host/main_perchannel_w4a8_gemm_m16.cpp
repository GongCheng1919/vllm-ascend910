#include "aclrtlaunch_perchannel_w4a8_gemm_m16.h"
#include "w4a8_harness.h"

int main(int argc, char** argv)
{
    return w4a8::Run(argc, argv, 16, 512, "perchannel_w4a8_gemm_m16",
        [](uint32_t blockDim, aclrtStream stream, void* ahi, void* alo, void* as,
           void* w, void* ws, void* ks, void* y, void* wsp,
           uint32_t M, uint32_t N, uint32_t K) {
            ACL_CHECK(aclrtlaunch_perchannel_w4a8_gemm_m16(
                blockDim, stream, ahi, alo, as, w, ws, ks, y, wsp, M, N, K));
        });
}
