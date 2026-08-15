#include "aclrtlaunch_perchannel_w4a8_gemm.h"
#include "w4a8_harness.h"

int main(int argc, char** argv)
{
    return w4a8::Run(argc, argv, 128, 256, "perchannel_w4a8_gemm",
        [](uint32_t blockDim, aclrtStream stream, void* ahi, void* alo, void* as,
           void* w, void* ws, void* ks, void* y, void* wsp,
           uint32_t M, uint32_t N, uint32_t K) {
            ACL_CHECK(aclrtlaunch_perchannel_w4a8_gemm(
                blockDim, stream, ahi, alo, as, w, ws, ks, y, wsp, M, N, K));
        });
}
