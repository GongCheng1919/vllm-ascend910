#include "aclrtlaunch_midgroup_w4a8_gemm_m64_g512_asym_nofix.h"
#include "mg_harness.h"

int main(int argc, char** argv)
{
    return mg::Run(argc, argv, 64, 512, 512, "midgroup_w4a8_gemm_m64_g512_asym_nofix",
        [](uint32_t blockDim, aclrtStream stream, void* ahi, void* alo, void* as,
           void* w, void* ws, void* ks, void* wz, void* ak, void* y, void* wsp,
           uint32_t M, uint32_t N, uint32_t K) {
            ACL_CHECK(aclrtlaunch_midgroup_w4a8_gemm_m64_g512_asym_nofix(
                blockDim, stream, ahi, alo, as, w, ws, ks, wz, ak, y, wsp, M, N, K));
        }, /*asym=*/true);
}
