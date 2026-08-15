#include "aclrtlaunch_midgroup_w4a8_gemm_m16_g1024_asym_dmaonly_gkb512.h"
#include "mg_harness.h"

int main(int argc, char** argv)
{
    return mg::Run(argc, argv, 16, 1024, 512, "midgroup_w4a8_gemm_m16_g1024_asym_dmaonly_gkb512",
        [](uint32_t blockDim, aclrtStream stream, void* ahi, void* alo, void* as,
           void* w, void* ws, void* ks, void* wz, void* ak, void* y, void* wsp,
           uint32_t M, uint32_t N, uint32_t K) {
            ACL_CHECK(aclrtlaunch_midgroup_w4a8_gemm_m16_g1024_asym_dmaonly_gkb512(
                blockDim, stream, ahi, alo, as, w, ws, ks, wz, ak, y, wsp, M, N, K));
        }, /*asym=*/true);
}
