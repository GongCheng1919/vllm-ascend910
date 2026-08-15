#include "aclrtlaunch_midgroup_quant_a_g1024.h"
#include "quant_harness.h"

int main(int argc, char** argv)
{
    return qa::Run(argc, argv, 1024, "midgroup_quant_a_g1024",
        [](uint32_t blockDim, aclrtStream stream, void* x, void* ahi, void* alo,
           void* as, void* ak, uint32_t M, uint32_t K) {
            // Mp == M: this harness allocates exactly M rows and checks them
            // bit-for-bit against the CPU model, so it exercises the unpadded
            // path.  The Mp > M path is covered by the torch-side seam test
            // (npu_ops), which is where the padding is actually used.
            ACL_CHECK(aclrtlaunch_midgroup_quant_a_g1024(
                blockDim, stream, x, ahi, alo, as, ak, M, M, K));
        });
}
