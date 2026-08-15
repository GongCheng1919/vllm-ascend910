// mid-group W4A8 GEMM, TILE_M=64, GK=1024, asymmetric weight.
#define TILE_M_CFG 64
#define GK_CFG 1024
#define ASYM_CFG 1
#define KERNEL_NAME midgroup_w4a8_gemm_m64_g1024_asym
#include "midgroup_w4a8_gemm.inc"
