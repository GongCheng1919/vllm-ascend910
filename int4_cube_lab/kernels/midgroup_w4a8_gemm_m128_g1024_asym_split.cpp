// mid-group W4A8 GEMM, TILE_M=128, GK=1024, asymmetric weight,
// MSD flush SPLIT into two half-height L0C tiles (P5_KERNEL_OPT.md D6).
// Numerically identical to midgroup_w4a8_gemm_m128_g1024_asym -- same workspace
// bytes, same AIV path -- so it must stay bit-exact under --check.
#define TILE_M_CFG 128
#define GK_CFG 1024
#define ASYM_CFG 1
#define SPLIT_MSD_FLUSH_CFG 1
#define KERNEL_NAME midgroup_w4a8_gemm_m128_g1024_asym_split
#include "midgroup_w4a8_gemm.inc"
