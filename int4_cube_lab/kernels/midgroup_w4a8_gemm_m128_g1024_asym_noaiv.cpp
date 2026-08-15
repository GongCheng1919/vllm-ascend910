// P5 ABLATION BUILD (ABLATE_AIV_CFG=1, ABLATE_AIC_CFG=0) -- output is GARBAGE.
// Timing-only twin of midgroup_w4a8_gemm_m128_g1024_asym, for the LARGE-M
// ladder (M=512/1024).  See MGCKPT/P5_KERNEL_OPT.md D4.  Do NOT run with --check.
#define TILE_M_CFG 128
#define GK_CFG 1024
#define ASYM_CFG 1
#define ABLATE_AIV_CFG 1
#define ABLATE_AIC_CFG 0
#define KERNEL_NAME midgroup_w4a8_gemm_m128_g1024_asym_noaiv
#include "midgroup_w4a8_gemm.inc"
