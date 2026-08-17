// P5 ABLATION BUILD (ABLATE_AIV_CFG=1, ABLATE_AIC_CFG=1) -- output is GARBAGE. Timing-only twin of midgroup_w4a8_gemm_m128_g512_asym. Do NOT run with --check.
#define TILE_M_CFG 128
#define GK_CFG 512
#define ASYM_CFG 1
#define ABLATE_AIV_CFG 1
#define ABLATE_AIC_CFG 1
#define KERNEL_NAME midgroup_w4a8_gemm_m128_g512_asym_nofix
#include "midgroup_w4a8_gemm.inc"
