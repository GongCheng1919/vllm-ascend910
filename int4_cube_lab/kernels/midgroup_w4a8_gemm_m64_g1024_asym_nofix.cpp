// P5 ABLATION BUILD (ABLATE_AIV_CFG=1, ABLATE_AIC_CFG=1) -- GARBAGE output.
// TILE_M=64 twin: at this tile the L0C tile is 64 KB so kL0CDepth is ALREADY 2
// (cross-group double buffering, the structure mid_group_gemm_fwd.cpp relies on).
// Purpose: does cross-group L0C depth-2 actually hide the flush?  Timing only.
#define TILE_M_CFG 64
#define GK_CFG 1024
#define ASYM_CFG 1
#define ABLATE_AIV_CFG 1
#define ABLATE_AIC_CFG 1
#define KERNEL_NAME midgroup_w4a8_gemm_m64_g1024_asym_nofix
#include "midgroup_w4a8_gemm.inc"
