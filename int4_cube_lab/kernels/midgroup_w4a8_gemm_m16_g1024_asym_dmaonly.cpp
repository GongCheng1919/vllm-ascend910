// P5 ABLATION BUILD (ABLATE_AIC_CFG=2, AIV handshake-only) -- output is GARBAGE.
// Timing-only twin of midgroup_w4a8_gemm_m16_g1024_asym.
// See MGCKPT/P5_KERNEL_OPT.md §4.  Do NOT run with --check.
#define TILE_M_CFG 16
#define GK_CFG 1024
#define ASYM_CFG 1
#define ABLATE_AIV_CFG 1
#define ABLATE_AIC_CFG 2
#define KERNEL_NAME midgroup_w4a8_gemm_m16_g1024_asym_dmaonly
#include "midgroup_w4a8_gemm.inc"
