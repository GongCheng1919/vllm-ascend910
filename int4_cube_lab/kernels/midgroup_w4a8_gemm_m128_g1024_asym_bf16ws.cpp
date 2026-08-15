// SPARE KERNEL -- mid-group W4A8, TILE_M=128, GK=1024, asym, L2-banded,
// with a BF16 WORKSPACE FLUSH (D10).
//
// *** NOT BIT-EXACT with fakequant_lab.  NOT a drop-in for the shipped path. ***
// The cube converts each int32 partial to bf16 on the way out, halving the
// workspace traffic in both directions.  Kept built and measured so it can be
// switched on the day the accuracy budget for it has been paid (P2), without
// re-deriving anything.  See MGCKPT/P5_KERNEL_OPT.md D10.
#define TILE_M_CFG 128
#define GK_CFG 1024
#define ASYM_CFG 1
#define BAND_N_CFG 1
#define BF16_FLUSH_CFG 1
#define KERNEL_NAME midgroup_w4a8_gemm_m128_g1024_asym_bf16ws
#include "midgroup_w4a8_gemm.inc"
