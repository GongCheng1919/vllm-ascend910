// mid-group W4A8 GEMM, TILE_M=128, GK=1024, asym, L2-BANDED traversal (D9)
// Pure scheduling change (tileId -> (m,n) mapping); must stay bit-exact.
#define TILE_M_CFG 128
#define GK_CFG 1024
#define ASYM_CFG 1
#define BAND_N_CFG 1
#define SPLIT_MSD_FLUSH_CFG 0
#define KERNEL_NAME midgroup_w4a8_gemm_m128_g1024_asym_band
#include "midgroup_w4a8_gemm.inc"
