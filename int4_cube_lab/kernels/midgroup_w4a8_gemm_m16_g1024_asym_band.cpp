// mid-group W4A8, TILE_M=16, GK=1024, asym, L2-BANDED (D9 at decode M). Pure scheduling change; must stay bit-exact.
#define TILE_M_CFG 16
#define GK_CFG 1024
#define ASYM_CFG 1
#define BAND_N_CFG 1
#define SPLIT_MSD_FLUSH_CFG 0
#define KERNEL_NAME midgroup_w4a8_gemm_m16_g1024_asym_band
#include "midgroup_w4a8_gemm.inc"
