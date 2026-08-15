// P5 sweep build: GM->L1 burst = 512 bytes/row (full).
#define TILE_M_CFG 16
#define GK_CFG 1024
#define ASYM_CFG 1
#define TARGET_GROUP_KB_CFG 512

#define KERNEL_NAME midgroup_w4a8_gemm_m16_g1024_asym_gkb512
#include "midgroup_w4a8_gemm.inc"
