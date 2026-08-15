// P5 sweep build: GM->L1 burst = 1536 bytes/row (dmaonly).
#define TILE_M_CFG 16
#define GK_CFG 1024
#define ASYM_CFG 1
#define TARGET_GROUP_KB_CFG 1536
#define ABLATE_AIV_CFG 1
#define ABLATE_AIC_CFG 2
#define KERNEL_NAME midgroup_w4a8_gemm_m16_g1024_asym_dmaonly_gkb1536
#include "midgroup_w4a8_gemm.inc"
