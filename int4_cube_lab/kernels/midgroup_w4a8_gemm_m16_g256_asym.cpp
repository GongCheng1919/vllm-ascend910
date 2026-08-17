// mid-group W4A8, TILE_M=16, GK=256, asymmetric weight. TP route: K_rank = K/TP pairs with GK = 1024/TP.
#define TILE_M_CFG 16
#define GK_CFG 256
#define ASYM_CFG 1
#define KERNEL_NAME midgroup_w4a8_gemm_m16_g256_asym
#include "midgroup_w4a8_gemm.inc"
