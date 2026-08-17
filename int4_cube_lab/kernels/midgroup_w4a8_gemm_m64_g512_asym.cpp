// mid-group W4A8, TILE_M=64, GK=512, asymmetric weight. TP route: K_rank = K/TP pairs with GK = 1024/TP.
#define TILE_M_CFG 64
#define GK_CFG 512
#define ASYM_CFG 1
#define KERNEL_NAME midgroup_w4a8_gemm_m64_g512_asym
#include "midgroup_w4a8_gemm.inc"
