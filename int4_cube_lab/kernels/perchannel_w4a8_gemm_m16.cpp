// W4A8 per-channel GEMM, TILE_M=16 — short-tile build for decode-shaped M.
#define TILE_M_CFG 16
#define KERNEL_NAME perchannel_w4a8_gemm_m16
#include "perchannel_w4a8_gemm.inc"
