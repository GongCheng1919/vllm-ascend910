// W4A8 per-channel GEMM, TILE_M=32 — short-tile build for decode-shaped M.
#define TILE_M_CFG 32
#define KERNEL_NAME perchannel_w4a8_gemm_m32
#include "perchannel_w4a8_gemm.inc"
