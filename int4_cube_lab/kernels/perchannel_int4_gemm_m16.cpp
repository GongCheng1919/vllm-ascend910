// INT4 per-channel GEMM, TILE_M=16 — short-tile build for decode-shaped M.
#define QBITS 4
#define TILE_M_CFG 16
#define KERNEL_NAME perchannel_int4_gemm_m16
#include "perchannel_lowbit_gemm.inc"
