// W4A8 per-channel GEMM, TILE_M=128 — compute-bound shapes.
#define TILE_M_CFG 128
#define KERNEL_NAME perchannel_w4a8_gemm
#include "perchannel_w4a8_gemm.inc"
