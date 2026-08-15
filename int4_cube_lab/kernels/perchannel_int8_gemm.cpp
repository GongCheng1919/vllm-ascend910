// INT8 x INT8 -> INT32 cube, per-token/per-channel dequant to bf16.
// Byte-for-byte identical pipeline to perchannel_int4_gemm — the controlled
// baseline for measuring the s4 cube speedup.
#define QBITS 8
#define KERNEL_NAME perchannel_int8_gemm
#include "perchannel_lowbit_gemm.inc"
