// INT4 x INT4 -> INT32 cube, per-token/per-channel dequant to bf16.
#define QBITS 4
#define KERNEL_NAME perchannel_int4_gemm
#include "perchannel_lowbit_gemm.inc"
