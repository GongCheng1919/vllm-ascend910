// P5 ABLATION BUILD -- AIV is handshake-only, output is GARBAGE.
// Timing-only twin of midgroup_w4a8_gemm_m16_g1024_asym: same AIC, same flush
// count, zero-cost consumer.  See MGCKPT/P5_KERNEL_OPT.md §4 step 1.
// Do NOT run with --check; it will (correctly) fail.
#define TILE_M_CFG 16
#define GK_CFG 1024
#define ASYM_CFG 1
#define ABLATE_AIV_CFG 1
#define KERNEL_NAME midgroup_w4a8_gemm_m16_g1024_asym_noaiv
#include "midgroup_w4a8_gemm.inc"
