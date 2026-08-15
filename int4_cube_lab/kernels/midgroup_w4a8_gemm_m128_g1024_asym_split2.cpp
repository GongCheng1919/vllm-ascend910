// mid-group W4A8 GEMM, TILE_M=128, GK=1024, asym, TWO-PASS MSD flush (D8).
// hi plane -> flush -> lo plane -> flush, so each Fixpipe hides under the other
// plane's Mmads and the next group's hi pass never waits on L0C.
// Numerically identical to midgroup_w4a8_gemm_m128_g1024_asym: must stay
// bit-exact under --check.
#define TILE_M_CFG 128
#define GK_CFG 1024
#define ASYM_CFG 1
#define SPLIT_MSD_FLUSH_CFG 2
#define KERNEL_NAME midgroup_w4a8_gemm_m128_g1024_asym_split2
#include "midgroup_w4a8_gemm.inc"
