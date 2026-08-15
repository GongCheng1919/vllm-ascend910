// midgroup_w4a8_gemm: mid-group W4A8 GEMM with an ASYMMETRIC int4 weight.
//
//   y[m,n] = sum_g as[g,m]*ws[g,n]*( 16*S_hi + S_lo + 8*w_ksum[g,n]
//                                    - wz[g,n]*a_ksum[g,m] )
//
// The activation is int8, split along the VALUE axis into two int4 planes (MSD);
// the weight is int4 with a per-(group, channel) zero point.  Both sides share
// the K-direction group boundary.  See MGCKPT/P3_KERNEL.md for the contract and
// the negative controls that verify it.
//
// Inputs (all group-major where grouped; G = K/GROUP):
//   a_hi    int8  [M, K/2]  packed int4, high nibbles
//   a_lo    int8  [M, K/2]  packed int4, low nibbles biased by -8
//   a_scale bf16  [G, M]
//   a_ksum  int32 [G, M]    NEGATED sum_{k in g} A_q[m,k]
//   w_q     int8  [N, K/2]  packed int4
//   w_scale bf16  [G, N]
//   w_ksum  int32 [G, N]    sum_{k in g} W_q[n,k]   (RAW codes, NOT zero-corrected)
//   w_zero  bf16  [G, N]    zero point in CODE units
// Output:
//   y       bf16  [M, N]
//
// `midgroup_quant_a` produces a_hi / a_lo / a_scale / a_ksum from a bf16
// activation in one pass, in exactly this layout.

#include "utils.h"
#include "aclrtlaunch_midgroup_w4a8_gemm_m16_g1024_asym.h"
#include "aclrtlaunch_midgroup_w4a8_gemm_m64_g1024_asym.h"
#include "aclrtlaunch_midgroup_w4a8_gemm_m128_g1024_asym.h"

namespace ascendc_path {

namespace {
constexpr int64_t kGroup   = 1024;   // the GK P0/P2 froze
constexpr int64_t kTileN   = 128;
constexpr uint32_t kAicNum = 20;     // 910B4 AI cores (this kernel is MIX)

// P3 §4.7: TILE_M is chosen by M the same way the final sweep did.  Keep this in
// sync with int4_cube_lab/scripts/sweep_p3_final.sh -- the published latency
// table is only meaningful if the dispatch matches.
int64_t TileMFor(int64_t M) { return (M <= 16) ? 16 : ((M <= 64) ? 64 : 128); }
}  // namespace

at::Tensor run_midgroup_w4a8_gemm(const at::Tensor &a_hi, const at::Tensor &a_lo,
                                  const at::Tensor &a_scale, const at::Tensor &a_ksum,
                                  const at::Tensor &w_q, const at::Tensor &w_scale,
                                  const at::Tensor &w_ksum, const at::Tensor &w_zero,
                                  int64_t K)
{
    TORCH_CHECK(a_hi.dim() == 2 && a_lo.dim() == 2 && w_q.dim() == 2,
                "midgroup_w4a8_gemm: a_hi/a_lo [M,K/2] and w_q [N,K/2] must be 2-D");
    TORCH_CHECK(a_hi.scalar_type() == at::kChar && a_lo.scalar_type() == at::kChar &&
                w_q.scalar_type() == at::kChar,
                "midgroup_w4a8_gemm: packed int4 planes must be int8 tensors");
    TORCH_CHECK(a_scale.scalar_type() == at::kBFloat16 &&
                w_scale.scalar_type() == at::kBFloat16 &&
                w_zero.scalar_type() == at::kBFloat16,
                "midgroup_w4a8_gemm: a_scale/w_scale/w_zero must be bfloat16");
    TORCH_CHECK(a_ksum.scalar_type() == at::kInt && w_ksum.scalar_type() == at::kInt,
                "midgroup_w4a8_gemm: a_ksum/w_ksum must be int32 "
                "(a_ksum in bf16 would round a ~1.3e5 row sum to ~0.2%; see P3 D6)");

    const int64_t M = a_hi.size(0);
    const int64_t N = w_q.size(0);
    TORCH_CHECK(a_hi.size(1) == K / 2 && w_q.size(1) == K / 2,
                "midgroup_w4a8_gemm: packed row length must be K/2");
    TORCH_CHECK(K % kGroup == 0, "midgroup_w4a8_gemm: K must be a multiple of ", kGroup,
                "; got ", K);
    TORCH_CHECK(N % kTileN == 0, "midgroup_w4a8_gemm: N must be a multiple of ", kTileN,
                "; got ", N);
    const int64_t G = K / kGroup;
    TORCH_CHECK(a_scale.size(0) == G && a_scale.size(1) == M,
                "midgroup_w4a8_gemm: a_scale must be group-major [K/group, M]");
    TORCH_CHECK(w_scale.size(0) == G && w_scale.size(1) == N,
                "midgroup_w4a8_gemm: w_scale must be group-major [K/group, N]");

    const int64_t tileM = TileMFor(M);
    const int64_t Mp    = CeilTo(M, tileM);

    // Pad M only.  Padded activation rows carry scale 0 and contribute nothing;
    // their output rows are sliced off below, so the logical result is unchanged.
    //
    // On the vLLM path these four are NO-OPS: `midgroup_quant_a` already emits
    // Mp rows, and its TileMFor is the same function as this one.  They are kept
    // for direct callers that hand over a ragged activation (the benches do).
    // If you ever see PadV3/MemSet in an engine profile, the two TileMFor
    // definitions have drifted -- see P4_E2E.md D10.
    const at::Tensor aHi = Pad2D(a_hi.contiguous(), Mp, K / 2);
    const at::Tensor aLo = Pad2D(a_lo.contiguous(), Mp, K / 2);
    const at::Tensor aS  = Pad2D(a_scale.contiguous(), G, Mp);
    const at::Tensor aK  = Pad2D(a_ksum.contiguous(), G, Mp);
    const at::Tensor wQ  = w_q.contiguous();
    const at::Tensor wS  = w_scale.contiguous();
    const at::Tensor wK  = w_ksum.contiguous();
    const at::Tensor wZ  = w_zero.contiguous();

    at::Tensor y = at::empty({Mp, N}, a_scale.options().dtype(at::kBFloat16));

    const int64_t numTiles = (Mp / tileM) * (N / kTileN);
    const uint32_t blockDim = static_cast<uint32_t>(
        std::max<int64_t>(1, std::min<int64_t>(kAicNum, numTiles)));

    // Two flush slots per block, each holding one stacked-A cube tile (2*tileM x 128)
    // of int32 -- the AIC double-buffers the Fixpipe target so it can run a group
    // ahead of the AIV.
    at::Tensor workspace = at::empty(
        {static_cast<int64_t>(blockDim) * 2 * 2 * tileM * kTileN},
        a_scale.options().dtype(at::kInt));

    const uint32_t uM = static_cast<uint32_t>(Mp);
    const uint32_t uN = static_cast<uint32_t>(N);
    const uint32_t uK = static_cast<uint32_t>(K);

    switch (tileM) {
        case 16:
            EXEC_KERNEL_CMD(midgroup_w4a8_gemm_m16_g1024_asym, blockDim,
                            aHi, aLo, aS, wQ, wS, wK, wZ, aK, y, workspace, uM, uN, uK);
            break;
        case 64:
            EXEC_KERNEL_CMD(midgroup_w4a8_gemm_m64_g1024_asym, blockDim,
                            aHi, aLo, aS, wQ, wS, wK, wZ, aK, y, workspace, uM, uN, uK);
            break;
        default:
            EXEC_KERNEL_CMD(midgroup_w4a8_gemm_m128_g1024_asym, blockDim,
                            aHi, aLo, aS, wQ, wS, wK, wZ, aK, y, workspace, uM, uN, uK);
            break;
    }

    return Slice2D(y, M, N);
}

}  // namespace ascendc_path

// The FRAGMENT must live in an ANONYMOUS namespace: it expands to a static
// initialiser object whose generated name restarts at 0 in every translation
// unit, so in a NAMED namespace two host files emit the same external symbol
// and the linker silently keeps one of them.
namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("midgroup_w4a8_gemm(Tensor a_hi, Tensor a_lo, Tensor a_scale, Tensor a_ksum, "
          "Tensor w_q, Tensor w_scale, Tensor w_ksum, Tensor w_zero, int K) -> Tensor");
}
}

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("midgroup_w4a8_gemm", TORCH_FN(ascendc_path::run_midgroup_w4a8_gemm));
}
}
