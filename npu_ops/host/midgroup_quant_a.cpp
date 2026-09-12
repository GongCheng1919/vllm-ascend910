// midgroup_quant_a: per-token mid-group INT8 activation quant + MSD int4 split.
//
// bf16 [M,K] -> the four tensors `midgroup_w4a8_gemm` reads on the activation
// side, in one pass, ALREADY PADDED to the GEMM's TILE_M (Mp >= M):
//   a_hi    int8  [Mp, K/2]  packed int4, high nibbles
//   a_lo    int8  [Mp, K/2]  packed int4, low nibbles biased by -8
//   a_scale bf16  [G, Mp]    group-major
//   a_ksum  int32 [G, Mp]    NEGATED sum_{k in g} A_q[m,k]
//
// Padding here rather than in the caller is what makes the decode path free of
// shape-dependent work.  The GEMM's own `Pad2D` then no-ops (Mp is already a
// multiple of the TILE_M it picks), and the caller needs no `if m != mp` branch
// -- a python branch there gets folded away by dynamo under graph capture and
// silently falls back to four `constant_pad_nd` launches per GEMM, which
// measured 33% of decode device time in the real engine (P4_E2E.md D6, D10).
//
// Semantics are `fakequant_lab/quant.py` verbatim (bf16-rounded scale, divide by
// it, round-half-to-even, clamp) -- verified byte-exact against a CPU model of
// quant.py.  Getting this "close enough" instead of exact would silently make
// P2's accuracy numbers not apply to the deployed kernel.

#include "utils.h"
#include "../kernel/mg_kgeom.h"
#include "aclrtlaunch_midgroup_quant_a_g1024.h"

namespace ascendc_path {

namespace {
constexpr int64_t  kGroup  = 1024;
constexpr int64_t  kJobRows = 16;   // kBr in the kernel
constexpr uint32_t kAivNum = 40;    // 910B4: 2 vector cores per AI core

// MUST stay identical to `TileMFor` in midgroup_w4a8_gemm.cpp.  If the two ever
// disagree the GEMM re-pads what we already padded -- silently correct, silently
// back to the D10 cost.  Mp is always a multiple of kJobRows, which the kernel
// relies on to store whole jobs.
int64_t TileMFor(int64_t M) { return (M <= 16) ? 16 : ((M <= 64) ? 64 : 128); }
}  // namespace

std::vector<at::Tensor> run_midgroup_quant_a(const at::Tensor &x)
{
    TORCH_CHECK(x.dim() == 2, "midgroup_quant_a: x must be [M, K]");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "midgroup_quant_a: x must be bfloat16");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    // K is ARBITRARY: the kernel emits the group-padded layout of mg_kgeom.h, so
    // the planes are Kpad/2 bytes per row and the final group may be short.
    const int64_t G = static_cast<int64_t>(
        mgk::NumGroups(static_cast<unsigned>(K), static_cast<unsigned>(kGroup)));
    const int64_t KbPad = static_cast<int64_t>(
        mgk::KPadElems(static_cast<unsigned>(K), static_cast<unsigned>(kGroup),
                       mgk::kCubeKElemsInt4)) / 2;

    const int64_t Mp = CeilTo(M, TileMFor(M));

    const at::Tensor xc = x.contiguous();
    at::Tensor a_hi   = at::empty({Mp, KbPad}, x.options().dtype(at::kChar));
    at::Tensor a_lo   = at::empty({Mp, KbPad}, x.options().dtype(at::kChar));
    at::Tensor a_scale = at::empty({G, Mp}, x.options().dtype(at::kBFloat16));
    at::Tensor a_ksum  = at::empty({G, Mp}, x.options().dtype(at::kInt));

    // AIV-only kernel: blockDim counts VECTOR cores (40), not AI cores (20).
    // Same parameter name as the GEMM's blockDim, different unit -- see P3 D7.
    // Jobs are counted over the PADDED rows: the padded rows still have to be
    // written, so they are real work (at M=1 that is 16 rows, not 1).
    const int64_t jobs = (Mp / kJobRows) * G;
    const uint32_t blockDim = static_cast<uint32_t>(
        std::max<int64_t>(1, std::min<int64_t>(kAivNum, jobs)));

    const uint32_t uM  = static_cast<uint32_t>(M);
    const uint32_t uMp = static_cast<uint32_t>(Mp);
    const uint32_t uK  = static_cast<uint32_t>(K);
    EXEC_KERNEL_CMD(midgroup_quant_a_g1024, blockDim,
                    xc, a_hi, a_lo, a_scale, a_ksum, uM, uMp, uK);

    return {a_hi, a_lo, a_scale, a_ksum};
}

}  // namespace ascendc_path

// The FRAGMENT must live in an ANONYMOUS namespace: it expands to a static
// initialiser object whose generated name restarts at 0 in every translation
// unit, so in a NAMED namespace two host files emit the same external symbol
// and the linker silently keeps one of them.
namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("midgroup_quant_a(Tensor x) -> Tensor[]");
}
}

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("midgroup_quant_a", TORCH_FN(ascendc_path::run_midgroup_quant_a));
}
}
