// midgroup_w4a8_gemm.inc -- mid-group W4A8 per-channel GEMM.
//
// The including .cpp must define, before the include:
//   TILE_M_CFG   output rows per tile (16 / 32 / 64 / 128)
//   GK_CFG       quantisation group size along K, in ELEMENTS (256 / 512 / 1024)
//   KERNEL_NAME  the extern "C" __global__ symbol to emit
//   ASYM_CFG     1 = asymmetric weight (per-group zero point).  Default 0.
//
// Contract (G = K / GK groups; scales are GROUP-MAJOR):
//   a_hi:    [M, K/2] int4 packed   -- high nibbles of the int8 activation
//   a_lo:    [M, K/2] int4 packed   -- low nibbles, biased by -8
//   a_scale: [G, M] bf16            -- per (group, token)
//   w_q:     [N, K/2] int4 packed, row-major; computes A @ W.T
//   w_scale: [G, N] bf16            -- per (group, channel)
//   w_ksum:  [G, N] int32           -- sum_{k in g} W[n,k], the MSD bias correction
//   w_zero:  [G, N] bf16            -- weight zero point, CODE units (asym only)
//   a_ksum:  [G, M] int32           -- NEGATED sum_{k in g} A_q[m,k] (asym only)
//   y:       [M, N] bf16
//   workspace: [blockDim, 2*TILE_M, 128] int32
//
// The two asym tensors are in the signature unconditionally so sym and asym
// share one host path; a sym build simply never reads them (pass anything).
//
// Relative to perchannel_w4a8_gemm.inc the ONLY structural change is that the
// L0C accumulator no longer spans the whole K: it is flushed every GK elements,
// and the AIV rescales each group's int32 partial sum and accumulates in fp32.
// The MSD split is along the VALUE axis and the grouping along K, so the two are
// orthogonal -- the stacked-A layout (h and l planes stacked into one 2*TILE_M
// cube tile, one Mmad, one Fixpipe) is untouched.  Only w_ksum grows from [N]
// to [G, N], because the +8 of the split is now corrected per group.
//
//   y[m,n] = sum_g as[g,m] * ws[g,n] * (16*S_hi[g] + S_lo[g] + 8*w_ksum[g,n]
//                                       - wz[g,n] * a_ksum[g,m])
//
// The zero-point term is rank-1 within a group, so it does NOT need a
// materialised tile: one MulAddDst whose two sources use different broadcast
// modes (src0 a Brcb block per row, src1 a 64-float row re-read) folds it into
// the existing chain for one extra vector instruction per group per chunk-half.
// `w_ksum` keeps using the RAW codes and is NOT zero-corrected -- it comes from
// the MSD +8 bias and is unrelated to asymmetry.  Mixing the two silently costs
// SNR instead of failing.
//
// a_ksum is int32, NOT bf16, on purpose: at GK=1024 a row sum reaches ~1.3e5,
// and bf16's 8 mantissa bits would round it to ~0.2%.  The zero-point term is
// the same order of magnitude as the main product, so that rounding would land
// straight on the output -- and fakequant_lab computes the same sum exactly in
// fp32, so the kernel would no longer match the algorithm it is supposed to
// implement.  w_zero stays bf16: it is an integer in [-8,7] (see quant.py's
// `zero = qmin - round(mn/scale)`), which bf16 represents exactly.
//
// Group-major scale layout and "AIC waits for the AIV credit only immediately
// before Fixpipe" both follow NPU-OP/mid_group_gemm_fwd_lab/kernels/
// mid_group_gemm_fwd.cpp (O2 / O4 there) -- they are load-bearing for perf.

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_type.h"
#include "basic_api/kernel_operator_fixpipe_intf.h"

using namespace AscendC;

namespace {

#ifndef TILE_M_CFG
#define TILE_M_CFG 128
#endif
#ifndef GK_CFG
#define GK_CFG 1024
#endif
#ifndef ASYM_CFG
#define ASYM_CFG 0
#endif

constexpr bool     kAsym   = (ASYM_CFG != 0);
constexpr uint32_t TILE_M  = TILE_M_CFG;
constexpr uint32_t TILE_N  = 128;
constexpr uint32_t TILE_MC = 2 * TILE_M;         // cube rows: h-plane then l-plane
constexpr uint32_t HALF_M  = TILE_M / 2;         // output rows per AIV
constexpr uint32_t GK      = GK_CFG;             // quant group, in K elements

constexpr uint32_t CHUNK_M        = (HALF_M < 32) ? HALF_M : 32;
constexpr uint32_t CHUNK_ELEMS    = CHUNK_M * TILE_N;
constexpr uint32_t CHUNKS_PER_AIV = HALF_M / CHUNK_M;

static_assert(TILE_M >= 16 && (TILE_M % 16) == 0, "TILE_M must be a multiple of 16");
static_assert((HALF_M % 8) == 0, "TILE_M/2 must be a multiple of 8 (Brcb repeat)");
static_assert(CHUNKS_PER_AIV <= 2, "AIV pipeline is written for at most 2 chunks");
static_assert(GK == 256 || GK == 512 || GK == 1024, "GK must be 256 / 512 / 1024");

// ---- K geometry, in BYTES (int4: 2 elements per byte) ----
// The L0 slice must divide the quant group, and A holds 2*TILE_M rows so the
// slice depth also has to keep the depth-2 L0A ping-pong inside 64 KB.
constexpr uint32_t kSliceCap = (TILE_M <= 64) ? 256 : 128;
constexpr uint32_t INNER_KB  = (kSliceCap < GK / 2) ? kSliceCap : GK / 2;
constexpr uint32_t INNER_K   = INNER_KB * 2;
constexpr uint32_t SLICES_PER_QGROUP = GK / INNER_K;

// The L1 slab holds a whole number of quant groups, sized back up to the same
// GM->L1 granularity the per-channel kernel used (so DMA efficiency is
// unchanged and only the flush rate varies with GK).
constexpr uint32_t kTargetGroupKB  = (TILE_M <= 64) ? 1024 : 512;
constexpr uint32_t QGROUP_KB       = SLICES_PER_QGROUP * INNER_KB;
constexpr uint32_t QGROUPS_PER_L1  = (kTargetGroupKB / QGROUP_KB) ? (kTargetGroupKB / QGROUP_KB) : 1;
constexpr uint32_t SLICES_PER_GROUP = QGROUPS_PER_L1 * SLICES_PER_QGROUP;
constexpr uint32_t GROUP_KB        = SLICES_PER_GROUP * INNER_KB;

static_assert(GK % INNER_K == 0, "GK must be a multiple of the L0 slice");

constexpr uint32_t kBlock        = 16;
constexpr uint32_t kK0B          = 32;                   // C0 in bytes
constexpr uint32_t kMBlocks      = TILE_MC / kBlock;
constexpr uint32_t kNBlocks      = TILE_N / kBlock;
constexpr uint32_t kKBlocksInner = INNER_KB / kK0B;
constexpr uint32_t kNzBlockSize  = kK0B * kBlock;        // 512 B per fractal
constexpr uint32_t kSliceStrideA = INNER_KB * TILE_MC;
constexpr uint32_t kSliceStrideB = INNER_KB * TILE_N;

// On-chip budgets (910B4): L1 512 KB, L0A/L0B 64 KB each, L0C 128 KB, UB 192 KB.
// L0C gets depth 2 whenever it fits, so the next group's Mmad overlaps this
// group's Fixpipe -- that overlap is the whole point at small GK.
constexpr uint32_t kL0CTileBytes = TILE_MC * TILE_N * sizeof(int32_t);
constexpr uint32_t kL0CDepth     = (2 * kL0CTileBytes <= 128 * 1024) ? 2 : 1;

static_assert(2 * TILE_MC * INNER_KB <= 64 * 1024, "L0A over budget");
static_assert(2 * TILE_N  * INNER_KB <= 64 * 1024, "L0B over budget");
static_assert(kL0CDepth * kL0CTileBytes <= 128 * 1024, "L0C over budget");
static_assert(2 * (TILE_MC + TILE_N) * GROUP_KB <= 512 * 1024, "L1 over budget");

// Two flag pairs and a double-buffered workspace slot.  With a single pair the
// AIC and AIV lock-step once per quant group -- cube(g+1) can never overlap the
// AIV's drain of group g -- and that serialisation, not the data volume, is
// what a small GK actually costs (measured ~0.5 us per flush at TILE_M=16,
// where the AIV only touches 1024 elements).  Alternating slots lets the AIC
// run one group ahead.
constexpr uint16_t kAivToAicFlag0 = 3;
constexpr uint16_t kAivToAicFlag1 = 4;
constexpr uint16_t kAicToAivFlag0 = 5;
constexpr uint16_t kAicToAivFlag1 = 6;
constexpr uint8_t  kCrossMode     = 2;

// ---- AIV scale staging: BATCH_G quant groups at a time ----
// The scale preamble (3 GM reads + a full MTE2 drain + 5 vector ops) used to sit
// on the critical path of *every* quant group, where it was the dominant part of
// the mid-group cost: the three loads are tiny (256 B / 32 B / 512 B) but the
// back-to-back SetFlag/WaitFlag<MTE2_V> exposed raw GM latency once per group,
// while the AIC's work per group only scales with GK.  Group-major scales make
// each load one strided DataCopy with blockCount = batch size, so the whole
// preamble is hoisted out and paid once per BATCH_G groups instead.
//
// BATCH_G is whatever the 192 KB UB has left after the hot buffers; 32 is plenty
// (K/GK is 5..27 for QwQ-32B at GK=1024) and capping it keeps `down` at GK=256
// (numQGroups = 108) from asking for more UB than exists.
constexpr uint32_t kUbTotalBytes = 192 * 1024;
constexpr uint32_t kUbReserve    = 8 * 1024;          // TPipe bookkeeping slack
constexpr uint32_t kUbHotBytes   = HALF_M * TILE_N *
    (2 * sizeof(int32_t) + 2 * sizeof(float) + sizeof(bfloat16_t));
// The asym term costs ~60% more staging per group, which is the real risk: it
// eats BATCH_G, and batching the preamble is worth 17-22% (P0 COST §2.5) --
// an order of magnitude more than the one extra vector instruction.
constexpr uint32_t kAsymBytesPerG = !kAsym ? 0u : (
      TILE_N * sizeof(bfloat16_t)     // w_zero bf16 staging
    + TILE_M * sizeof(int32_t)        // a_ksum int32 staging
    + TILE_N * sizeof(float)          // w_zero fp32
    + TILE_M * sizeof(float)          // a_ksum fp32
    + HALF_M * 8 * sizeof(float));    // a_ksum Brcb blocks
constexpr uint32_t kScaleBytesPerG =
      TILE_N * sizeof(bfloat16_t)     // w_scale bf16 staging
    + TILE_M * sizeof(bfloat16_t)     // a_scale bf16 staging
    + TILE_N * sizeof(int32_t)        // w_ksum  int32 staging
    + TILE_N * sizeof(float)          // w_scale fp32
    + TILE_N * sizeof(float)          // 8 * w_ksum fp32
    + TILE_M * sizeof(float)          // a_scale fp32
    + HALF_M * 8 * sizeof(float)      // a_scale Brcb blocks
    + kAsymBytesPerG;
constexpr uint32_t kBatchGFit = (kUbTotalBytes - kUbHotBytes - kUbReserve) / kScaleBytesPerG;
constexpr uint32_t BATCH_G    = (kBatchGFit < 32) ? kBatchGFit : 32;

static_assert(BATCH_G >= 1, "UB cannot stage even one quant group's scales");
static_assert(kUbHotBytes + BATCH_G * kScaleBytesPerG + kUbReserve <= kUbTotalBytes,
              "AIV UB over budget");

__aicore__ inline void LoadSliceL1ToL0(const LocalTensor<int8_t>& aL0,
                                       const LocalTensor<int8_t>& bL0,
                                       const LocalTensor<int8_t>& aL1,
                                       const LocalTensor<int8_t>& bL1,
                                       const LoadData2dParams& ldA,
                                       const LoadData2dParams& ldB)
{
    for (uint32_t mi = 0; mi < kMBlocks; mi++) {
        LoadData(aL0[mi * kKBlocksInner * kNzBlockSize].ReinterpretCast<int4b_t>(),
                 aL1[mi * kNzBlockSize].ReinterpretCast<int4b_t>(), ldA);
    }
    LoadData(bL0.ReinterpretCast<int4b_t>(), bL1.ReinterpretCast<int4b_t>(), ldB);
}

template <typename A1Q, typename B1Q>
__aicore__ inline void IssueGroupLoad(A1Q& a1Queue, B1Q& b1Queue,
                                      const GlobalTensor<int8_t>& aHiGm,
                                      const GlobalTensor<int8_t>& aLoGm,
                                      const GlobalTensor<int8_t>& bGm,
                                      Nd2NzParams& ndA, Nd2NzParams& ndB,
                                      uint64_t aBase, uint64_t bBase,
                                      uint64_t gOffB, uint32_t slices)
{
    const uint16_t dv = static_cast<uint16_t>(slices * INNER_KB);
    ndA.dValue = dv;
    ndB.dValue = dv;
    auto aL1 = a1Queue.template AllocTensor<int8_t>();
    auto bL1 = b1Queue.template AllocTensor<int8_t>();
    DataCopy(aL1,                aHiGm[aBase + gOffB], ndA);
    DataCopy(aL1[TILE_M * kK0B], aLoGm[aBase + gOffB], ndA);
    DataCopy(bL1,                bGm[bBase + gOffB],   ndB);
    a1Queue.EnQue(aL1);
    b1Queue.EnQue(bL1);
}

// acc += ((16*C_hi + C_lo) + 8*w_ksum[g,n] - wz[g,n]*a_ksum[g,m])
//        * w_scale[g,n] * a_scale[g,m]
// for one CHUNK_M-row chunk of one quant group.  The int32 combine happens
// before the fp32 cast so the two partial products add exactly.
// `wsF32` / `ksF32` / `rowScaleBrcb` (and, for asym, `wzF32` / `akBrcb`) are
// already rebased onto this quant group's slice of the batched staging buffers.
__aicore__ inline void AccumulateGroupChunk(const LocalTensor<int32_t>& hi,
                                            const LocalTensor<int32_t>& lo,
                                            const LocalTensor<float>& temp,
                                            const LocalTensor<float>& acc,
                                            const LocalTensor<float>& wsF32,
                                            const LocalTensor<float>& ksF32,
                                            const LocalTensor<float>& rowScaleBrcb,
                                            const LocalTensor<float>& wzF32,
                                            const LocalTensor<float>& akBrcb,
                                            uint32_t chunk)
{
    const uint32_t off         = chunk * CHUNK_ELEMS;
    const uint32_t rowScaleOff = chunk * CHUNK_M * 8;

    Muls(hi[off], hi[off], 16, CHUNK_ELEMS);
    Add(hi[off], hi[off], lo[off], CHUNK_ELEMS);
    Cast(temp[off], hi[off], RoundMode::CAST_NONE, CHUNK_ELEMS);

    // src1 is a 64-float row re-read on every repeat (src1RepStride = 0).
    BinaryRepeatParams rowParam(1, 1, 1, 16, 16, 0);
    // src1 is one 8-float Brcb block per repeat (src1BlkStride 0, RepStride 1).
    BinaryRepeatParams colParam(1, 1, 0, 16, 16, 1);
    // The zero-point term is an outer product: src0 broadcasts down the rows
    // (Brcb block per repeat), src1 across them (row re-read).  MulAddDst takes
    // independent strides for the two sources, so no tile has to be
    // materialised -- a_ksum is stored NEGATED so this stays an add.
    BinaryRepeatParams zpParam(1, 0, 1, 16, 1, 0);

    for (uint32_t half = 0; half < 2; half++) {
        const uint32_t o = off + half * 64;
        const uint32_t s = half * 64;
        Add(temp[o], temp[o], ksF32[s], 64, CHUNK_M, rowParam);
        if constexpr (kAsym) {
            MulAddDst(temp[o], akBrcb[rowScaleOff], wzF32[s], 64, CHUNK_M, zpParam);
        }
        Mul(temp[o], temp[o], wsF32[s], 64, CHUNK_M, rowParam);
        MulAddDst(acc[o], temp[o], rowScaleBrcb[rowScaleOff], 64, CHUNK_M, colParam);
    }
}

}  // namespace

extern "C" __global__ __aicore__
void KERNEL_NAME(GM_ADDR a_hi, GM_ADDR a_lo, GM_ADDR a_scale,
                 GM_ADDR w_q,  GM_ADDR w_scale, GM_ADDR w_ksum,
                 GM_ADDR w_zero, GM_ADDR a_ksum,
                 GM_ADDR y,    GM_ADDR workspace,
                 uint32_t M, uint32_t N, uint32_t K)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    const uint32_t blockIdx  = GetBlockIdx() / GetTaskRation();
    const uint32_t blockDim  = GetBlockNum();
    const uint32_t numNTiles = N / TILE_N;
    const uint32_t numTiles  = (M / TILE_M) * numNTiles;

    const uint32_t Kb          = K / 2;                     // bytes per row (int4)
    const uint32_t numSlices   = Kb / INNER_KB;
    const uint32_t numQGroups  = K / GK;
    const uint32_t numG        = (numSlices + SLICES_PER_GROUP - 1) / SLICES_PER_GROUP;
    const uint32_t lastGSlices = numSlices - (numG - 1) * SLICES_PER_GROUP;
    const uint32_t g0Slices    = (numG == 1) ? lastGSlices : SLICES_PER_GROUP;

    // Two slots per block: slot = (global flush index) & 1.
    constexpr uint32_t kWsElems = TILE_MC * TILE_N;
    GlobalTensor<int32_t> wsGm;
    wsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(workspace) + (uint64_t)blockIdx * 2 * kWsElems,
        2 * kWsElems);

    // ============================ AIC ============================
    if ASCEND_IS_AIC {
        GlobalTensor<int8_t> aHiGm, aLoGm, bGm;
        aHiGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a_hi), (uint64_t)M * Kb);
        aLoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a_lo), (uint64_t)M * Kb);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w_q),   (uint64_t)N * Kb);

        TQue<QuePosition::A1, 2> a1Queue;
        TQue<QuePosition::B1, 2> b1Queue;
        TQue<QuePosition::A2, 2> a2Queue;
        TQue<QuePosition::B2, 2> b2Queue;
        TQue<QuePosition::CO1, kL0CDepth> c1Queue;
        pipe.InitBuffer(a1Queue, 2, TILE_MC * GROUP_KB);
        pipe.InitBuffer(b1Queue, 2, TILE_N  * GROUP_KB);
        pipe.InitBuffer(a2Queue, 2, TILE_MC * INNER_KB);
        pipe.InitBuffer(b2Queue, 2, TILE_N  * INNER_KB);
        pipe.InitBuffer(c1Queue, kL0CDepth, kL0CTileBytes);

        Nd2NzParams ndA;
        ndA.ndNum             = 1;
        ndA.nValue            = TILE_M;
        ndA.dValue            = GROUP_KB;                 // overwritten per group
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue         = static_cast<uint16_t>(Kb);
        ndA.dstNzC0Stride     = TILE_MC;
        ndA.dstNzNStride      = 1;
        ndA.dstNzMatrixStride = 0;

        Nd2NzParams ndB = ndA;
        ndB.nValue        = TILE_N;
        ndB.dstNzC0Stride = TILE_N;

        LoadData2dParams ldA;
        ldA.startIndex  = 0;
        ldA.repeatTimes = static_cast<uint8_t>(kKBlocksInner);
        ldA.srcStride   = static_cast<uint16_t>(kMBlocks);
        ldA.dstGap      = 0;
        ldA.ifTranspose = false;
        ldA.sid         = 0;
        ldA.addrMode    = 0;

        LoadData2dParams ldB;
        ldB.startIndex  = 0;
        ldB.repeatTimes = static_cast<uint8_t>(kNBlocks * kKBlocksInner);
        ldB.srcStride   = 1;
        ldB.dstGap      = 0;
        ldB.ifTranspose = false;
        ldB.sid         = 0;
        ldB.addrMode    = 0;

        FixpipeParamsV220 fp;
        fp.nSize       = TILE_N;
        fp.mSize       = TILE_MC;
        fp.srcStride   = TILE_MC;
        fp.dstStride   = TILE_N;
        fp.quantPre    = QuantMode_t::NoQuant;
        fp.ndNum       = 1;
        fp.srcNdStride = 1;
        fp.dstNdStride = 1;

        MmadParams mm;
        mm.m             = TILE_MC;
        mm.n             = TILE_N;
        mm.k             = INNER_K;
        mm.cmatrixSource = false;

        uint32_t flushIdx = 0;          // global flush counter; AIV mirrors it
        bool tileG0Prefetched = false;
        for (uint32_t tileId = blockIdx; tileId < numTiles; tileId += blockDim) {
            const uint32_t mTile = tileId / numNTiles;
            const uint32_t nTile = tileId % numNTiles;
            const uint64_t aTileBase = (uint64_t)mTile * TILE_M * Kb;
            const uint64_t bTileBase = (uint64_t)nTile * TILE_N * Kb;

            if (!tileG0Prefetched) {
                IssueGroupLoad(a1Queue, b1Queue, aHiGm, aLoGm, bGm, ndA, ndB,
                               aTileBase, bTileBase, 0, g0Slices);
            }
            tileG0Prefetched = false;

            LocalTensor<int32_t> cL0;
            uint32_t sInQ = 0;              // slices accumulated into the live L0C

            for (uint32_t g = 0; g < numG; g++) {
                const uint32_t slices = (g + 1 == numG) ? lastGSlices : SLICES_PER_GROUP;

                if (g + 1 < numG) {
                    const uint32_t nextSlices =
                        (g + 2 == numG) ? lastGSlices : SLICES_PER_GROUP;
                    IssueGroupLoad(a1Queue, b1Queue, aHiGm, aLoGm, bGm, ndA, ndB,
                                   aTileBase, bTileBase,
                                   (uint64_t)(g + 1) * GROUP_KB, nextSlices);
                }

                auto aL1 = a1Queue.DeQue<int8_t>();
                auto bL1 = b1Queue.DeQue<int8_t>();

                {
                    auto aL0p = a2Queue.AllocTensor<int8_t>();
                    auto bL0p = b2Queue.AllocTensor<int8_t>();
                    LoadSliceL1ToL0(aL0p, bL0p, aL1, bL1, ldA, ldB);
                    a2Queue.EnQue(aL0p);
                    b2Queue.EnQue(bL0p);
                }

                for (uint32_t s = 0; s < slices; s++) {
                    if (s + 1 < slices) {
                        auto aL0n = a2Queue.AllocTensor<int8_t>();
                        auto bL0n = b2Queue.AllocTensor<int8_t>();
                        LoadSliceL1ToL0(aL0n, bL0n,
                                        aL1[(s + 1) * kSliceStrideA],
                                        bL1[(s + 1) * kSliceStrideB], ldA, ldB);
                        a2Queue.EnQue(aL0n);
                        b2Queue.EnQue(bL0n);
                    }

                    auto aL0 = a2Queue.DeQue<int8_t>();
                    auto bL0 = b2Queue.DeQue<int8_t>();

                    if (s + 1 == slices) {
                        a1Queue.FreeTensor(aL1);
                        b1Queue.FreeTensor(bL1);
                    }

                    if (sInQ == 0) {
                        cL0 = c1Queue.AllocTensor<int32_t>();
                    }
                    mm.cmatrixInitVal = (sInQ == 0);
                    Mmad(cL0, aL0.ReinterpretCast<int4b_t>(),
                         bL0.ReinterpretCast<int4b_t>(), mm);

                    a2Queue.FreeTensor(aL0);
                    b2Queue.FreeTensor(bL0);

                    if (++sInQ == SLICES_PER_QGROUP) {
                        sInQ = 0;
                        c1Queue.EnQue(cL0);
                        auto cOut = c1Queue.DeQue<int32_t>();
                        // Wait for the AIV credit only here, and only for the
                        // slot about to be reused (two flushes ago): the loads
                        // and Mmads above already ran ahead of it.
                        const uint32_t slot = flushIdx & 1u;
                        if (flushIdx >= 2) {
                            if (slot == 0) CrossCoreWaitFlag(kAivToAicFlag0);
                            else           CrossCoreWaitFlag(kAivToAicFlag1);
                        }
                        Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                            wsGm[slot * kWsElems], cOut, fp);
                        c1Queue.FreeTensor(cOut);
                        if (slot == 0) CrossCoreSetFlag<kCrossMode, PIPE_FIX>(kAicToAivFlag0);
                        else           CrossCoreSetFlag<kCrossMode, PIPE_FIX>(kAicToAivFlag1);
                        flushIdx++;
                    }
                }
            }

            const uint32_t nextTileId = tileId + blockDim;
            if (nextTileId < numTiles) {
                const uint32_t nextMTile = nextTileId / numNTiles;
                const uint32_t nextNTile = nextTileId % numNTiles;
                IssueGroupLoad(a1Queue, b1Queue, aHiGm, aLoGm, bGm, ndA, ndB,
                               (uint64_t)nextMTile * TILE_M * Kb,
                               (uint64_t)nextNTile * TILE_N * Kb, 0, g0Slices);
                tileG0Prefetched = true;
            }
        }

        // Drain the outstanding AIV credits (at most one per slot).
        const uint32_t outstanding = (flushIdx < 2) ? flushIdx : 2;
        for (uint32_t i = 0; i < outstanding; i++) {
            if (((flushIdx - outstanding + i) & 1u) == 0) CrossCoreWaitFlag(kAivToAicFlag0);
            else                                          CrossCoreWaitFlag(kAivToAicFlag1);
        }
    }

    // ============================ AIV ============================
    if ASCEND_IS_AIV {
        const uint32_t subIdx = static_cast<uint32_t>(GetSubBlockIdx());
        const uint32_t mBase  = subIdx * HALF_M;
        constexpr uint32_t myM     = HALF_M;
        constexpr uint32_t myElems = HALF_M * TILE_N;

        GlobalTensor<bfloat16_t> aScaleGm, wScaleGm, outGm, wZeroGm;
        GlobalTensor<int32_t> ksumGm, aKsumGm;
        aScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a_scale),
                                 (uint64_t)numQGroups * M);
        wScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w_scale),
                                 (uint64_t)numQGroups * N);
        ksumGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(w_ksum),
                               (uint64_t)numQGroups * N);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y), (uint64_t)M * N);
        if constexpr (kAsym) {
            wZeroGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w_zero),
                                    (uint64_t)numQGroups * N);
            aKsumGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a_ksum),
                                    (uint64_t)numQGroups * M);
        }

        TBuf<QuePosition::VECCALC> ubHiBuf, ubLoBuf, ubTempBuf, ubAccBuf, ubOutBuf;
        TBuf<QuePosition::VECCALC> ubWsBf16Buf, ubAsBf16Buf, ubKsI32Buf;
        TBuf<QuePosition::VECCALC> ubWsF32Buf, ubKsF32Buf, ubAsF32Buf, ubRowScaleBuf;
        TBuf<QuePosition::VECCALC> ubWzBf16Buf, ubAkI32Buf, ubWzF32Buf, ubAkF32Buf, ubAkBrcbBuf;

        pipe.InitBuffer(ubHiBuf,       myElems * sizeof(int32_t));
        pipe.InitBuffer(ubLoBuf,       myElems * sizeof(int32_t));
        pipe.InitBuffer(ubTempBuf,     myElems * sizeof(float));
        pipe.InitBuffer(ubAccBuf,      myElems * sizeof(float));
        pipe.InitBuffer(ubOutBuf,      myElems * sizeof(bfloat16_t));
        pipe.InitBuffer(ubWsBf16Buf,   BATCH_G * TILE_N * sizeof(bfloat16_t));
        pipe.InitBuffer(ubAsBf16Buf,   BATCH_G * TILE_M * sizeof(bfloat16_t));
        pipe.InitBuffer(ubKsI32Buf,    BATCH_G * TILE_N * sizeof(int32_t));
        pipe.InitBuffer(ubWsF32Buf,    BATCH_G * TILE_N * sizeof(float));
        pipe.InitBuffer(ubKsF32Buf,    BATCH_G * TILE_N * sizeof(float));
        pipe.InitBuffer(ubAsF32Buf,    BATCH_G * TILE_M * sizeof(float));
        pipe.InitBuffer(ubRowScaleBuf, BATCH_G * myM * 8 * sizeof(float));
        if constexpr (kAsym) {
            pipe.InitBuffer(ubWzBf16Buf, BATCH_G * TILE_N * sizeof(bfloat16_t));
            pipe.InitBuffer(ubAkI32Buf,  BATCH_G * TILE_M * sizeof(int32_t));
            pipe.InitBuffer(ubWzF32Buf,  BATCH_G * TILE_N * sizeof(float));
            pipe.InitBuffer(ubAkF32Buf,  BATCH_G * TILE_M * sizeof(float));
            pipe.InitBuffer(ubAkBrcbBuf, BATCH_G * myM * 8 * sizeof(float));
        }

        auto ubHi       = ubHiBuf.Get<int32_t>();
        auto ubLo       = ubLoBuf.Get<int32_t>();
        auto ubTemp     = ubTempBuf.Get<float>();
        auto ubAcc      = ubAccBuf.Get<float>();
        auto ubOut      = ubOutBuf.Get<bfloat16_t>();
        auto ubWsBf16   = ubWsBf16Buf.Get<bfloat16_t>();
        auto ubAsBf16   = ubAsBf16Buf.Get<bfloat16_t>();
        auto ubKsI32    = ubKsI32Buf.Get<int32_t>();
        auto ubWsF32    = ubWsF32Buf.Get<float>();
        auto ubKsF32    = ubKsF32Buf.Get<float>();
        auto ubAsF32    = ubAsF32Buf.Get<float>();
        auto ubRowScale = ubRowScaleBuf.Get<float>();
        LocalTensor<bfloat16_t> ubWzBf16;
        LocalTensor<int32_t> ubAkI32;
        LocalTensor<float> ubWzF32, ubAkF32, ubAkBrcb;
        if constexpr (kAsym) {
            ubWzBf16 = ubWzBf16Buf.Get<bfloat16_t>();
            ubAkI32  = ubAkI32Buf.Get<int32_t>();
            ubWzF32  = ubWzF32Buf.Get<float>();
            ubAkF32  = ubAkF32Buf.Get<float>();
            ubAkBrcb = ubAkBrcbBuf.Get<float>();
        } else {
            // Never read (the MulAddDst is compiled out), but AccumulateGroupChunk
            // still indexes them to form its arguments -- alias live buffers so no
            // offset is ever computed against a null tensor.
            ubWzF32  = ubWsF32;
            ubAkBrcb = ubRowScale;
        }

        // One strided DataCopy per scale tensor per batch: group-major layout puts
        // this tile's TILE_N (resp. TILE_M) scales of consecutive groups N (resp. M)
        // apart, so blockCount = the batch size walks the group axis.
        DataCopyParams wsDc, asDc, ksDc;
        wsDc.blockLen  = static_cast<uint16_t>(TILE_N * sizeof(bfloat16_t) / 32);
        wsDc.srcStride = static_cast<uint16_t>((N - TILE_N) * sizeof(bfloat16_t) / 32);
        wsDc.dstStride = 0;
        asDc.blockLen  = static_cast<uint16_t>(TILE_M * sizeof(bfloat16_t) / 32);
        asDc.srcStride = static_cast<uint16_t>((M - TILE_M) * sizeof(bfloat16_t) / 32);
        asDc.dstStride = 0;
        ksDc.blockLen  = static_cast<uint16_t>(TILE_N * sizeof(int32_t) / 32);
        ksDc.srcStride = static_cast<uint16_t>((N - TILE_N) * sizeof(int32_t) / 32);
        ksDc.dstStride = 0;
        // a_ksum is int32 (see the header note), so it needs its own descriptor
        // rather than reusing a_scale's.
        DataCopyParams akDc;
        akDc.blockLen  = static_cast<uint16_t>(TILE_M * sizeof(int32_t) / 32);
        akDc.srcStride = static_cast<uint16_t>((M - TILE_M) * sizeof(int32_t) / 32);
        akDc.dstStride = 0;

        event_t evMte2V0 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        event_t evMte2V1 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        event_t evMte2Vs = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        event_t evVMte3  = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        event_t evMte3V  = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
        event_t evVMte2  = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());

        uint32_t flushIdx = 0;          // mirrors the AIC's global flush order

        for (uint32_t tileId = blockIdx; tileId < numTiles; tileId += blockDim) {
            const uint32_t mTile = tileId / numNTiles;
            const uint32_t nTile = tileId % numNTiles;
            const uint64_t mOff  = (uint64_t)mTile * TILE_M;
            const uint64_t nOff  = (uint64_t)nTile * TILE_N;

            for (uint32_t chunk = 0; chunk < CHUNKS_PER_AIV; chunk++) {
                Duplicate(ubAcc[chunk * CHUNK_ELEMS], 0.0f, CHUNK_ELEMS);
            }

            const uint64_t rowOffHi = (uint64_t)mBase * TILE_N;
            const uint64_t rowOffLo = (uint64_t)(TILE_M + mBase) * TILE_N;

            for (uint32_t gb = 0; gb < numQGroups; gb += BATCH_G) {
                const uint32_t cnt =
                    ((numQGroups - gb) < BATCH_G) ? (numQGroups - gb) : BATCH_G;

                // WAR: the previous batch's combines (V) still read the staging
                // buffers this DataCopy is about to overwrite.
                SetFlag<HardEvent::V_MTE2>(evVMte2);
                WaitFlag<HardEvent::V_MTE2>(evVMte2);

                wsDc.blockCount = static_cast<uint16_t>(cnt);
                asDc.blockCount = static_cast<uint16_t>(cnt);
                ksDc.blockCount = static_cast<uint16_t>(cnt);
                // Both AIVs load the whole tile's a_scale, not just their half: at
                // TILE_M=16 a half is 8 bf16 = 16 B, under the 32 B DataCopy floor.
                DataCopy(ubWsBf16, wScaleGm[(uint64_t)gb * N + nOff], wsDc);
                DataCopy(ubAsBf16, aScaleGm[(uint64_t)gb * M + mOff], asDc);
                DataCopy(ubKsI32,  ksumGm[(uint64_t)gb * N + nOff],   ksDc);
                // w_zero has w_scale's exact shape, so that descriptor is reused.
                if constexpr (kAsym) {
                    akDc.blockCount = static_cast<uint16_t>(cnt);
                    DataCopy(ubWzBf16, wZeroGm[(uint64_t)gb * N + nOff], wsDc);
                    DataCopy(ubAkI32,  aKsumGm[(uint64_t)gb * M + mOff], akDc);
                }
                SetFlag<HardEvent::MTE2_V>(evMte2Vs);
                WaitFlag<HardEvent::MTE2_V>(evMte2Vs);

                // The whole batch casts in one shot; only Brcb has to walk the
                // groups, because its source is TILE_M-strided, not contiguous.
                Cast(ubWsF32, ubWsBf16, RoundMode::CAST_NONE, cnt * TILE_N);
                Cast(ubAsF32, ubAsBf16, RoundMode::CAST_NONE, cnt * TILE_M);
                Cast(ubKsF32, ubKsI32,  RoundMode::CAST_NONE, cnt * TILE_N);
                // The MSD split leaves + 8 * sum_{k in g} W[n,k] on every element.
                Muls(ubKsF32, ubKsF32, 8.0f, cnt * TILE_N);
                for (uint32_t j = 0; j < cnt; j++) {
                    Brcb(ubRowScale[j * myM * 8], ubAsF32[j * TILE_M + mBase],
                         myM / 8, {1, 8});
                }
                if constexpr (kAsym) {
                    Cast(ubWzF32, ubWzBf16, RoundMode::CAST_NONE, cnt * TILE_N);
                    Cast(ubAkF32, ubAkI32,  RoundMode::CAST_NONE, cnt * TILE_M);
                    for (uint32_t j = 0; j < cnt; j++) {
                        Brcb(ubAkBrcb[j * myM * 8], ubAkF32[j * TILE_M + mBase],
                             myM / 8, {1, 8});
                    }
                }

                for (uint32_t j = 0; j < cnt; j++) {
                    const uint32_t slot = flushIdx & 1u;
                    const uint64_t hiRowOff = slot * kWsElems + rowOffHi;
                    const uint64_t loRowOff = slot * kWsElems + rowOffLo;
                    if (slot == 0) CrossCoreWaitFlag(kAicToAivFlag0);
                    else           CrossCoreWaitFlag(kAicToAivFlag1);

                    // WAR on ubHi / ubLo: the previous group's combine (V) is
                    // still reading them.  Sinking this to just before the write
                    // (rather than the end of the group) lets that combine run
                    // under the CrossCoreWaitFlag above.
                    //
                    // j == 0 needs no drain: the batch's V_MTE2 above already
                    // emptied the V pipe, and the only V issued since is the
                    // scale preamble, which never touches ubHi / ubLo.  Draining
                    // there anyway would cost one extra full drain per batch --
                    // measurably so at TILE_M=128, where the pipe holds a whole
                    // 8192-element combine.
                    if (j > 0) {
                        SetFlag<HardEvent::V_MTE2>(evVMte2);
                        WaitFlag<HardEvent::V_MTE2>(evVMte2);
                    }

                    DataCopy(ubHi, wsGm[hiRowOff], CHUNK_ELEMS);
                    DataCopy(ubLo, wsGm[loRowOff], CHUNK_ELEMS);
                    SetFlag<HardEvent::MTE2_V>(evMte2V0);
                    if constexpr (CHUNKS_PER_AIV > 1) {
                        DataCopy(ubHi[CHUNK_ELEMS], wsGm[hiRowOff + CHUNK_ELEMS], CHUNK_ELEMS);
                        DataCopy(ubLo[CHUNK_ELEMS], wsGm[loRowOff + CHUNK_ELEMS], CHUNK_ELEMS);
                        SetFlag<HardEvent::MTE2_V>(evMte2V1);
                    }
                    if (slot == 0) CrossCoreSetFlag<kCrossMode, PIPE_MTE2>(kAivToAicFlag0);
                    else           CrossCoreSetFlag<kCrossMode, PIPE_MTE2>(kAivToAicFlag1);
                    flushIdx++;

                    const uint32_t so = j * TILE_N;
                    const uint32_t ro = j * myM * 8;
                    WaitFlag<HardEvent::MTE2_V>(evMte2V0);
                    AccumulateGroupChunk(ubHi, ubLo, ubTemp, ubAcc,
                                         ubWsF32[so], ubKsF32[so], ubRowScale[ro],
                                         ubWzF32[so], ubAkBrcb[ro], 0);
                    if constexpr (CHUNKS_PER_AIV > 1) {
                        WaitFlag<HardEvent::MTE2_V>(evMte2V1);
                        AccumulateGroupChunk(ubHi, ubLo, ubTemp, ubAcc,
                                             ubWsF32[so], ubKsF32[so], ubRowScale[ro],
                                             ubWzF32[so], ubAkBrcb[ro], 1);
                    }
                }
            }

            DataCopyParams outDcParams;
            outDcParams.blockCount = static_cast<uint16_t>(CHUNK_M);
            outDcParams.blockLen   = static_cast<uint16_t>(TILE_N * sizeof(uint16_t) / 32);
            outDcParams.srcStride  = 0;
            outDcParams.dstStride  = static_cast<uint16_t>((N - TILE_N) * sizeof(uint16_t) / 32);

            Cast(ubOut, ubAcc, RoundMode::CAST_ROUND, CHUNK_ELEMS);
            SetFlag<HardEvent::V_MTE3>(evVMte3);
            WaitFlag<HardEvent::V_MTE3>(evVMte3);
            DataCopy(outGm[(mOff + mBase) * N + nOff], ubOut, outDcParams);

            if constexpr (CHUNKS_PER_AIV > 1) {
                Cast(ubOut[CHUNK_ELEMS], ubAcc[CHUNK_ELEMS],
                     RoundMode::CAST_ROUND, CHUNK_ELEMS);
                SetFlag<HardEvent::V_MTE3>(evVMte3);
                WaitFlag<HardEvent::V_MTE3>(evVMte3);
                DataCopy(outGm[(mOff + mBase + CHUNK_M) * N + nOff],
                         ubOut[CHUNK_ELEMS], outDcParams);
            }

            // The next tile's Duplicate / Cast (V) overwrites ubAcc and ubOut,
            // which the output DataCopy (MTE3) is still reading.
            SetFlag<HardEvent::MTE3_V>(evMte3V);
            WaitFlag<HardEvent::MTE3_V>(evMte3V);
        }

        pipe.ReleaseEventID<HardEvent::MTE2_V>(evMte2V0);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(evMte2V1);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(evMte2Vs);
        pipe.ReleaseEventID<HardEvent::V_MTE3>(evVMte3);
        pipe.ReleaseEventID<HardEvent::MTE3_V>(evMte3V);
        pipe.ReleaseEventID<HardEvent::V_MTE2>(evVMte2);

        event_t evMte3S = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_S>());
        SetFlag<HardEvent::MTE3_S>(evMte3S);
        WaitFlag<HardEvent::MTE3_S>(evMte3S);
        pipe.ReleaseEventID<HardEvent::MTE3_S>(evMte3S);
    }

    pipe.Destroy();
}
