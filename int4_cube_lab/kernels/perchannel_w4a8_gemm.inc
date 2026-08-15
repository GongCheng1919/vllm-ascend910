// perchannel_w4a8_gemm.inc -- shared body for the W4A8 per-channel GEMM.
//
// The including .cpp must define, before the include:
//   TILE_M_CFG   output rows per tile (16 / 32 / 64 / 128)
//   KERNEL_NAME  the extern "C" __global__ symbol to emit
//
// Contract:
//   a_hi:    [M, K/2] int4 packed   -- high nibbles of the int8 activation
//   a_lo:    [M, K/2] int4 packed   -- low nibbles, biased by -8
//   a_scale: [M] bf16   (per-token)
//   w_q:     [N, K/2] int4 packed, row-major; computes A @ W.T
//   w_scale: [N] bf16   (per-channel)
//   w_ksum:  [N] int32  -- sum_k W[n,k], the MSD bias-correction term
//   y:       [M, N] bf16
//   workspace: [blockDim, 2*TILE_M, 128] int32
//
// int4 packing is the same as the W4A4 kernel: 2 elements per byte along K,
// even k in the low nibble.
//
// Why a W4A8 kernel looks like this
// ---------------------------------
// The cube has no mixed-width MAD: `MmadCal` accepts (int32,int8,int8) and
// (int32,int4b_t,int4b_t) and nothing else.  So an int8 activation reaches the
// s4 pipe only via a most-significant-digit split
//
//     a = 16*h + l + 8,   h = a >> 4        in [-8,7]
//                         l = (a & 15) ^ 8  in [-8,7]
//
// which is exact for every int8 (no clamping of the quantiser's range).  Both
// planes are int4, so A costs K/2 + K/2 = K bytes -- identical to int8 A -- while
// W costs K/2 bytes instead of K.  Two s4 MADs at 2x rate take exactly as long
// as one s8 MAD, so W4A8 should match W8A8 on cube time and beat it on every
// byte moved.  That is the hypothesis this kernel exists to measure.
//
// The +8 in the split leaves a term 8 * sum_k W[n,k], constant along m.  It is
// passed in as `w_ksum` (a weight-side preprocessing quantity, computed once)
// and folded in on the AIV.
//
// Stacked-A layout
// ----------------
// Rather than issue two MADs into two accumulators, the h-plane and l-plane
// tiles are stacked along M into one 2*TILE_M cube tile that shares one B, one
// Mmad, one L0C and one Fixpipe.  Every descriptor below is then literally the
// W4A4 kernel's with TILE_M -> TILE_MC.  The AIV recombines
// 16*C[0:TILE_M] + C[TILE_M:2*TILE_M] when it drains the workspace.
//
// Pipeline (per output tile, one AIC + two AIV) -- unchanged from W4A4:
//   * L1 depth-2 ping-pong, L0 depth-2 ping-pong
//   * one L0C accumulator across the whole K -- one Fixpipe per output tile
//   * one CrossCore handshake pair per output tile
//   * cross-tile g=0 prefetch issued while the AIVs drain the workspace

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_type.h"
#include "basic_api/kernel_operator_fixpipe_intf.h"

using namespace AscendC;

namespace {

#ifndef TILE_M_CFG
#define TILE_M_CFG 128
#endif

constexpr uint32_t TILE_M  = TILE_M_CFG;         // output rows per tile
constexpr uint32_t TILE_N  = 128;
constexpr uint32_t TILE_MC = 2 * TILE_M;         // cube rows: h-plane then l-plane
constexpr uint32_t HALF_M  = TILE_M / 2;         // output rows per AIV

constexpr uint32_t CHUNK_M        = (HALF_M < 32) ? HALF_M : 32;
constexpr uint32_t CHUNK_ELEMS    = CHUNK_M * TILE_N;
constexpr uint32_t CHUNKS_PER_AIV = HALF_M / CHUNK_M;

static_assert(TILE_M >= 16 && (TILE_M % 16) == 0, "TILE_M must be a multiple of 16");
static_assert((HALF_M % 8) == 0, "TILE_M/2 must be a multiple of 8 (Brcb repeat)");
static_assert(CHUNKS_PER_AIV <= 2, "AIV pipeline is written for at most 2 chunks");

// ---- K geometry, in BYTES ----
// L0A now holds 2*TILE_M rows, so the slice depth has to shrink once TILE_M
// passes 64 or the depth-2 ping-pong no longer fits in the 64 KB L0A.
constexpr uint32_t INNER_KB         = (TILE_M <= 64) ? 256 : 128;
constexpr uint32_t SLICES_PER_GROUP = 4;
constexpr uint32_t GROUP_KB         = INNER_KB * SLICES_PER_GROUP;

// ---- Mmad k is in ELEMENTS (int4: 2 per byte) ----
constexpr uint32_t INNER_K = INNER_KB * 2;

constexpr uint32_t kBlock        = 16;
constexpr uint32_t kK0B          = 32;                   // C0 in bytes
constexpr uint32_t kMBlocks      = TILE_MC / kBlock;
constexpr uint32_t kNBlocks      = TILE_N / kBlock;
constexpr uint32_t kKBlocksInner = INNER_KB / kK0B;      // fractals along K
constexpr uint32_t kNzBlockSize  = kK0B * kBlock;        // 512 B per fractal
constexpr uint32_t kSliceStrideA = INNER_KB * TILE_MC;
constexpr uint32_t kSliceStrideB = INNER_KB * TILE_N;

// On-chip budgets (910B4): L1 512 KB, L0A/L0B 64 KB each, L0C 128 KB, UB 192 KB.
static_assert(2 * TILE_MC * INNER_KB <= 64 * 1024, "L0A over budget");
static_assert(2 * TILE_N  * INNER_KB <= 64 * 1024, "L0B over budget");
static_assert(TILE_MC * TILE_N * sizeof(int32_t) <= 128 * 1024, "L0C over budget");
static_assert(2 * (TILE_MC + TILE_N) * GROUP_KB <= 512 * 1024, "L1 over budget");

constexpr uint16_t kAivToAicFlag = 3;
constexpr uint16_t kAicToAivFlag = 5;
constexpr uint8_t  kCrossMode    = 2;

// ---- AIV scratch offsets inside one small TBuf (bytes 0..2559) ----
// bf16 staging | int32 staging | fp32 results, kept disjoint so the Casts below
// can read and write the same TBuf.
constexpr uint32_t kOffWsBf16  = 0;      // bf16 index: TILE_N w_scale
constexpr uint32_t kOffAsBf16  = 128;    // bf16 index: TILE_M a_scale
constexpr uint32_t kOffKsI32   = 128;    // int32 index: TILE_N w_ksum
constexpr uint32_t kOffWsF32   = 256;    // float index
constexpr uint32_t kOffKsF32   = 384;    // float index
constexpr uint32_t kOffAsF32   = 512;    // float index

// L1 -> L0 for one INNER_KB slice.  Everything is a byte view up to the call,
// where it is reinterpreted to int4b_t so AscendC dispatches to
// load_cbuf_to_ca_s4 / load_cbuf_to_cb_s4 rather than the generic b8 loads.
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

// Issue the GM->L1 triple for one K-group: the h-plane and l-plane of A stack
// into one 2*TILE_M-row NZ tile (dstNzC0Stride = TILE_MC, l-plane starting one
// C0-row block of TILE_M rows in), plus B.
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
    DataCopy(aL1,                  aHiGm[aBase + gOffB], ndA);
    DataCopy(aL1[TILE_M * kK0B],   aLoGm[aBase + gOffB], ndA);
    DataCopy(bL1,                  bGm[bBase + gOffB],   ndB);
    a1Queue.EnQue(aL1);
    b1Queue.EnQue(bL1);
}

// out = ((16*C_hi + C_lo) + 8*w_ksum[n]) * w_scale[n] * a_scale[m], in place in
// `acc`, on a CHUNK_M-row chunk.  The int32 combine happens before the fp32 cast
// so the two partial products add exactly.
__aicore__ inline void CombineAndScaleChunk(const LocalTensor<int32_t>& hi,
                                            const LocalTensor<int32_t>& lo,
                                            const LocalTensor<float>& acc,
                                            const LocalTensor<float>& misc,
                                            const LocalTensor<float>& rowScaleBrcb,
                                            uint32_t chunk)
{
    const uint32_t off         = chunk * CHUNK_ELEMS;
    const uint32_t rowScaleOff = chunk * CHUNK_M * 8;

    Muls(hi[off], hi[off], 16, CHUNK_ELEMS);
    Add(hi[off], hi[off], lo[off], CHUNK_ELEMS);
    Cast(acc[off], hi[off], RoundMode::CAST_NONE, CHUNK_ELEMS);

    // src1 is a 64-float row re-read on every repeat (src1RepStride = 0).
    BinaryRepeatParams rowParam(1, 1, 1, 16, 16, 0);
    // src1 is one 8-float Brcb block per repeat (src1BlkStride 0, RepStride 1).
    BinaryRepeatParams colParam(1, 1, 0, 16, 16, 1);

    for (uint32_t half = 0; half < 2; half++) {
        const uint32_t o = off + half * 64;
        const uint32_t s = half * 64;
        Add(acc[o], acc[o], misc[kOffKsF32 + s], 64, CHUNK_M, rowParam);
        Mul(acc[o], acc[o], misc[kOffWsF32 + s], 64, CHUNK_M, rowParam);
        Mul(acc[o], acc[o], rowScaleBrcb[rowScaleOff], 64, CHUNK_M, colParam);
    }
}

}  // namespace

extern "C" __global__ __aicore__
void KERNEL_NAME(GM_ADDR a_hi, GM_ADDR a_lo, GM_ADDR a_scale,
                 GM_ADDR w_q,  GM_ADDR w_scale, GM_ADDR w_ksum,
                 GM_ADDR y,    GM_ADDR workspace,
                 uint32_t M, uint32_t N, uint32_t K)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    const uint32_t blockIdx  = GetBlockIdx() / GetTaskRation();
    const uint32_t blockDim  = GetBlockNum();
    const uint32_t numNTiles = N / TILE_N;
    const uint32_t numTiles  = (M / TILE_M) * numNTiles;

    const uint32_t Kb        = K / 2;                       // bytes per row (int4)
    const uint32_t numSlices = Kb / INNER_KB;
    const uint32_t numG      = (numSlices + SLICES_PER_GROUP - 1) / SLICES_PER_GROUP;
    const uint32_t lastGSlices = numSlices - (numG - 1) * SLICES_PER_GROUP;
    const uint32_t g0Slices  = (numG == 1) ? lastGSlices : SLICES_PER_GROUP;

    constexpr uint32_t kWsElems = TILE_MC * TILE_N;
    GlobalTensor<int32_t> wsGm;
    wsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(workspace) + (uint64_t)blockIdx * kWsElems,
        kWsElems);

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
        TQue<QuePosition::CO1, 1> c1Queue;
        pipe.InitBuffer(a1Queue, 2, TILE_MC * GROUP_KB);
        pipe.InitBuffer(b1Queue, 2, TILE_N  * GROUP_KB);
        pipe.InitBuffer(a2Queue, 2, TILE_MC * INNER_KB);
        pipe.InitBuffer(b2Queue, 2, TILE_N  * INNER_KB);
        pipe.InitBuffer(c1Queue, 1, kWsElems * sizeof(int32_t));

        // One plane at a time: nValue = TILE_M rows, but the destination NZ tile
        // is TILE_MC rows tall so the two planes interleave per C0 block.
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

        bool firstTile = true;
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

            auto cL0 = c1Queue.AllocTensor<int32_t>();
            bool initC = true;

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

                    mm.cmatrixInitVal = initC;
                    initC = false;
                    Mmad(cL0, aL0.ReinterpretCast<int4b_t>(),
                         bL0.ReinterpretCast<int4b_t>(), mm);

                    a2Queue.FreeTensor(aL0);
                    b2Queue.FreeTensor(bL0);
                }
            }

            c1Queue.EnQue(cL0);
            cL0 = c1Queue.DeQue<int32_t>();

            if (!firstTile) {
                CrossCoreWaitFlag(kAivToAicFlag);
            }
            firstTile = false;

            Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(wsGm, cL0, fp);
            c1Queue.FreeTensor(cL0);
            CrossCoreSetFlag<kCrossMode, PIPE_FIX>(kAicToAivFlag);

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

        if (numTiles > blockIdx) {
            CrossCoreWaitFlag(kAivToAicFlag);
        }
    }

    // ============================ AIV ============================
    if ASCEND_IS_AIV {
        const uint32_t subIdx = static_cast<uint32_t>(GetSubBlockIdx());
        const uint32_t mBase  = subIdx * HALF_M;
        constexpr uint32_t myM     = HALF_M;
        constexpr uint32_t myElems = HALF_M * TILE_N;

        GlobalTensor<bfloat16_t> aScaleGm, wScaleGm, outGm;
        GlobalTensor<int32_t> ksumGm;
        aScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a_scale), M);
        wScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w_scale), N);
        ksumGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(w_ksum), N);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y), (uint64_t)M * N);

        TBuf<QuePosition::VECCALC> ubHiBuf, ubLoBuf, ubAccBuf, ubOutBuf;
        TBuf<QuePosition::VECCALC> ubMiscBuf, ubRowScaleBuf;

        pipe.InitBuffer(ubHiBuf,       myElems * sizeof(int32_t));
        pipe.InitBuffer(ubLoBuf,       myElems * sizeof(int32_t));
        pipe.InitBuffer(ubAccBuf,      myElems * sizeof(float));
        pipe.InitBuffer(ubOutBuf,      myElems * sizeof(bfloat16_t));
        pipe.InitBuffer(ubMiscBuf,     4096);
        pipe.InitBuffer(ubRowScaleBuf, myM * 8 * sizeof(float));

        auto ubHi        = ubHiBuf.Get<int32_t>();
        auto ubLo        = ubLoBuf.Get<int32_t>();
        auto ubAcc       = ubAccBuf.Get<float>();
        auto ubOut       = ubOutBuf.Get<bfloat16_t>();
        auto ubMiscBf16  = ubMiscBuf.Get<bfloat16_t>();
        auto ubMiscI32   = ubMiscBuf.Get<int32_t>();
        auto ubMiscF32   = ubMiscBuf.Get<float>();
        auto ubRowScale  = ubRowScaleBuf.Get<float>();

        event_t evMte2V0 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        event_t evMte2V1 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        event_t evVMte3  = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        event_t evMte3V  = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
        event_t evVMte2  = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());

        for (uint32_t tileId = blockIdx; tileId < numTiles; tileId += blockDim) {
            const uint32_t mTile = tileId / numNTiles;
            const uint32_t nTile = tileId % numNTiles;
            const uint64_t mOff  = (uint64_t)mTile * TILE_M;
            const uint64_t nOff  = (uint64_t)nTile * TILE_N;

            // Both AIVs load the whole tile's a_scale, not just their half: at
            // TILE_M=16 a half is 8 bf16 = 16 B, under the 32 B DataCopy floor.
            DataCopy(ubMiscBf16[kOffWsBf16], wScaleGm[nOff], TILE_N);
            DataCopy(ubMiscBf16[kOffAsBf16], aScaleGm[mOff], TILE_M);
            DataCopy(ubMiscI32[kOffKsI32],   ksumGm[nOff],   TILE_N);
            SetFlag<HardEvent::MTE2_V>(evMte2V0);
            WaitFlag<HardEvent::MTE2_V>(evMte2V0);
            Cast(ubMiscF32[kOffWsF32], ubMiscBf16[kOffWsBf16], RoundMode::CAST_NONE, TILE_N);
            Cast(ubMiscF32[kOffAsF32], ubMiscBf16[kOffAsBf16], RoundMode::CAST_NONE, TILE_M);
            Cast(ubMiscF32[kOffKsF32], ubMiscI32[kOffKsI32],   RoundMode::CAST_NONE, TILE_N);
            // The MSD split leaves + 8 * sum_k W[n,k] on every output element.
            Muls(ubMiscF32[kOffKsF32], ubMiscF32[kOffKsF32], 8.0f, TILE_N);
            Brcb(ubRowScale, ubMiscF32[kOffAsF32 + mBase], myM / 8, {1, 8});

            CrossCoreWaitFlag(kAicToAivFlag);

            // C_hi occupies workspace rows [0, TILE_M), C_lo rows [TILE_M, 2*TILE_M).
            const uint64_t hiRowOff = (uint64_t)mBase * TILE_N;
            const uint64_t loRowOff = (uint64_t)(TILE_M + mBase) * TILE_N;
            DataCopy(ubHi, wsGm[hiRowOff], CHUNK_ELEMS);
            DataCopy(ubLo, wsGm[loRowOff], CHUNK_ELEMS);
            SetFlag<HardEvent::MTE2_V>(evMte2V0);
            if constexpr (CHUNKS_PER_AIV > 1) {
                DataCopy(ubHi[CHUNK_ELEMS], wsGm[hiRowOff + CHUNK_ELEMS], CHUNK_ELEMS);
                DataCopy(ubLo[CHUNK_ELEMS], wsGm[loRowOff + CHUNK_ELEMS], CHUNK_ELEMS);
                SetFlag<HardEvent::MTE2_V>(evMte2V1);
            }
            CrossCoreSetFlag<kCrossMode, PIPE_MTE2>(kAivToAicFlag);

            DataCopyParams outDcParams;
            outDcParams.blockCount = static_cast<uint16_t>(CHUNK_M);
            outDcParams.blockLen   = static_cast<uint16_t>(TILE_N * sizeof(uint16_t) / 32);
            outDcParams.srcStride  = 0;
            outDcParams.dstStride  = static_cast<uint16_t>((N - TILE_N) * sizeof(uint16_t) / 32);

            WaitFlag<HardEvent::MTE2_V>(evMte2V0);
            CombineAndScaleChunk(ubHi, ubLo, ubAcc, ubMiscF32, ubRowScale, 0);
            Cast(ubOut, ubAcc, RoundMode::CAST_ROUND, CHUNK_ELEMS);
            SetFlag<HardEvent::V_MTE3>(evVMte3);
            WaitFlag<HardEvent::V_MTE3>(evVMte3);
            DataCopy(outGm[(mOff + mBase) * N + nOff], ubOut, outDcParams);

            if constexpr (CHUNKS_PER_AIV > 1) {
                WaitFlag<HardEvent::MTE2_V>(evMte2V1);
                CombineAndScaleChunk(ubHi, ubLo, ubAcc, ubMiscF32, ubRowScale, 1);
                Cast(ubOut[CHUNK_ELEMS], ubAcc[CHUNK_ELEMS],
                     RoundMode::CAST_ROUND, CHUNK_ELEMS);
                SetFlag<HardEvent::V_MTE3>(evVMte3);
                WaitFlag<HardEvent::V_MTE3>(evVMte3);
                DataCopy(outGm[(mOff + mBase + CHUNK_M) * N + nOff],
                         ubOut[CHUNK_ELEMS], outDcParams);
            }

            // Write-after-read barriers before the next tile reuses these UBs:
            //   MTE3_V  — the output DataCopy still reads ubOut.
            //   V_MTE2  — the int32 combine still reads ubHi / ubLo.
            SetFlag<HardEvent::MTE3_V>(evMte3V);
            WaitFlag<HardEvent::MTE3_V>(evMte3V);
            SetFlag<HardEvent::V_MTE2>(evVMte2);
            WaitFlag<HardEvent::V_MTE2>(evVMte2);
        }

        pipe.ReleaseEventID<HardEvent::MTE2_V>(evMte2V0);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(evMte2V1);
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
