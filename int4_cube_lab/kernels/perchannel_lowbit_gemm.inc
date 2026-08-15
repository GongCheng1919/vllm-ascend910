// perchannel_lowbit_gemm.inc -- shared body for INT4 and INT8 per-channel GEMM.
//
// The including .cpp must define, before the include:
//   QBITS        4 or 8   -- element width of A and W
//   KERNEL_NAME  the extern "C" __global__ symbol to emit
//
// Contract (identical for both widths):
//   A:       [M, K] int4/int8, row-major   (int4 packed 2 elems/byte along K,
//                                           even k in the low nibble)
//   W:       [N, K] int4/int8, row-major   (same packing); computes A @ W.T
//   a_scale: [M] bf16   (per-token)
//   w_scale: [N] bf16   (per-channel)
//   Y:       [M, N] bf16
//   workspace: [blockDim, 128, 128] int32
//
// Why one source for both widths
// ------------------------------
// Every buffer, DMA descriptor and LoadData descriptor below is expressed in
// BYTES, not elements.  An INT4 fractal (16 x 64 elems) and an INT8 fractal
// (16 x 32 elems) are both 512 bytes, and C0 is 32 bytes either way, so the
// two builds move byte-for-byte identical traffic and issue exactly the same
// number of Nd2Nz / LoadData / Mmad instructions.  The only difference is that
// each Mmad covers twice the K in the INT4 build (mm.k = INNER_KB * 8/QBITS).
// That makes the INT8 build a controlled baseline: if the s4 cube pipe really
// is 2x the s8 pipe, the two builds run in the SAME wall time for the same
// byte-K, i.e. INT4 delivers 2x the FLOPS.
//
// Pipeline (per output tile, one AIC + two AIV):
//   * L1 depth-2 ping-pong: GM->L1 of group g+1 overlaps L1->L0 + Mmad of g
//   * L0 depth-2 ping-pong: LoadData of slice s+1 overlaps Mmad of slice s
//   * one L0C accumulator across the whole K -- one Fixpipe per output tile
//   * one CrossCore handshake pair per output tile
//   * cross-tile g=0 prefetch issued while the AIVs drain the workspace

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_type.h"
#include "basic_api/kernel_operator_fixpipe_intf.h"

#if (QBITS != 4) && (QBITS != 8)
#error "QBITS must be 4 or 8"
#endif

using namespace AscendC;

namespace {

constexpr uint32_t ELEMS_PER_BYTE = 8 / QBITS;   // 2 for int4, 1 for int8

// TILE_M is a build parameter: small-M (decode) shapes want a short tile so the
// cube is not padded up to 128 rows.  16 is the floor — that is the fractal
// height, so any cube kernel pads to it.  Must be a multiple of 16, and the two
// AIVs each take TILE_M/2 rows, so TILE_M/2 must be a multiple of 8 (Brcb).
#ifndef TILE_M_CFG
#define TILE_M_CFG 128
#endif

constexpr uint32_t TILE_M         = TILE_M_CFG;
constexpr uint32_t TILE_N         = 128;
constexpr uint32_t HALF_M         = TILE_M / 2;
constexpr uint32_t TILE_ELEMS     = TILE_M * TILE_N;
constexpr uint32_t HALF_ELEMS     = HALF_M * TILE_N;
constexpr uint32_t CHUNK_M        = (HALF_M < 32) ? HALF_M : 32;
constexpr uint32_t CHUNK_ELEMS    = CHUNK_M * TILE_N;
constexpr uint32_t CHUNKS_PER_AIV = HALF_M / CHUNK_M;

static_assert(TILE_M >= 16 && (TILE_M % 16) == 0, "TILE_M must be a multiple of 16");
static_assert((HALF_M % 8) == 0, "TILE_M/2 must be a multiple of 8 (Brcb repeat)");
static_assert(CHUNKS_PER_AIV <= 2, "AIV pipeline is written for at most 2 chunks");

// ---- K geometry, in BYTES (shared by both widths) ----
constexpr uint32_t GROUP_KB         = 1024;                 // K bytes per L1 buffer
constexpr uint32_t INNER_KB         = 256;                  // K bytes per L0 slice
constexpr uint32_t SLICES_PER_GROUP = GROUP_KB / INNER_KB;  // 4

// ---- Mmad k is in ELEMENTS ----
constexpr uint32_t INNER_K = INNER_KB * ELEMS_PER_BYTE;     // 512 (int4) / 256 (int8)

constexpr uint32_t kBlock          = 16;
constexpr uint32_t kK0B            = 32;                    // C0 in bytes
constexpr uint32_t kMBlocks        = TILE_M / kBlock;       // 8
constexpr uint32_t kNBlocks        = TILE_N / kBlock;       // 8
constexpr uint32_t kKBlocksInner   = INNER_KB / kK0B;       // 8 fractals along K
constexpr uint32_t kNzBlockSize    = kK0B * kBlock;         // 512 B per fractal
// Byte distance between consecutive K-slices inside an L1 buffer.  A holds
// TILE_M rows and B holds TILE_N rows, so these differ whenever TILE_M != TILE_N
// — sharing one constant silently works only for the square 128x128 tile.
constexpr uint32_t kSliceStrideA   = INNER_KB * TILE_M;
constexpr uint32_t kSliceStrideB   = INNER_KB * TILE_N;

constexpr uint16_t kAivToAicFlag = 3;
constexpr uint16_t kAicToAivFlag = 5;
constexpr uint8_t  kCrossMode    = 2;

// L1 -> L0 for one INNER_KB slice.  aL1/aL0/bL1/bL0 are byte views; for QBITS==4
// the tensors are reinterpreted to int4b_t at the call so AscendC dispatches to
// load_cbuf_to_ca_s4 / load_cbuf_to_cb_s4 instead of the generic b8 loads.
__aicore__ inline void LoadSliceL1ToL0(const LocalTensor<int8_t>& aL0,
                                       const LocalTensor<int8_t>& bL0,
                                       const LocalTensor<int8_t>& aL1,
                                       const LocalTensor<int8_t>& bL1,
                                       const LoadData2dParams& ldA,
                                       const LoadData2dParams& ldB)
{
    for (uint32_t mi = 0; mi < kMBlocks; mi++) {
#if QBITS == 4
        LoadData(aL0[mi * kKBlocksInner * kNzBlockSize].ReinterpretCast<int4b_t>(),
                 aL1[mi * kNzBlockSize].ReinterpretCast<int4b_t>(), ldA);
#else
        LoadData(aL0[mi * kKBlocksInner * kNzBlockSize],
                 aL1[mi * kNzBlockSize], ldA);
#endif
    }
#if QBITS == 4
    LoadData(bL0.ReinterpretCast<int4b_t>(), bL1.ReinterpretCast<int4b_t>(), ldB);
#else
    LoadData(bL0, bL1, ldB);
#endif
}

__aicore__ inline void MmadSlice(const LocalTensor<int32_t>& cL0,
                                 const LocalTensor<int8_t>& aL0,
                                 const LocalTensor<int8_t>& bL0,
                                 const MmadParams& mm)
{
#if QBITS == 4
    Mmad(cL0, aL0.ReinterpretCast<int4b_t>(), bL0.ReinterpretCast<int4b_t>(), mm);
#else
    Mmad(cL0, aL0, bL0, mm);
#endif
}

// Issue the GM->L1 pair for one K-group.  gOffB is the group's byte offset
// inside the tile's K extent; `slices` says how much of the group is live
// (only the last group of a tile can be partial).
template <typename A1Q, typename B1Q>
__aicore__ inline void IssueGroupLoad(A1Q& a1Queue, B1Q& b1Queue,
                                      const GlobalTensor<int8_t>& aGm,
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
    DataCopy(aL1, aGm[aBase + gOffB], ndA);
    DataCopy(bL1, bGm[bBase + gOffB], ndB);
    a1Queue.EnQue(aL1);
    b1Queue.EnQue(bL1);
}

// acc += (int32 tile as fp32) * w_scale[n] * a_scale[m], on a CHUNK_M-row chunk.
__aicore__ inline void ApplyScaledChunk(const LocalTensor<float>& acc,
                                        const LocalTensor<float>& temp,
                                        const LocalTensor<float>& rowScaleBrcb,
                                        const LocalTensor<float>& scale,
                                        uint32_t chunk)
{
    const uint32_t chunkOff    = chunk * CHUNK_ELEMS;
    const uint32_t rowScaleOff = chunk * CHUNK_M * 8;

    BinaryRepeatParams wScaleParam(1, 1, 1, 16, 16, 0);
    BinaryRepeatParams aScaleParam(1, 1, 0, 16, 16, 1);

    Mul(temp[chunkOff], temp[chunkOff], scale[64], 64, CHUNK_M, wScaleParam);
    MulAddDst(acc[chunkOff], temp[chunkOff], rowScaleBrcb[rowScaleOff], 64, CHUNK_M, aScaleParam);
    Mul(temp[chunkOff + 64], temp[chunkOff + 64], scale[128], 64, CHUNK_M, wScaleParam);
    MulAddDst(acc[chunkOff + 64], temp[chunkOff + 64], rowScaleBrcb[rowScaleOff], 64, CHUNK_M, aScaleParam);
}

}  // namespace

extern "C" __global__ __aicore__
void KERNEL_NAME(GM_ADDR a_q, GM_ADDR a_scale,
                 GM_ADDR w_q, GM_ADDR w_scale,
                 GM_ADDR y,   GM_ADDR workspace,
                 uint32_t M, uint32_t N, uint32_t K)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    const uint32_t blockIdx  = GetBlockIdx() / GetTaskRation();
    const uint32_t blockDim  = GetBlockNum();
    const uint32_t numNTiles = N / TILE_N;
    const uint32_t numTiles  = (M / TILE_M) * numNTiles;

    // K in bytes, and the flat slice / group decomposition.
    const uint32_t Kb        = K / ELEMS_PER_BYTE;
    const uint32_t numSlices = Kb / INNER_KB;
    const uint32_t numG      = (numSlices + SLICES_PER_GROUP - 1) / SLICES_PER_GROUP;
    const uint32_t lastGSlices = numSlices - (numG - 1) * SLICES_PER_GROUP;
    const uint32_t g0Slices  = (numG == 1) ? lastGSlices : SLICES_PER_GROUP;

    GlobalTensor<int32_t> wsGm;
    wsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(workspace) + (uint64_t)blockIdx * TILE_ELEMS,
        TILE_ELEMS);

    // ============================ AIC ============================
    if ASCEND_IS_AIC {
        GlobalTensor<int8_t> aGm, bGm;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a_q), (uint64_t)M * Kb);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w_q), (uint64_t)N * Kb);

        TQue<QuePosition::A1, 2> a1Queue;
        TQue<QuePosition::B1, 2> b1Queue;
        TQue<QuePosition::A2, 2> a2Queue;
        TQue<QuePosition::B2, 2> b2Queue;
        TQue<QuePosition::CO1, 1> c1Queue;
        pipe.InitBuffer(a1Queue, 2, TILE_M * GROUP_KB);             // 2 x 128 KB L1
        pipe.InitBuffer(b1Queue, 2, TILE_N * GROUP_KB);             // 2 x 128 KB L1
        pipe.InitBuffer(a2Queue, 2, TILE_M * INNER_KB);             // 2 x 32 KB L0A
        pipe.InitBuffer(b2Queue, 2, TILE_N * INNER_KB);             // 2 x 32 KB L0B
        pipe.InitBuffer(c1Queue, 1, TILE_ELEMS * sizeof(int32_t));  // 1 x 64 KB L0C

        Nd2NzParams ndA;
        ndA.ndNum             = 1;
        ndA.nValue            = TILE_M;
        ndA.dValue            = GROUP_KB;                 // overwritten per group
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue         = static_cast<uint16_t>(Kb);
        ndA.dstNzC0Stride     = TILE_M;
        ndA.dstNzNStride      = 1;
        ndA.dstNzMatrixStride = 0;

        Nd2NzParams ndB = ndA;
        ndB.nValue        = TILE_N;
        ndB.dstNzC0Stride = TILE_N;

        // A: one call per m-block, srcStride = number of m-blocks in L1.
        LoadData2dParams ldA;
        ldA.startIndex  = 0;
        ldA.repeatTimes = static_cast<uint8_t>(kKBlocksInner);
        ldA.srcStride   = static_cast<uint16_t>(kMBlocks);
        ldA.dstGap      = 0;
        ldA.ifTranspose = false;
        ldA.sid         = 0;
        ldA.addrMode    = 0;

        // B: single call, contiguous fractals.
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
        fp.mSize       = TILE_M;
        fp.srcStride   = TILE_M;
        fp.dstStride   = TILE_N;
        fp.quantPre    = QuantMode_t::NoQuant;
        fp.ndNum       = 1;
        fp.srcNdStride = 1;
        fp.dstNdStride = 1;

        MmadParams mm;
        mm.m             = TILE_M;
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
                IssueGroupLoad(a1Queue, b1Queue, aGm, bGm, ndA, ndB,
                               aTileBase, bTileBase, 0, g0Slices);
            }
            tileG0Prefetched = false;

            auto cL0 = c1Queue.AllocTensor<int32_t>();
            bool initC = true;

            for (uint32_t g = 0; g < numG; g++) {
                const uint32_t slices = (g + 1 == numG) ? lastGSlices : SLICES_PER_GROUP;

                // Intra-tile L1 prefetch for g+1 (depth-2 ping-pong gives the slot).
                if (g + 1 < numG) {
                    const uint32_t nextSlices =
                        (g + 2 == numG) ? lastGSlices : SLICES_PER_GROUP;
                    IssueGroupLoad(a1Queue, b1Queue, aGm, bGm, ndA, ndB,
                                   aTileBase, bTileBase,
                                   (uint64_t)(g + 1) * GROUP_KB, nextSlices);
                }

                auto aL1 = a1Queue.DeQue<int8_t>();
                auto bL1 = b1Queue.DeQue<int8_t>();

                // Prime the L0 ping-pong with slice 0 of this group.
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

                    // Release L1 before the last Mmad so the next group's
                    // GM->L1 refill overlaps this Mmad.
                    if (s + 1 == slices) {
                        a1Queue.FreeTensor(aL1);
                        b1Queue.FreeTensor(bL1);
                    }

                    mm.cmatrixInitVal = initC;
                    initC = false;
                    MmadSlice(cL0, aL0, bL0, mm);

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

            // Kick off the next tile's g=0 GM->L1 while the AIVs read workspace.
            const uint32_t nextTileId = tileId + blockDim;
            if (nextTileId < numTiles) {
                const uint32_t nextMTile = nextTileId / numNTiles;
                const uint32_t nextNTile = nextTileId % numNTiles;
                IssueGroupLoad(a1Queue, b1Queue, aGm, bGm, ndA, ndB,
                               (uint64_t)nextMTile * TILE_M * Kb,
                               (uint64_t)nextNTile * TILE_N * Kb, 0, g0Slices);
                tileG0Prefetched = true;
            }
        }

        // Drain the last outstanding AIV->AIC credit.
        if (numTiles > blockIdx) {
            CrossCoreWaitFlag(kAivToAicFlag);
        }
    }

    // ============================ AIV ============================
    if ASCEND_IS_AIV {
        const uint32_t subIdx = static_cast<uint32_t>(GetSubBlockIdx());
        const uint32_t mBase  = subIdx * HALF_M;
        constexpr uint32_t myM     = HALF_M;
        constexpr uint32_t myElems = HALF_ELEMS;

        GlobalTensor<bfloat16_t> aScaleGm, wScaleGm, outGm;
        aScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a_scale), M);
        wScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w_scale), N);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y), (uint64_t)M * N);

        TBuf<QuePosition::VECCALC> ubInt32Buf, ubFp32AccBuf, ubFp32TempBuf;
        TBuf<QuePosition::VECCALC> ubRowScaleBuf, ubScaleBuf, ubBf16OutBuf;

        pipe.InitBuffer(ubInt32Buf,    myElems * sizeof(int32_t));
        pipe.InitBuffer(ubFp32AccBuf,  myElems * sizeof(float));
        pipe.InitBuffer(ubFp32TempBuf, myElems * sizeof(float));
        pipe.InitBuffer(ubRowScaleBuf, myM * 8 * sizeof(float));
        pipe.InitBuffer(ubScaleBuf,    2048);
        pipe.InitBuffer(ubBf16OutBuf,  myElems * sizeof(bfloat16_t));

        auto ubInt32     = ubInt32Buf.Get<int32_t>();
        auto ubFp32Acc   = ubFp32AccBuf.Get<float>();
        auto ubFp32Temp  = ubFp32TempBuf.Get<float>();
        auto ubRowScale  = ubRowScaleBuf.Get<float>();
        auto ubScaleFp32 = ubScaleBuf.Get<float>();
        auto ubScaleBf16 = ubScaleBuf.Get<bfloat16_t>();
        auto ubBf16Out   = ubBf16OutBuf.Get<bfloat16_t>();

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

            for (uint32_t chunk = 0; chunk < CHUNKS_PER_AIV; chunk++) {
                Duplicate(ubFp32Acc[chunk * CHUNK_ELEMS], 0.0f, CHUNK_ELEMS);
            }

            DataCopy(ubScaleBf16, wScaleGm[nOff], TILE_N);
            // Load the whole tile's a_scale, not just this AIV's half: at
            // TILE_M=16 a half is 8 bf16 = 16 B, under the 32 B DataCopy floor.
            // mBase is a multiple of 8 floats, so the Brcb source stays 32 B aligned.
            DataCopy(ubScaleBf16[384], aScaleGm[mOff], TILE_M);
            SetFlag<HardEvent::MTE2_V>(evMte2V0);
            WaitFlag<HardEvent::MTE2_V>(evMte2V0);
            Cast(ubScaleFp32[64],  ubScaleBf16[0],   RoundMode::CAST_NONE, TILE_N);
            Cast(ubScaleFp32[256], ubScaleBf16[384], RoundMode::CAST_NONE, TILE_M);
            Brcb(ubRowScale, ubScaleFp32[256 + mBase], myM / 8, {1, 8});

            CrossCoreWaitFlag(kAicToAivFlag);

            const uint64_t wsRowOff = (uint64_t)mBase * TILE_N;
            DataCopy(ubInt32, wsGm[wsRowOff], CHUNK_ELEMS);
            SetFlag<HardEvent::MTE2_V>(evMte2V0);
            if constexpr (CHUNKS_PER_AIV > 1) {
                DataCopy(ubInt32[CHUNK_ELEMS], wsGm[wsRowOff + CHUNK_ELEMS], CHUNK_ELEMS);
                SetFlag<HardEvent::MTE2_V>(evMte2V1);
            }
            CrossCoreSetFlag<kCrossMode, PIPE_MTE2>(kAivToAicFlag);

            WaitFlag<HardEvent::MTE2_V>(evMte2V0);
            Cast(ubFp32Temp, ubInt32, RoundMode::CAST_NONE, CHUNK_ELEMS);
            ApplyScaledChunk(ubFp32Acc, ubFp32Temp, ubRowScale, ubScaleFp32, 0);

            if constexpr (CHUNKS_PER_AIV > 1) {
                WaitFlag<HardEvent::MTE2_V>(evMte2V1);
                Cast(ubFp32Temp[CHUNK_ELEMS],
                     ubInt32[CHUNK_ELEMS], RoundMode::CAST_NONE, CHUNK_ELEMS);
                ApplyScaledChunk(ubFp32Acc, ubFp32Temp, ubRowScale, ubScaleFp32, 1);
            }

            DataCopyParams outDcParams;
            outDcParams.blockCount = static_cast<uint16_t>(CHUNK_M);
            outDcParams.blockLen   = static_cast<uint16_t>(TILE_N * sizeof(uint16_t) / 32);
            outDcParams.srcStride  = 0;
            outDcParams.dstStride  = static_cast<uint16_t>((N - TILE_N) * sizeof(uint16_t) / 32);

            Cast(ubBf16Out, ubFp32Acc, RoundMode::CAST_ROUND, CHUNK_ELEMS);
            SetFlag<HardEvent::V_MTE3>(evVMte3);
            WaitFlag<HardEvent::V_MTE3>(evVMte3);
            DataCopy(outGm[(mOff + mBase) * N + nOff], ubBf16Out, outDcParams);

            if constexpr (CHUNKS_PER_AIV > 1) {
                Cast(ubBf16Out[CHUNK_ELEMS],
                     ubFp32Acc[CHUNK_ELEMS], RoundMode::CAST_ROUND, CHUNK_ELEMS);
                SetFlag<HardEvent::V_MTE3>(evVMte3);
                WaitFlag<HardEvent::V_MTE3>(evVMte3);
                DataCopy(outGm[(mOff + mBase + CHUNK_M) * N + nOff],
                         ubBf16Out[CHUNK_ELEMS], outDcParams);
            }

            // Write-after-read barriers before the next tile reuses these UBs:
            //   MTE3_V  — the output DataCopy still reads ubBf16Out, which the
            //             next tile's Cast overwrites.
            //   V_MTE2  — the Cast still reads ubInt32, which the next tile's
            //             workspace DataCopy overwrites.
            // Both hazards are latent at TILE_M=128 (two chunks and a slow cube
            // keep the window shut) and become reproducible corruption at
            // TILE_M=16, where the cube laps the vector.
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
