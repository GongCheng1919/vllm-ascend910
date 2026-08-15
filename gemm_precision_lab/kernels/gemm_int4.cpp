// gemm_int4.cpp — INT4 x INT4 -> INT32 GEMM (C = A @ B^T), 纯 mad_s4 无反量化。
// 结构同 gemm_int8（手写流水）。int4 打包: GM 数据为 M*K/2 字节（每字节 2 元素）。
// 处理: Nd2Nz/L1/L0 全部按 int8 字节视图（K/2 字节 = K/2 个 int8），
// 仅 Mmad 用 ReinterpretCast<int4b_t> + k=INNER_K(int4 元素) 走 mad_s4。
// 字节布局一致性: int8 视图的 K0=32 字节 = 64 个 int4 元素 = int4 的 K0。
// 结构复刻自 mid_group_gemm_fwd_lab/kernels/perchannel_int8_gemm.cpp（去掉 scale/dequant）：
//   * L1 depth-2 ping-pong (TQue<A1/B1, 2>)   — GM->L1 藏在 L1->L0 + Mmad 后
//   * L0 depth-2 ping-pong (TQue<A2/B2, 2>)   — LoadData 藏在 Mmad 后
//   * PHYSICAL_K=1024 组内 4-slice 软件流水    — INNER_K=256, 单 L0C 跨 K 累加
//   * 每次输出 tile 一次 Fixpipe + 一次 CrossCore 同步
//   * tile 流: tileId = blockIdx; tileId < numTiles; tileId += blockDim
//   * 跨 tile g=0 GM->L1 预取
// AIV 侧: 从 workspace 读 int32 直接写回 GM (无 scale 应用)。
//
// A: [M, K] int8 row-major; B: [N, K] int8 row-major; C: [M, N] int32。
// M/N 必须是 128 的倍数, K 必须是 1024 的倍数。

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_type.h"
#include "basic_api/kernel_operator_fixpipe_intf.h"

using namespace AscendC;

namespace {

constexpr uint32_t TILE_M        = 128;
constexpr uint32_t TILE_N        = 128;
constexpr uint32_t HALF_M        = TILE_M / 2;
constexpr uint32_t TILE_ELEMS    = TILE_M * TILE_N;
constexpr uint32_t HALF_ELEMS    = HALF_M * TILE_N;
constexpr uint32_t CHUNK_M       = 32;
constexpr uint32_t CHUNK_ELEMS   = CHUNK_M * TILE_N;
constexpr uint32_t CHUNKS_PER_AIV = HALF_M / CHUNK_M;

constexpr uint32_t kBlock        = 16;
constexpr uint32_t kK0Int8       = 32;
constexpr uint32_t PHYSICAL_K    = 1024;
constexpr uint32_t INNER_K       = 256;
constexpr uint32_t INNER_PER_G   = PHYSICAL_K / INNER_K;

constexpr uint32_t kMBlocks      = TILE_M / kBlock;
constexpr uint32_t kNBlocks      = TILE_N / kBlock;
// INNER_K/PHYSICAL_K 是 int4 元素数；kK0Int8=32 是字节块大小(=64 个 int4 元素/块)。
// 字节数 = INNER_K/2，块数 = 字节数/kK0Int8 —— 直接照抄 int8 公式（未除以2）会导致块数
// 多算一倍，LoadData 的 repeatTimes/偏移全部错位，是 int4 kernel "结果全错" 的根因。
constexpr uint32_t kKBlocksInner = INNER_K / (2 * kK0Int8);
constexpr uint32_t kNzBlockSize  = kK0Int8 * kBlock;
constexpr uint32_t kInnerStrideL1Bytes = (INNER_K / 2) * TILE_M;  // int8 字节视图

constexpr uint16_t kAivToAicFlag = 3;
constexpr uint16_t kAicToAivFlag = 5;
constexpr uint8_t  kCrossMode    = 2;

__aicore__ inline void LoadAicTileL1ToL0(const LocalTensor<int8_t>& aL0,
                                         const LocalTensor<int8_t>& bL0,
                                         const LocalTensor<int8_t>& aL1,
                                         const LocalTensor<int8_t>& bL1,
                                         const LoadData2dParams& ldA,
                                         const LoadData2dParams& ldB)
{
    // L1->L0 必须以 int4b_t 类型调用 LoadData，才会派发到 load_cbuf_to_ca_s4/
    // load_cbuf_to_cb_s4 硬件指令（AscendC 对 int4b_t 走单独分支，int8_t 类型
    // 会走普通 load_cbuf_to_ca/cb，产出的 L0 布局与 mad_s4 期望的不一致）。
    // Nd2Nz(GM->L1) 阶段两者共用同一条 b8 指令，故 aL1/aL0 的字节偏移算法不变，
    // 仅在此处（真正调用 LoadData 前）reinterpret 成 int4b_t。
    for (uint32_t mi = 0; mi < kMBlocks; mi++) {
        LoadData(aL0[mi * kKBlocksInner * kNzBlockSize].ReinterpretCast<int4b_t>(),
                 aL1[mi * kNzBlockSize].ReinterpretCast<int4b_t>(), ldA);
    }
    LoadData(bL0.ReinterpretCast<int4b_t>(), bL1.ReinterpretCast<int4b_t>(), ldB);
}

}  // namespace

extern "C" __global__ __aicore__
void gemm_int4(GM_ADDR a_q, GM_ADDR b_q,
               GM_ADDR y,   GM_ADDR workspace,
               uint32_t M, uint32_t N, uint32_t K)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    const uint32_t blockIdx   = GetBlockIdx() / GetTaskRation();
    const uint32_t blockDim   = GetBlockNum();
    const uint32_t numMTiles  = M / TILE_M;
    const uint32_t numNTiles  = N / TILE_N;
    const uint32_t numTiles   = numMTiles * numNTiles;
    const uint32_t physicalG  = K / PHYSICAL_K;
    if (GetBlockIdx() == 0) {
        printf("sizeof(int4b_t)=%d\n", (int)sizeof(int4b_t));
    }

    GlobalTensor<int32_t> wsGm;
    wsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(workspace) + (uint64_t)blockIdx * TILE_ELEMS,
        TILE_ELEMS);

    // ============================ AIC ============================
    if ASCEND_IS_AIC {
        GlobalTensor<int8_t> aGm, bGm;
        // int4 打包: 字节视图元素数 = M*K/2
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a_q), (uint64_t)M * K / 2);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(b_q), (uint64_t)N * K / 2);

        TQue<QuePosition::A1, 2> a1Queue;
        TQue<QuePosition::B1, 2> b1Queue;
        TQue<QuePosition::A2, 2> a2Queue;
        TQue<QuePosition::B2, 2> b2Queue;
        TQue<QuePosition::CO1, 1> c1Queue;
        pipe.InitBuffer(a1Queue, 2, TILE_M * PHYSICAL_K / 2);       // int4 字节视图: 2 x 64 KB L1
        pipe.InitBuffer(b1Queue, 2, TILE_N * PHYSICAL_K / 2);
        pipe.InitBuffer(a2Queue, 2, TILE_M * INNER_K / 2);          // 2 x 16 KB L0A (int8 字节视图)
        pipe.InitBuffer(b2Queue, 2, TILE_N * INNER_K / 2);          // 2 x 16 KB L0B
        pipe.InitBuffer(c1Queue, 1, TILE_ELEMS * sizeof(int32_t));  // 1 x 64 KB L0C

        Nd2NzParams ndA;
        ndA.ndNum             = 1;
        ndA.nValue            = TILE_M;
        ndA.dValue            = PHYSICAL_K / 2;  // int4: 字节视图
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue         = K / 2;
        ndA.dstNzC0Stride     = TILE_M;
        ndA.dstNzNStride      = 1;
        ndA.dstNzMatrixStride = 0;

        Nd2NzParams ndB;
        ndB.ndNum             = 1;
        ndB.nValue            = TILE_N;
        ndB.dValue            = PHYSICAL_K / 2;
        ndB.srcNdMatrixStride = 0;
        ndB.srcDValue         = K / 2;
        ndB.dstNzC0Stride     = TILE_N;
        ndB.dstNzNStride      = 1;
        ndB.dstNzMatrixStride = 0;

        LoadData2dParams ldA;
        ldA.startIndex  = 0;
        ldA.repeatTimes = static_cast<uint8_t>(kKBlocksInner);      // 与 int8 版一致
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
        fp.mSize       = TILE_M;
        fp.srcStride   = TILE_M;
        fp.dstStride   = TILE_N;
        fp.quantPre    = QuantMode_t::NoQuant;
        fp.ndNum       = 1;
        fp.srcNdStride = 1;
        fp.dstNdStride = 1;

        MmadParams mm;
        mm.m              = TILE_M;
        mm.n              = TILE_N;
        mm.k              = INNER_K;  // int4 元素数 (256)
        mm.cmatrixSource  = false;

        bool firstTile = true;
        bool tileG0Prefetched = false;
        for (uint32_t tileId = blockIdx; tileId < numTiles; tileId += blockDim) {
            const uint32_t mTile = tileId / numNTiles;
            const uint32_t nTile = tileId % numNTiles;
            const uint64_t mOff  = (uint64_t)mTile * TILE_M;
            const uint64_t nOff  = (uint64_t)nTile * TILE_N;
            // aGm/bGm 是字节视图 (int4 打包, 2 元素/字节)，行偏移须除以 2。
            const uint64_t aTileBase = mOff * K / 2;
            const uint64_t bTileBase = nOff * K / 2;

            if (!tileG0Prefetched) {
                auto aL1Prime = a1Queue.AllocTensor<int8_t>();
                auto bL1Prime = b1Queue.AllocTensor<int8_t>();
                DataCopy(aL1Prime, aGm[aTileBase], ndA);
                DataCopy(bL1Prime, bGm[bTileBase], ndB);
                a1Queue.EnQue(aL1Prime);
                b1Queue.EnQue(bL1Prime);
            }
            tileG0Prefetched = false;

            auto cL0 = c1Queue.AllocTensor<int32_t>();
            bool initC = true;

            for (uint32_t g = 0; g < physicalG; g++) {
                if (g + 1 < physicalG) {
                    const uint64_t nextGKOffset = (uint64_t)(g + 1) * PHYSICAL_K / 2;
                    auto aL1Next = a1Queue.AllocTensor<int8_t>();
                    auto bL1Next = b1Queue.AllocTensor<int8_t>();
                    DataCopy(aL1Next, aGm[aTileBase + nextGKOffset], ndA);
                    DataCopy(bL1Next, bGm[bTileBase + nextGKOffset], ndB);
                    a1Queue.EnQue(aL1Next);
                    b1Queue.EnQue(bL1Next);
                }

                auto aL1 = a1Queue.DeQue<int8_t>();
                auto bL1 = b1Queue.DeQue<int8_t>();

                // 4-slice software pipeline accumulating into cL0.
                auto aL0_0 = a2Queue.AllocTensor<int8_t>();
                auto bL0_0 = b2Queue.AllocTensor<int8_t>();
                LoadAicTileL1ToL0(aL0_0, bL0_0, aL1, bL1, ldA, ldB);
                a2Queue.EnQue(aL0_0);
                b2Queue.EnQue(bL0_0);

                auto aL0_1 = a2Queue.AllocTensor<int8_t>();
                auto bL0_1 = b2Queue.AllocTensor<int8_t>();
                LoadAicTileL1ToL0(aL0_1, bL0_1,
                                  aL1[kInnerStrideL1Bytes],
                                  bL1[kInnerStrideL1Bytes], ldA, ldB);
                a2Queue.EnQue(aL0_1);
                b2Queue.EnQue(bL0_1);

                aL0_0 = a2Queue.DeQue<int8_t>();
                bL0_0 = b2Queue.DeQue<int8_t>();
                mm.cmatrixInitVal = initC;
                initC = false;
                Mmad(cL0, aL0_0.ReinterpretCast<int4b_t>(), bL0_0.ReinterpretCast<int4b_t>(), mm);
                a2Queue.FreeTensor(aL0_0);
                b2Queue.FreeTensor(bL0_0);

                aL0_0 = a2Queue.AllocTensor<int8_t>();
                bL0_0 = b2Queue.AllocTensor<int8_t>();
                LoadAicTileL1ToL0(aL0_0, bL0_0,
                                  aL1[2 * kInnerStrideL1Bytes],
                                  bL1[2 * kInnerStrideL1Bytes], ldA, ldB);
                a2Queue.EnQue(aL0_0);
                b2Queue.EnQue(bL0_0);

                aL0_1 = a2Queue.DeQue<int8_t>();
                bL0_1 = b2Queue.DeQue<int8_t>();
                mm.cmatrixInitVal = false;
                Mmad(cL0, aL0_1.ReinterpretCast<int4b_t>(), bL0_1.ReinterpretCast<int4b_t>(), mm);
                a2Queue.FreeTensor(aL0_1);
                b2Queue.FreeTensor(bL0_1);

                aL0_1 = a2Queue.AllocTensor<int8_t>();
                bL0_1 = b2Queue.AllocTensor<int8_t>();
                LoadAicTileL1ToL0(aL0_1, bL0_1,
                                  aL1[3 * kInnerStrideL1Bytes],
                                  bL1[3 * kInnerStrideL1Bytes], ldA, ldB);
                a2Queue.EnQue(aL0_1);
                b2Queue.EnQue(bL0_1);

                aL0_0 = a2Queue.DeQue<int8_t>();
                bL0_0 = b2Queue.DeQue<int8_t>();
                Mmad(cL0, aL0_0.ReinterpretCast<int4b_t>(), bL0_0.ReinterpretCast<int4b_t>(), mm);
                a2Queue.FreeTensor(aL0_0);
                b2Queue.FreeTensor(bL0_0);

                aL0_1 = a2Queue.DeQue<int8_t>();
                bL0_1 = b2Queue.DeQue<int8_t>();
                a1Queue.FreeTensor(aL1);
                b1Queue.FreeTensor(bL1);
                Mmad(cL0, aL0_1.ReinterpretCast<int4b_t>(), bL0_1.ReinterpretCast<int4b_t>(), mm);
                a2Queue.FreeTensor(aL0_1);
                b2Queue.FreeTensor(bL0_1);
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
                const uint64_t nextAStart = (uint64_t)nextMTile * TILE_M * K / 2;
                const uint64_t nextBStart = (uint64_t)nextNTile * TILE_N * K / 2;
                auto aL1Next = a1Queue.AllocTensor<int8_t>();
                auto bL1Next = b1Queue.AllocTensor<int8_t>();
                DataCopy(aL1Next, aGm[nextAStart], ndA);
                DataCopy(bL1Next, bGm[nextBStart], ndB);
                a1Queue.EnQue(aL1Next);
                b1Queue.EnQue(bL1Next);
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
        const uint32_t mBase = subIdx * HALF_M;

        GlobalTensor<int32_t> outGm;
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y), (uint64_t)M * N);

        TBuf<QuePosition::VECCALC> ubInt32Buf;
        TBuf<QuePosition::VECCALC> ubOutBuf;
        pipe.InitBuffer(ubInt32Buf, HALF_ELEMS * sizeof(int32_t));
        pipe.InitBuffer(ubOutBuf, HALF_ELEMS * sizeof(int32_t));

        auto ubInt32 = ubInt32Buf.Get<int32_t>();
        auto ubOut = ubOutBuf.Get<int32_t>();

        event_t evMte2V0 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        event_t evMte2V1 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        event_t evVMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());

        DataCopyParams outDcParams;
        outDcParams.blockCount = static_cast<uint16_t>(CHUNK_M);
        outDcParams.blockLen   = static_cast<uint16_t>(TILE_N * sizeof(int32_t) / 32);
        outDcParams.srcStride  = 0;
        outDcParams.dstStride  = static_cast<uint16_t>((N - TILE_N) * sizeof(int32_t) / 32);

        for (uint32_t tileId = blockIdx; tileId < numTiles; tileId += blockDim) {
            const uint32_t mTile = tileId / numNTiles;
            const uint32_t nTile = tileId % numNTiles;
            const uint64_t mOff  = (uint64_t)mTile * TILE_M;
            const uint64_t nOff  = (uint64_t)nTile * TILE_N;

            CrossCoreWaitFlag(kAicToAivFlag);

            const uint64_t wsRowOff = (uint64_t)mBase * TILE_N;
            DataCopy(ubInt32, wsGm[wsRowOff], CHUNK_ELEMS);
            SetFlag<HardEvent::MTE2_V>(evMte2V0);
            DataCopy(ubInt32[CHUNK_ELEMS], wsGm[wsRowOff + CHUNK_ELEMS], CHUNK_ELEMS);
            SetFlag<HardEvent::MTE2_V>(evMte2V1);
            CrossCoreSetFlag<kCrossMode, PIPE_MTE2>(kAivToAicFlag);

            WaitFlag<HardEvent::MTE2_V>(evMte2V0);
            Adds(ubOut, ubInt32, 0, CHUNK_ELEMS);
            SetFlag<HardEvent::V_MTE3>(evVMte3);
            WaitFlag<HardEvent::V_MTE3>(evVMte3);
            const uint64_t outOff0 = (mOff + mBase) * N + nOff;
            DataCopy(outGm[outOff0], ubOut, outDcParams);

            WaitFlag<HardEvent::MTE2_V>(evMte2V1);
            Adds(ubOut[CHUNK_ELEMS], ubInt32[CHUNK_ELEMS], 0, CHUNK_ELEMS);
            SetFlag<HardEvent::V_MTE3>(evVMte3);
            WaitFlag<HardEvent::V_MTE3>(evVMte3);
            const uint64_t outOff1 = (mOff + mBase + CHUNK_M) * N + nOff;
            DataCopy(outGm[outOff1], ubOut[CHUNK_ELEMS], outDcParams);
        }

        pipe.ReleaseEventID<HardEvent::MTE2_V>(evMte2V0);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(evMte2V1);
        pipe.ReleaseEventID<HardEvent::V_MTE3>(evVMte3);

        event_t evMte3S = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_S>());
        SetFlag<HardEvent::MTE3_S>(evMte3S);
        WaitFlag<HardEvent::MTE3_S>(evMte3S);
        pipe.ReleaseEventID<HardEvent::MTE3_S>(evMte3S);
    }

    pipe.Destroy();
}
