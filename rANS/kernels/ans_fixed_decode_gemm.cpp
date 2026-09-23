/*
 * MIX kernel: AIV fixed-rate BF16 decode + Cube GEMM.
 *
 * Replaces the AIV half of ans_decode_gemm.cpp. That one runs a per-symbol rANS
 * decode, measured at ~1.6 cycles/weight on 910B4 -- four times the budget set
 * by simply reading BF16 from HBM. This decode is shifts and ors only and
 * measures at 0.44 (tools/ans/vgather_bench_lab, tools/ans/ans_fixed_decode_lab).
 * The Cube side and the cross-core handshake are unchanged.
 *
 * Being fixed-rate removes three things the rANS path needs before it can
 * decode anything: the per-tile byte offsets, the per-tile symbol counts, and
 * the shared codebook. Tile t's planes sit at t * (TILE_WEIGHTS/2),
 * t * (TILE_WEIGHTS/4), ... -- pure arithmetic, no indirection.
 *
 * Work split: AIV subIdx takes alternate k-tiles, NOT alternate row halves.
 * A block's four quarters all live in the same E-plane words, so splitting a
 * tile by rows would have both AIVs read the whole E plane -- 8 bits/weight of
 * payload traffic instead of 4, cancelling the compression outright.
 *
 * Layout and encoder: tools/ans/fixed_codec.py (encode_fixed_gemm). One Cube
 * tile is exactly one encoder block, which is what makes the fixed offsets work.
 *
 * STATUS: verified on hardware, and MEASURABLY SLOWER THAN NOT COMPRESSING.
 * Both facts matter; do not read the first without the second.
 *
 * Correctness. ProcessTilesOnly is bit-exact on 14 shapes x 3 runs, including
 * 4096x64 where every group is wide. The fused path matches torch.matmul on
 * the same decoded weights on 10 shapes: bit-identical on most, and where it
 * differs the difference is one bf16 ULP of a dot product that cancels -- an
 * fp64 reference puts |ours - truth| == |torch - truth| on every such element.
 *
 * Speed. msprof device time, 5120x5120, M=1, eight weight sets rotated so
 * nothing is L2-resident:
 *
 *      torch bf16 matmul        44.8 us   (1156 GB/s -- at peak)
 *      decode only              134.7 us  3.01x   vec 106.1 us, vec ratio 81%
 *      fused decode + GEMM      151.7 us  3.39x   vec 106.1 us, vec ratio 71%
 *
 * Read that last column first: the VECTOR TIME IS THE SAME 106.1 us in both.
 * The Cube, the cross-core handshake, the panel flush, the wide pass and the
 * MTE3 store are all already hidden underneath it, and each was confirmed by
 * switching it off and re-measuring (dbg in the tiling head). So 106.1 us,
 * i.e. 2.37x, is this design's floor, and the only thing that moves it is
 * issuing fewer vector instructions.
 *
 * What was measured and does NOT help, so that nobody pays for it twice:
 *
 *   NSLOT 2 -> 4 (deeper AIV/Cube ping-pong)   150.7 vs 151.3 us, 2x workspace
 *   NBUF_IN 2 -> 4 (deeper plane prefetch)     +0.8 us, +25 KB UB
 *   interleaving two quarters on separate      no change; these loops are
 *     scratch to break the dependency chain    issue-bound, not latency-bound
 *   shortening the quarter ops 2048 -> 512     no change: ~98% of a vector
 *                                              op's cost is independent of its
 *                                              length, so batching tiles into
 *                                              longer vectors buys nothing
 *   Level 2 -> Level 0 ops with the vector     ~10%: kept, it is also what
 *     mask hoisted out of the loop             shrinks the masks to 256 B
 *
 * What DID help: NBUF 1 -> 2. Before it, MTE2 / V / MTE3 took strict turns;
 * after it msprof reports mte2 21% and mte3 17% against vec 71%.
 *
 * And on the host side, caching the Cube tiling per shape (it was rebuilt and
 * copied H2D on every call) took wall-clock time from ~311 us to ~165 us. That
 * was worth more than every kernel change here put together -- always check
 * device time against wall time before optimising a kernel.
 *
 * The round trip through GM is unavoidable -- Cube and Vector cooperate only
 * through GM on 910B -- so the fused path moves 12.33 + 16 + 16 = 44.3 bits per
 * weight against 16 for simply reading bf16.
 *
 * So this kernel buys WEIGHT CAPACITY (27.5 -> 21.2 GB on Qwen3-14B) at about
 * 3.4x the latency of streaming bf16. It is not a speed optimisation.
 *
 * The cross-core handshake is settled from CANN's own sources rather than
 * guessed. A mode-2 flag is COUNTED over both AIVs of a pair:
 * sparse_flash_attention_grad's VecCompute does the work under
 * `if (subBlockIdx == 0)` but calls CrossCoreSetFlag<2, PIPE_MTE3> from BOTH
 * subblocks, against a single CrossCoreWaitFlag on the Cube -- the redundant
 * set exists precisely because the Cube's wait needs two. So both AIVs setting
 * flagReady for one panel is correct, as long as they always set it the same
 * number of times, which they do: both take the same trip count through the
 * ni loop. In the other direction one AIC set releases both AIVs' waits, from
 * the same op's ping-pong loop.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

namespace {
using namespace AscendC;

constexpr int32_t TILE_N0 = 64;
constexpr int32_t TILE_K0 = 128;
constexpr int32_t TILE_WEIGHTS = TILE_N0 * TILE_K0;              // 8192 = one encoder block
constexpr int32_t QUARTER = TILE_WEIGHTS / 4;                    // 2048, the vector length
constexpr int32_t GROUP = 16;
constexpr int32_t SUPER = 256;
constexpr int32_t GROUPS_PER_TILE = TILE_WEIGHTS / GROUP;        // 512
constexpr int32_t SUPERS_PER_TILE = TILE_WEIGHTS / SUPER;        // 32
constexpr int32_t DQUARTER = GROUPS_PER_TILE / 4;                // 128
constexpr int32_t BRCB_PER_REPEAT = 8;
constexpr uint16_t MASK_OFF = 0x000F;
constexpr uint16_t MASK_BYTE = 0x00FF;
constexpr uint16_t MASK_SIGN = 0x8000;
constexpr uint16_t MASK_MANT = 0x007F;

constexpr int32_t TILING_HEAD_INTS = 16;
constexpr int32_t CACHE_LINE_BYTES = 512;
// Depth of the per-tile software pipeline, so MTE2 / V / MTE3 run at the same
// time instead of taking turns. Going from 1 to 2 is what matters; msprof then
// reports mte2 21% / mte3 17% against vec 72%, i.e. both memory directions sit
// underneath the vector work. NBUF_IN = 4 was measured and changed the kernel
// by 0.8 us, so the extra 25 KB of UB is not bought.
constexpr int32_t NBUF_IN = 2;    // plane buffers, MTE2 -> V
constexpr int32_t NBUF_OUT = 2;   // out_ buffers,   V -> MTE3
// One vector repeat is 256 B = 128 uint16 lanes; QUARTER is 16 of them.
constexpr int32_t LANES = 128;
constexpr uint8_t QREPS = QUARTER / LANES;            // 16
constexpr uint8_t BLK_PER_REP = 8;                    // 256 B / 32 B
constexpr uint32_t SINGLE_CORE_M = 128;
constexpr uint32_t SINGLE_CORE_N = 64;
constexpr uint32_t SINGLE_CORE_K = 20480;
constexpr uint32_t BASIC_M = 128;
constexpr uint32_t BASIC_N = 64;
constexpr uint32_t BASIC_K = 128;
constexpr uint32_t STEP_M = 1;
constexpr uint32_t STEP_N = 1;
constexpr uint32_t STEP_KA = 4;
constexpr uint32_t STEP_KB = 4;
constexpr uint32_t DEPTH_A1 = 8;
constexpr uint32_t DEPTH_B1 = 8;
// Cross-core flag ids. Only 0..7 are free: catlass reserves 8/9/10 for its
// inter-block and inter-subblock barriers (FFTS_MAX_FLAG == 7), and the CANN
// runtime itself owns 11..14 (SYNC_AIC_FLAG, SYNC_AIV_FLAG, SYNC_AIC_AIV_FLAG,
// SYNC_AIV_ONLY_ALL in dav_c220/kernel_operator_sync_impl.h). The old default
// of 8 put flagDone on 10 and 11 -- straight through two of them.
constexpr uint16_t FLAG_AIV_READY = 0;   // ready: 0..NSLOT-1, done: NSLOT..2*NSLOT-1
// Panels in flight between the AIV pair and its Cube. Two is enough: NSLOT = 4
// was measured at 150.7 us against 151.3 and doubled the workspace, so the Cube
// is NOT what the AIV waits on. (The 16 us by which the fused path exceeds the
// tiles-only path survives stubbing the GEMM entirely -- see the header.)
constexpr int32_t NSLOT = 2;

using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

constexpr MatmulConfig ANS_MM_MDL = GetMDLConfig(false, false, 0, false, false, false, true, true, true, false, true);

constexpr MatmulConfig GetAnsMMCFG()
{
    MatmulConfig cfg = ANS_MM_MDL;
    cfg.singleCoreM = SINGLE_CORE_M;
    cfg.singleCoreN = SINGLE_CORE_N;
    cfg.singleCoreK = SINGLE_CORE_K;
    cfg.basicM = BASIC_M;
    cfg.basicN = BASIC_N;
    cfg.basicK = BASIC_K;
    return cfg;
}

constexpr MatmulApiStaticTiling GetAnsMMTiling(const MatmulApiStaticTiling &src)
{
    MatmulApiStaticTiling t = src;
    t.stepM = STEP_M;
    t.stepN = STEP_N;
    t.stepKa = STEP_KA;
    t.stepKb = STEP_KB;
    t.depthA1 = DEPTH_A1;
    t.depthB1 = DEPTH_B1;
    t.isBias = false;
    return t;
}

static constexpr MatmulConfig ANS_MM_CFG = GetAnsMMCFG();
static constexpr MatmulApiStaticTiling ANS_MM_TILING =
    GetAnsMMTiling(GetMatmulApiTiling<AType, BType, CType, BiasType>(ANS_MM_CFG));
using AnsMM = matmul::MatmulImpl<AType, BType, CType, BiasType, ANS_MM_TILING>;

__aicore__ inline void LoadCubeTiling(TCubeTiling &dst, __gm__ uint8_t *src)
{
    auto *s = reinterpret_cast<__gm__ uint32_t *>(src);
    auto *d = reinterpret_cast<uint32_t *>(&dst);
    constexpr int n = static_cast<int>(sizeof(TCubeTiling) / sizeof(uint32_t));
    for (int i = 0; i < n; ++i) {
        d[i] = s[i];
    }
}

class AnsFixedDecodeGemm {
public:
    __aicore__ inline AnsFixedDecodeGemm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR lplane, GM_ADDR eplane, GM_ADDR dplane, GM_ADDR sbase,
                                __gm__ int32_t *wideGroup, GM_ADDR wideL, GM_ADDR wideExp,
                                __gm__ int32_t *wideTileOff, GM_ADDR y, GM_ADDR workspace,
                                __gm__ int32_t *tiling)
    {
        int32_t head[TILING_HEAD_INTS];
        auto *src = reinterpret_cast<__gm__ uint32_t *>(tiling);
        auto *dst = reinterpret_cast<uint32_t *>(head);
        for (int i = 0; i < TILING_HEAD_INTS; ++i) {
            dst[i] = src[i];
        }
        m_ = static_cast<uint32_t>(head[0]);
        nPad_ = static_cast<uint32_t>(head[1]);
        kPad_ = static_cast<uint32_t>(head[2]);
        tilesN_ = head[3];
        tilesK_ = head[4];
        flagReady_ = static_cast<uint16_t>(head[7] > 0 ? head[7] : FLAG_AIV_READY);
        // head[8] is a profiling knob, 0 in every production path. Bit 0 makes
        // the Cube run the handshake without the GEMM, bit 1 drops the panel
        // flush. Together they attribute the fused time to AIV / Cube / flush.
        dbg_ = head[8];
        flagDone_ = static_cast<uint16_t>(flagReady_ + NSLOT);
        tileElems_ = TILE_N0 * static_cast<int32_t>(kPad_);
        mmTilingBytes_ = reinterpret_cast<__gm__ uint8_t *>(tiling) +
                         TILING_HEAD_INTS * static_cast<int32_t>(sizeof(int32_t));

        xPtr_ = reinterpret_cast<__gm__ bfloat16_t *>(x);
        yPtr_ = reinterpret_cast<__gm__ bfloat16_t *>(y);
        lGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(lplane));
        eGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(eplane));
        dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(dplane));
        sGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(sbase));
        wlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(wideL));
        weGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(wideExp));
        wgGm_.SetGlobalBuffer(wideGroup);
        wtGm_.SetGlobalBuffer(wideTileOff);

        if (workspace != nullptr) {          // null in the tiles-only path
            SetSysWorkspaceForce(workspace);
            int32_t sysBytes = head[5] > 0 ? head[5] : 0;
            wsPtr_ = reinterpret_cast<__gm__ uint16_t *>(workspace + sysBytes);
        }

        cubeIdx_ = GetBlockIdx();
        subIdx_ = 0;
        cubeNum_ = static_cast<uint32_t>(head[6] > 0 ? head[6] : 24);
        if ASCEND_IS_AIV {
            cubeIdx_ = GetBlockIdx() / 2;
            subIdx_ = GetBlockIdx() % 2;
        }
    }

    __aicore__ inline void Process()
    {
        if (cubeIdx_ >= cubeNum_ || tilesN_ <= 0 || tilesK_ <= 0) {
            return;
        }
        if ASCEND_IS_AIV {
            ProcessAiv();
            return;
        }
        if ASCEND_IS_AIC {
            ProcessAic();
        }
    }

    /* Verification entry point: decode every tile straight into a full
     * [n_pad, k_pad] output, no Cube and no cross-core flags. It runs the SAME
     * DecodeTile / PatchWideGroups / plane addressing / ScatterTile code as the
     * fused path, so checking this bit-exact pins down everything about the
     * decode half and leaves only the handshake unverified. */
    __aicore__ inline void ProcessTilesOnly(GM_ADDR out)
    {
        if (tilesN_ <= 0 || tilesK_ <= 0) {
            return;
        }
        if ASCEND_IS_AIC {
            return;
        }
        auto *outPtr = reinterpret_cast<__gm__ uint16_t *>(out);
        // This file contains MatmulImpl, so it builds as a MIX binary: the
        // launch blockDim counts AIC blocks, and each spawns two AIVs whose
        // GetBlockIdx() is the global AIV index. Launching with the AIV count
        // instead fails at registration with 107000.
        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        const int32_t blockNum = static_cast<int32_t>(cubeNum_) * 2;
        InitUb();
        BrcbRepeatParams brcb(1, 8);
        const int32_t nTiles = tilesN_ * tilesK_;
        if (blockIdx >= nTiles) {
            return;
        }
        const int32_t mine = (nTiles - blockIdx + blockNum - 1) / blockNum;

        // Software pipeline, NBUF deep: the loads for tile i+1 are issued
        // before tile i is decoded, and tile i's store is left in flight until
        // the vector pipe comes back round to that buffer.
        for (int32_t i = 0; i + 1 < NBUF_IN && i < mine; ++i) {
            LoadTile(blockIdx + i * blockNum, i % NBUF_IN, false);
        }
        for (int32_t i = 0; i < mine; ++i) {
            const int32_t pin = i % NBUF_IN;
            const int32_t pout = i % NBUF_OUT;
            const int32_t t = blockIdx + i * blockNum;
            const int32_t j = i + NBUF_IN - 1;
            if (j < mine) {
                LoadTile(blockIdx + j * blockNum, j % NBUF_IN, j >= NBUF_IN);
            }
            if (i >= NBUF_OUT) {
                WaitFlag<HardEvent::MTE3_V>(pout);
            }
            DecodeLoaded(t, pin, pout, brcb);
            const int32_t ni = t / tilesK_;
            ScatterTile(pout, outPtr + static_cast<int64_t>(ni) * TILE_N0 * kPad_, t - ni * tilesK_);
        }
        DrainTiles(mine);
    }

private:
    // Slot accessors for the NBUF-deep buffers. Offsets, not casts -- see the
    // note in DecodeLoaded about ReinterpretCast losing an offset.
    __aicore__ inline LocalTensor<uint16_t> Out(int32_t o) { return out_[o * TILE_WEIGHTS]; }
    __aicore__ inline LocalTensor<uint16_t> Ew(int32_t p) { return ew_[p * QUARTER]; }
    __aicore__ inline LocalTensor<uint16_t> Lw(int32_t p) { return lw_[p * 2 * QUARTER]; }
    __aicore__ inline LocalTensor<uint16_t> Dw(int32_t p) { return dw_[p * DQUARTER]; }
    __aicore__ inline LocalTensor<uint16_t> Sb(int32_t p) { return sb_[p * SUPERS_PER_TILE]; }

    __aicore__ inline __gm__ uint16_t *TilePtr(int32_t slot)
    {
        return wsPtr_ + (static_cast<int64_t>(cubeIdx_) * NSLOT + slot) * tileElems_;
    }

    __aicore__ inline void InitUb()
    {
        // out_ is read by MTE3 and the plane buffers are written by MTE2, so
        // those are the ones that need a second copy. Everything below them is
        // touched only by the vector pipe, which is in-order with itself.
        pipe_.InitBuffer(outBuf_, NBUF_OUT * TILE_WEIGHTS * sizeof(uint16_t));  // 32 KB
        pipe_.InitBuffer(eBuf, NBUF_IN * QUARTER * sizeof(uint16_t));          // 16 KB
        pipe_.InitBuffer(lBuf, NBUF_IN * 2 * QUARTER * sizeof(uint16_t));      // 32 KB
        pipe_.InitBuffer(dBuf, NBUF_IN * DQUARTER * sizeof(uint16_t));
        pipe_.InitBuffer(sBuf, NBUF_IN * SUPERS_PER_TILE * sizeof(uint16_t));
        pipe_.InitBuffer(supBuf, GROUPS_PER_TILE * sizeof(uint16_t));
        pipe_.InitBuffer(gbBuf, GROUPS_PER_TILE * sizeof(uint16_t));
        pipe_.InitBuffer(expBuf, TILE_WEIGHTS * sizeof(uint16_t));    // 16 KB
        pipe_.InitBuffer(offBuf, QUARTER * sizeof(uint16_t));
        pipe_.InitBuffer(tmpBuf, QUARTER * sizeof(uint16_t));
        pipe_.InitBuffer(sgnBuf, QUARTER * sizeof(uint16_t));
        pipe_.InitBuffer(manBuf, QUARTER * sizeof(uint16_t));
        pipe_.InitBuffer(resBuf, QUARTER * sizeof(uint16_t));
        pipe_.InitBuffer(wideBuf, GROUP * sizeof(uint16_t) * 4);
        // The masks are read with src1RepStride == 0, so one repeat's worth is
        // reused by every repeat -- 256 B each instead of one per element.
        pipe_.InitBuffer(mOffBuf, LANES * sizeof(uint16_t));
        pipe_.InitBuffer(mByteBuf, LANES * sizeof(uint16_t));
        pipe_.InitBuffer(mSignBuf, LANES * sizeof(uint16_t));
        pipe_.InitBuffer(mMantBuf, LANES * sizeof(uint16_t));

        out_ = outBuf_.Get<uint16_t>();
        ew_ = eBuf.Get<uint16_t>();
        lw_ = lBuf.Get<uint16_t>();
        dw_ = dBuf.Get<uint16_t>();
        sb_ = sBuf.Get<uint16_t>();
        sup_ = supBuf.Get<uint16_t>();
        gb_ = gbBuf.Get<uint16_t>();
        ex_ = expBuf.Get<uint16_t>();
        off_ = offBuf.Get<uint16_t>();
        tmp_ = tmpBuf.Get<uint16_t>();
        sgn_ = sgnBuf.Get<uint16_t>();
        man_ = manBuf.Get<uint16_t>();
        res_ = resBuf.Get<uint16_t>();
        wide_ = wideBuf.Get<uint16_t>();
        mOff_ = mOffBuf.Get<uint16_t>();
        mByte_ = mByteBuf.Get<uint16_t>();
        mSign_ = mSignBuf.Get<uint16_t>();
        mMant_ = mMantBuf.Get<uint16_t>();

        // AscendC has no scalar-operand And, so masks live in tensors.
        Duplicate(mOff_.ReinterpretCast<int16_t>(), int16_t(MASK_OFF), LANES);
        Duplicate(mByte_.ReinterpretCast<int16_t>(), int16_t(MASK_BYTE), LANES);
        Duplicate(mSign_.ReinterpretCast<int16_t>(), int16_t(MASK_SIGN), LANES);
        Duplicate(mMant_.ReinterpretCast<int16_t>(), int16_t(MASK_MANT), LANES);
        PipeBarrier<PIPE_V>();

        // The count-based (Level 2) API re-emits the vector mask setup on every
        // single call, and that setup -- not the arithmetic -- is what this
        // kernel was spending its time on: shortening the quarter ops from 2048
        // elements to 512 changed the runtime by under 1%. Set the mask once
        // here and issue Level 0 ops with isSetMask = false.
        SetMaskNorm();
        SetVectorMask<int16_t, MaskMode::NORMAL>(uint64_t(-1), uint64_t(-1));
    }

    __aicore__ inline void ProcessAiv()
    {
        InitUb();
        BrcbRepeatParams brcb(1, 8);

        int32_t slot = 0;
        int32_t produced = 0;
        for (int32_t ni = static_cast<int32_t>(cubeIdx_); ni < tilesN_; ni += static_cast<int32_t>(cubeNum_)) {
            if (produced >= NSLOT) {
                CrossCoreWaitFlag(static_cast<uint16_t>(flagDone_ + slot));
            }
            // Alternate k-tiles per AIV. The trip counts differ by one when
            // tilesK_ is odd, which does not matter: what the handshake needs
            // is that both AIVs reach the CrossCoreSetFlag below the same
            // number of times, and that is the ni loop's trip count, which is
            // identical because both share cubeIdx_.
            //
            // Pipelined NBUF deep, same as ProcessTilesOnly. The pipeline is
            // drained at the panel boundary because the Cube is about to read
            // the whole panel, so it cannot span panels -- with ~40 k-tiles per
            // panel the two tiles of fill and drain cost ~5%.
            const int32_t mine = (tilesK_ - static_cast<int32_t>(subIdx_) + 1) / 2;
            const int32_t base = ni * tilesK_ + static_cast<int32_t>(subIdx_);
            for (int32_t i = 0; i + 1 < NBUF_IN && i < mine; ++i) {
                LoadTile(base + 2 * i, i % NBUF_IN, false);
            }
            for (int32_t i = 0; i < mine; ++i) {
                const int32_t pin = i % NBUF_IN;
                const int32_t pout = i % NBUF_OUT;
                const int32_t j = i + NBUF_IN - 1;
                if (j < mine) {
                    LoadTile(base + 2 * j, j % NBUF_IN, j >= NBUF_IN);
                }
                if (i >= NBUF_OUT) {
                    WaitFlag<HardEvent::MTE3_V>(pout);
                }
                DecodeLoaded(base + 2 * i, pin, pout, brcb);
                ScatterTile(pout, TilePtr(slot), static_cast<int32_t>(subIdx_) + 2 * i);
            }
            DrainTiles(mine);
            if ((dbg_ & 0x2) == 0) {
                FlushPanel(TilePtr(slot));
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(static_cast<uint16_t>(flagReady_ + slot));
            produced += 1;
            slot = (slot + 1) % NSLOT;
        }
        // Drain whatever is still in flight. In-loop waits consumed
        // produced - NSLOT of the Cube's done flags; the rest are here, and the
        // totals have to match or the next launch inherits a set flag.
        for (int32_t i = 0; i < NSLOT && i < produced; ++i) {
            CrossCoreWaitFlag(static_cast<uint16_t>(flagDone_ + (produced - 1 - i) % NSLOT));
        }
    }

    __aicore__ inline void ProcessAic()
    {
        TCubeTiling cubeTiling;
        LoadCubeTiling(cubeTiling, mmTilingBytes_);
        AnsMM mm;
        mm.SetSubBlockIdx(0);
        mm.Init(&cubeTiling, &pipe_);

        int32_t slot = 0;
        for (int32_t ni = static_cast<int32_t>(cubeIdx_); ni < tilesN_; ni += static_cast<int32_t>(cubeNum_)) {
            CrossCoreWaitFlag(static_cast<uint16_t>(flagReady_ + slot));
            if ((dbg_ & 0x1) == 0) {
                GemmPanel(mm, ni, TilePtr(slot));
            }
            CrossCoreSetFlag<0x2, PIPE_FIX>(static_cast<uint16_t>(flagDone_ + slot));
            slot = (slot + 1) % NSLOT;
        }
        mm.End();
    }

    /* Issue the plane loads for tile t into plane-buffer slot p. Nothing waits
     * on the data here -- the matching WaitFlag<MTE2_V> is in DecodeLoaded, so
     * these loads run underneath the previous tile's vector work. `reuse` is
     * true once slot p has been used before and the vector pipe has to be done
     * with it first; that wait sits on the MTE2 queue, not the vector one. */
    __aicore__ inline void LoadTile(int32_t t, int32_t p, bool reuse)
    {
        if (reuse) {
            WaitFlag<HardEvent::V_MTE2>(p);
        }
        DataCopy(Ew(p), eGm_[static_cast<int64_t>(t) * QUARTER], QUARTER);
        DataCopy(Lw(p), lGm_[static_cast<int64_t>(t) * 2 * QUARTER], 2 * QUARTER);
        DataCopy(Dw(p), dGm_[static_cast<int64_t>(t) * DQUARTER], DQUARTER);
        DataCopy(Sb(p), sGm_[static_cast<int64_t>(t) * SUPERS_PER_TILE], SUPERS_PER_TILE);
        SetFlag<HardEvent::MTE2_V>(p);
    }

    /* Decode the tile already loaded in plane-buffer p into out-buffer o. Same
     * op sequence as csrc/kernels/ans_fixed_decode.cpp, which is verified
     * bit-exact; only the plane addressing differs, and here it is a plain
     * multiply by the tile index because the code is fixed-rate. */
    __aicore__ inline void DecodeLoaded(int32_t t, int32_t p, int32_t o, const BrcbRepeatParams &brcb)
    {
        WaitFlag<HardEvent::MTE2_V>(p);

        LocalTensor<uint16_t> ew = Ew(p);
        LocalTensor<uint16_t> lw = Lw(p);
        LocalTensor<uint16_t> out = Out(o);

        // Two-level base: 32 super bases -> 512 group bases -> 8192 per-weight
        // bases, in two Brcb instructions (each broadcasts one element into a
        // 32-byte block, which for uint16 is exactly GROUP lanes).
        Brcb(sup_, Sb(p), uint8_t(SUPERS_PER_TILE / BRCB_PER_REPEAT), brcb);
        PipeBarrier<PIPE_V>();
        for (int32_t r = 0; r < 4; ++r) {
            ShiftRight(gb_[r * DQUARTER], Dw(p), uint16_t(4 * r), DQUARTER);
            And(gb_[r * DQUARTER], gb_[r * DQUARTER], mOff_, DQUARTER);
        }
        PipeBarrier<PIPE_V>();
        Add(gb_.ReinterpretCast<int16_t>(), gb_.ReinterpretCast<int16_t>(),
            sup_.ReinterpretCast<int16_t>(), GROUPS_PER_TILE);
        PipeBarrier<PIPE_V>();
        Brcb(ex_, gb_, uint8_t(GROUPS_PER_TILE / BRCB_PER_REPEAT), brcb);
        PipeBarrier<PIPE_V>();
        ShiftLeft(ex_, ex_, uint16_t(7), TILE_WEIGHTS);
        PipeBarrier<PIPE_V>();

        // Everything accumulates in res_, which starts at offset 0. Writing
        // straight into out[r * QUARTER] looks tidier but needs
        // ReinterpretCast on an OFFSET tensor for the int16 Add, and that
        // silently loses the offset -- every quarter lands on top of quarter 0.
        // The standalone decoder never hit this because its result buffer was
        // always at offset 0.
        // Two quarters interleaved. Each step issues chain A then chain B on
        // disjoint scratch, so the vector pipe always has an independent
        // instruction to slot in behind a dependent one.
        // Level 0, mask already set, isSetMask = false: no per-call mask setup.
        // Back to one chain -- interleaving two on separate scratch bought
        // nothing (these loops are issue-bound, not latency-bound), so the
        // extra 20 KB of UB was not paying for itself.
        constexpr uint64_t M0 = 0;  // ignored when isSetMask is false
        const UnaryRepeatParams up(1, 1, BLK_PER_REP, BLK_PER_REP);
        const BinaryRepeatParams bp(1, 1, 1, BLK_PER_REP, BLK_PER_REP, BLK_PER_REP);
        // src1RepStride 0: every repeat re-reads the same 128-lane mask block.
        const BinaryRepeatParams mp(1, 1, 1, BLK_PER_REP, BLK_PER_REP, 0);

        for (int32_t r = 0; r < 4; ++r) {
            LocalTensor<int16_t> resI = res_.ReinterpretCast<int16_t>();
            ShiftRight<uint16_t, false>(tmp_, ew, uint16_t(4 * r), M0, QREPS, up);
            And<uint16_t, false>(off_, tmp_, mOff_, M0, QREPS, mp);
            ShiftLeft<uint16_t, false>(off_, off_, uint16_t(7), M0, QREPS, up);
            Add<int16_t, false>(resI, off_.ReinterpretCast<int16_t>(),
                                ex_[r * QUARTER].ReinterpretCast<int16_t>(), M0, QREPS, bp);
            ShiftRight<uint16_t, false>(tmp_, lw[(r / 2) * QUARTER], uint16_t(8 * (r & 1)), M0, QREPS, up);
            And<uint16_t, false>(tmp_, tmp_, mByte_, M0, QREPS, mp);
            ShiftLeft<uint16_t, false>(sgn_, tmp_, uint16_t(8), M0, QREPS, up);
            And<uint16_t, false>(sgn_, sgn_, mSign_, M0, QREPS, mp);
            And<uint16_t, false>(man_, tmp_, mMant_, M0, QREPS, mp);
            Or<uint16_t, false>(res_, res_, sgn_, M0, QREPS, bp);
            // Land the final value straight in out. Copying res_ across with
            // DataCopy instead puts a UB->UB transfer on an MTE pipe, which
            // PipeBarrier<PIPE_V> does not order against the next quarter's
            // vector writes to res_ -- quarters 2 and 3 came back partly stale.
            Or<uint16_t, false>(out[r * QUARTER], res_, man_, M0, QREPS, bp);
        }
        PipeBarrier<PIPE_V>();
        // The planes are free now; releasing them here rather than after the
        // wide pass lets the next tile's loads start that much earlier.
        SetFlag<HardEvent::V_MTE2>(p);
        PatchWideGroups(t, o);
    }

    /* Groups whose exponent spread did not fit 4 bits. Unlike the standalone
     * decoder these are patched in UB before the tile ever reaches GM, so there
     * is no cross-core ordering to get wrong -- the tile's UB is private to one
     * AIV. wideTileOff gives this tile's slice of the (group-sorted) wide list. */
    __aicore__ inline void PatchWideGroups(int32_t t, int32_t o)
    {
        if ((dbg_ & 0x4) != 0) {     // profiling only -- produces wrong output
            return;
        }
        const int32_t begin = wtGm_.GetValue(t);
        const int32_t end = wtGm_.GetValue(t + 1);
        if (end <= begin) {
            return;
        }
        LocalTensor<uint16_t> wl = wide_;
        LocalTensor<uint16_t> we = wide_[GROUP];
        LocalTensor<uint16_t> ws = wide_[GROUP * 2];
        LocalTensor<uint16_t> wm = wide_[GROUP * 3];
        for (int32_t k = begin; k < end; ++k) {
            const int32_t local = wgGm_.GetValue(k) - t * GROUPS_PER_TILE;
            DataCopy(wl, wlGm_[static_cast<int64_t>(k) * GROUP], GROUP);
            DataCopy(we, weGm_[static_cast<int64_t>(k) * GROUP], GROUP);
            SetFlag<HardEvent::MTE2_V>(EVT_WIDE);
            WaitFlag<HardEvent::MTE2_V>(EVT_WIDE);
            LocalTensor<uint16_t> dstw = Out(o)[local * GROUP];
            ShiftLeft(dstw, we, uint16_t(7), GROUP);
            ShiftLeft(ws, wl, uint16_t(8), GROUP);
            And(ws, ws, mSign_, GROUP);
            And(wm, wl, mMant_, GROUP);
            Or(dstw, dstw, ws, GROUP);
            Or(dstw, dstw, wm, GROUP);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVT_WIDE);
            WaitFlag<HardEvent::V_MTE2>(EVT_WIDE);
        }
    }

    /* Store out-buffer o. The MTE3_V that releases the buffer is NOT waited on
     * here -- it is consumed two tiles later, just before the vector pipe
     * writes this buffer again. That is the whole point of NBUF: waiting for
     * the store right after issuing it is what made the vector pipe idle
     * through every DataCopy. */
    __aicore__ inline void ScatterTile(int32_t o, __gm__ uint16_t *panel, int32_t kj)
    {
        GlobalTensor<uint16_t> dst;
        dst.SetGlobalBuffer(panel + static_cast<int64_t>(kj) * TILE_K0,
                            static_cast<uint32_t>(TILE_N0) * kPad_);
        DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(TILE_N0);
        params.blockLen = static_cast<uint16_t>((TILE_K0 * static_cast<int32_t>(sizeof(uint16_t))) / 32);
        params.srcStride = 0;
        params.dstStride = static_cast<uint16_t>(
            ((static_cast<int32_t>(kPad_) - TILE_K0) * static_cast<int32_t>(sizeof(uint16_t))) / 32);
        // out was written by the vector pipe and will be rewritten by it.
        // PipeBarrier only orders a pipe against ITSELF, so neither hazard is
        // covered by one -- MTE3 read out_ while V was still filling quarters 2
        // and 3, which is what made this kernel return garbage that changed
        // from run to run.
        SetFlag<HardEvent::V_MTE3>(o);
        WaitFlag<HardEvent::V_MTE3>(o);
        if ((dbg_ & 0x10) != 0) {
            // Profiling only -- WRONG output. No store at all.
        } else if ((dbg_ & 0x8) != 0) {
            // Profiling only -- WRONG output. One contiguous 16 KB burst
            // instead of 64 strided 256 B ones, to price the scatter itself.
            GlobalTensor<uint16_t> flat;
            flat.SetGlobalBuffer(panel + static_cast<int64_t>(kj) * TILE_WEIGHTS, TILE_WEIGHTS);
            DataCopy(flat, Out(o), TILE_WEIGHTS);
        } else {
            DataCopy(dst, Out(o), params);
        }
        SetFlag<HardEvent::MTE3_V>(o);
    }

    /* Run the NBUF-deep pipeline dry: one MTE3_V and one V_MTE2 are still
     * outstanding on each buffer that was used. Both must be consumed before
     * the panel is handed to the Cube, and before the counts carry into the
     * next panel. */
    __aicore__ inline void DrainTiles(int32_t nTiles)
    {
        for (int32_t b = 0; b < NBUF_OUT && b < nTiles; ++b) {
            WaitFlag<HardEvent::MTE3_V>(b);
        }
        for (int32_t b = 0; b < NBUF_IN && b < nTiles; ++b) {
            WaitFlag<HardEvent::V_MTE2>(b);
        }
    }

    /* MTE3 stores land in this core's cache. CrossCoreSetFlag<.., PIPE_MTE3>
     * orders the signal after the stores retire but does NOT push them out to
     * where the Cube reads, and the Cube reads GM through a different port.
     * Skipping this gives sparse wrong values whose count changes between runs
     * -- and a slow producer hides it, so it tends to show up only after the
     * decode gets faster. Each AIV flushes the whole panel range: the lines it
     * did not write are not dirty here, so those are no-ops.
     *
     * Cost is one instruction per 512 B of panel, i.e. ~0.004 per weight. */
    __aicore__ inline void FlushPanel(__gm__ uint16_t *panel)
    {
        SetFlag<HardEvent::MTE3_S>(EVT_OUT);
        WaitFlag<HardEvent::MTE3_S>(EVT_OUT);
        GlobalTensor<uint16_t> g;
        g.SetGlobalBuffer(panel, static_cast<uint32_t>(tileElems_));
        constexpr int32_t LINE_U16 = CACHE_LINE_BYTES / static_cast<int32_t>(sizeof(uint16_t));
        for (int32_t off = 0; off < tileElems_; off += LINE_U16) {
            DataCacheCleanAndInvalid<uint16_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g[off]);
        }
    }

    __aicore__ inline void GemmPanel(AnsMM &mm, int32_t ni, __gm__ uint16_t *panel)
    {
        GlobalTensor<bfloat16_t> gmA, gmB, gmC;
        gmA.SetGlobalBuffer(xPtr_);
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(panel));
        gmC.SetGlobalBuffer(yPtr_);
        mm.SetOrgShape(static_cast<int>(m_), static_cast<int>(nPad_), static_cast<int>(kPad_));
        for (uint32_t m0 = 0; m0 < m_; m0 += SINGLE_CORE_M) {
            uint32_t curM = m_ - m0;
            if (curM > SINGLE_CORE_M) {
                curM = SINGLE_CORE_M;
            }
            mm.SetSingleShape(curM, static_cast<uint32_t>(TILE_N0), kPad_);
            mm.SetTensorA(gmA[static_cast<int64_t>(m0) * kPad_]);
            mm.SetTensorB(gmB, true);
            mm.template IterateAll<true>(
                gmC[static_cast<int64_t>(m0) * nPad_ + static_cast<int64_t>(ni) * TILE_N0], 0);
        }
    }

    // One id per buffer handoff. Every Set is immediately followed by its Wait,
    // so nothing is ever in flight on two ids at once, but keeping them distinct
    // makes the pairing checkable by reading.
    static constexpr int32_t EVT_IN = 0;    // MTE2 <-> V on the plane buffers
    static constexpr int32_t EVT_OUT = 1;   // V <-> MTE3 on out_
    static constexpr int32_t EVT_WIDE = NBUF_IN;  // MTE2 <-> V on wide_, clear of the plane ids

    TPipe pipe_;
    TBuf<TPosition::VECCALC> outBuf_, eBuf, lBuf, dBuf, sBuf, supBuf, gbBuf, expBuf;
    TBuf<TPosition::VECCALC> offBuf, tmpBuf, sgnBuf, manBuf, resBuf, wideBuf;
    TBuf<TPosition::VECCALC> mOffBuf, mByteBuf, mSignBuf, mMantBuf;
    LocalTensor<uint16_t> out_, ew_, lw_, dw_, sb_, sup_, gb_, ex_;
    LocalTensor<uint16_t> off_, tmp_, sgn_, man_, res_, wide_;
    LocalTensor<uint16_t> mOff_, mByte_, mSign_, mMant_;
    GlobalTensor<uint16_t> lGm_, eGm_, dGm_, sGm_, wlGm_, weGm_;
    GlobalTensor<int32_t> wgGm_, wtGm_;
    __gm__ bfloat16_t *xPtr_{nullptr};
    __gm__ bfloat16_t *yPtr_{nullptr};
    __gm__ uint16_t *wsPtr_{nullptr};
    __gm__ uint8_t *mmTilingBytes_{nullptr};
    uint32_t m_{0}, nPad_{0}, kPad_{0}, cubeIdx_{0}, subIdx_{0}, cubeNum_{0};
    int32_t tilesN_{0}, tilesK_{0}, tileElems_{0};
    int32_t dbg_{0};
    uint16_t flagReady_{FLAG_AIV_READY};
    uint16_t flagDone_{FLAG_AIV_READY + 2};
};

}  // namespace

extern "C" __global__ __aicore__
void ans_fixed_decode_gemm_kernel(GM_ADDR x, GM_ADDR lplane, GM_ADDR eplane, GM_ADDR dplane, GM_ADDR sbase,
                                  GM_ADDR wideGroup, GM_ADDR wideL, GM_ADDR wideExp, GM_ADDR wideTileOff,
                                  GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AnsFixedDecodeGemm op;
    op.Init(x, lplane, eplane, dplane, sbase, reinterpret_cast<__gm__ int32_t *>(wideGroup), wideL, wideExp,
            reinterpret_cast<__gm__ int32_t *>(wideTileOff), y, workspace,
            reinterpret_cast<__gm__ int32_t *>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__
void ans_fixed_decode_tiles_kernel(GM_ADDR lplane, GM_ADDR eplane, GM_ADDR dplane, GM_ADDR sbase,
                                   GM_ADDR wideGroup, GM_ADDR wideL, GM_ADDR wideExp, GM_ADDR wideTileOff,
                                   GM_ADDR out, GM_ADDR tiling)
{
    AnsFixedDecodeGemm op;
    op.Init(nullptr, lplane, eplane, dplane, sbase, reinterpret_cast<__gm__ int32_t *>(wideGroup), wideL,
            wideExp, reinterpret_cast<__gm__ int32_t *>(wideTileOff), nullptr, nullptr,
            reinterpret_cast<__gm__ int32_t *>(tiling));
    op.ProcessTilesOnly(out);
}

namespace vllm_ascend {

void ans_fixed_decode_tiles_impl(void *stream, void *lplane, void *eplane, void *dplane, void *sbase,
                                 void *wide_group, void *wide_l, void *wide_exp, void *wide_tile_off, void *out,
                                 void *tiling, uint32_t cube_num)
{
    // blockDim is the AIC block count for a MIX binary, not the AIV count.
    ans_fixed_decode_tiles_kernel<<<cube_num, nullptr, stream>>>(
        lplane, eplane, dplane, sbase, wide_group, wide_l, wide_exp, wide_tile_off, out, tiling);
}

void ans_fixed_decode_gemm_impl(void *stream, void *x, void *lplane, void *eplane, void *dplane, void *sbase,
                                void *wide_group, void *wide_l, void *wide_exp, void *wide_tile_off, void *y,
                                void *workspace, void *tiling, uint32_t cube_num)
{
    ans_fixed_decode_gemm_kernel<<<cube_num, nullptr, stream>>>(
        x, lplane, eplane, dplane, sbase, wide_group, wide_l, wide_exp, wide_tile_off, y, workspace, tiling);
}

}  // namespace vllm_ascend
