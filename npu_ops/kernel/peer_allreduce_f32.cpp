// peer_allreduce -- cross-rank all-reduce with the barrier INSIDE the kernel.
// Two entry points: `peer_allreduce_f32` and `peer_allreduce_bf16`.
//
// WHY (P6.5 D16 -> D17).  `peer_read_sum_f32` proved a self-written kernel can read
// peer HBM and be captured into an NPUGraph, but it relied on the CALLER doing
// `dist.barrier()` between ranks.  A captured graph cannot contain a host-side
// collective, so a real fused GEMM + all-reduce has to synchronise the ranks from
// inside the kernel.  That is the only thing separating the D16 probe from an op
// that could sit in a vLLM decode graph.
//
// PROTOCOL (put-nothing / read-everything, so a rank only ever WRITES its own
// buffer and only ever READS its peers'):
//
//   1. seq   = ++private_counter           (in this rank's own control area)
//   2. slot  = seq & 1                     (double buffering, see below)
//   3. copy x -> own data[slot]
//   4. flush the written lines out of cache so peers can see them
//   5. publish seq into own control word, flush it
//   6. spin until every peer's published seq >= seq
//   7. read every peer's data[slot], sum -> out
//
// WHY TWO SLOTS ARE ENOUGH.  A rank can only reach iteration seq+2 after passing
// the barrier for seq+1, which requires every peer to have published seq+1, which a
// peer does only after it finished READING everyone's seq data (step 7 of its
// previous iteration precedes step 5 of its next).  So by the time slot (seq&1) is
// overwritten, every peer is done with it.  A single slot would race; three would
// buy nothing.
//
// WHY THE SEQUENCE NUMBER LIVES IN DEVICE MEMORY.  Under graph capture the kernel
// arguments are frozen, so a host-supplied counter would replay the same value
// forever and the barrier would pass immediately on every replay after the first.
// The counter is therefore bumped by the kernel itself.
//
// MULTI-CORE (D17, second pass).  The single-core version measured 27 us at M=1
// against HCCL's 250 us, but it is bandwidth-limited at ~5.8 GB/s and therefore
// LOSES from M=16 up -- one AIV core cannot carry a 1.3 MB reduction.  The fix
// needs no intra-rank synchronisation at all, because the slicing is made symmetric
// across ranks:
//
//   * every rank splits the SAME `n` into the SAME slices (the host derives the
//     block count from `n` alone, so all ranks agree without communicating);
//   * block b touches only slice b -- of its own buffer and of every peer's;
//   * the flag is PER BLOCK, so block b waits on the peers' block b only.
//
// So the barrier is really `usedBlocks` independent barriers running in parallel,
// and a block never has to know what the other blocks on its own rank are doing.
// The alternative -- one rank-wide flag plus a cross-block sync to decide who
// raises it -- would have needed an intra-rank barrier on top of the inter-rank
// one, which is the thing worth avoiding.
//
// Slices are cut on 512 B boundaries so that two blocks never share a cache line;
// they flush lines to make their writes visible to peers, and a shared line would
// let one block's flush race another block's staging.
//
// BF16 (D17, third pass).  bf16 is the format that matters: the mid-group GEMM
// emits bf16 (`midgroup_w4a8_gemm.cpp`), and so does everything else in the vLLM
// residual stream, so an f32-only op would have forced a cast on both sides of
// every call.  It is also strictly better on the two axes we care about:
//
//   * HALF THE BYTES cross the fabric, which is where the large-M loss came from;
//   * the reduction still ACCUMULATES IN F32 -- values are widened in UB, summed
//     in f32, and rounded once on the way out.  A bf16-accumulating reduction
//     would round after every partial sum, so this is more accurate than the
//     collective it replaces, not less.
//
// The staging layout is sized in f32 units for both dtypes, so a buffer mapped
// once serves either.

#include "kernel_operator.h"
#include "basic_api/kernel_operator_cache_intf.h"

using namespace AscendC;

namespace {
// Staging layout, in float/int32 elements from the buffer base.  The host copy of
// these constants lives in host/peer_allreduce.cpp and MUST agree.
//
//   [0   .. 320)   published seq, one 32 B slot per block -- POLLED BY PEERS
//   [512 .. 832)   private counters, one per block -- touched only by this rank
//   [1024 ..  )    data, two slots of `cap` FLOAT-sized elements
//
// Flags are 32 B apart so two blocks never share a polled line, and the two areas
// are far apart so a peer polling flags never pulls a private counter with them.
constexpr uint32_t kMaxBlocks  = 40;   // AIV cores on 910B4
constexpr uint32_t kFlagStride = 8;    // int32 elements between flags (32 B)
constexpr uint32_t kPubBase    = 0;
constexpr uint32_t kPrivBase   = 512;
constexpr uint32_t kDataBase   = 1024;      // in float elements
constexpr uint32_t kDataByte   = kDataBase * 4;
constexpr uint32_t kSlotByte   = 4;         // slot stride unit: sizeof(float)

constexpr uint32_t kChunk = 4096;   // elements per pass (16 KB of f32 accumulator)
constexpr uint32_t kCtl   = 8;      // int32 elements in one 32 B control block

__aicore__ inline uint32_t CeilTo(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

// Read one int32 from GM, defeating the cache.  Used for the spin, so the
// invalidate is the entire point: without it the poll reads a stale line forever.
__aicore__ inline int32_t ReadFresh(__gm__ int32_t *addr, LocalTensor<int32_t> &tmp)
{
    GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(addr, kCtl);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(g);
    DataCopy(tmp, g, kCtl);
    PipeBarrier<PIPE_ALL>();
    return tmp.GetValue(0);
}

// Publish one int32 to GM and push it out of cache so peers observe it.
__aicore__ inline void WriteFlush(__gm__ int32_t *addr, int32_t v,
                                  LocalTensor<int32_t> &tmp)
{
    GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(addr, kCtl);
    for (uint32_t i = 0; i < kCtl; ++i) tmp.SetValue(i, v);
    PipeBarrier<PIPE_ALL>();
    DataCopy(g, tmp, kCtl);
    PipeBarrier<PIPE_ALL>();
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(g);
    PipeBarrier<PIPE_ALL>();
}

// One body for both dtypes.  T is the WIRE type (what crosses the fabric); the
// accumulator is always f32.
template <typename T>
__aicore__ inline void PeerAllReduceBody(
    GM_ADDR p0, GM_ADDR p1, GM_ADDR p2, GM_ADDR p3,
    GM_ADDR p4, GM_ADDR p5, GM_ADDR p6, GM_ADDR p7,
    GM_ADDR selfAddr, GM_ADDR xAddr, GM_ADDR outAddr,
    uint32_t n, uint32_t world, uint32_t cap, uint32_t spinLimit,
    uint32_t sliceElems)
{
    constexpr uint32_t kAlign = 32 / sizeof(T);   // elements per 32 B DataCopy block

    // The host sizes the launch so that every block gets work, but a block that
    // lands past the end must still return WITHOUT publishing: its peers' matching
    // blocks are not running either, so there is nobody to wait for and nobody
    // waiting.  Never leave a published flag behind for a slice nobody reduces.
    const uint32_t blk = GetBlockIdx();
    const uint32_t base = blk * sliceElems;
    if (blk >= kMaxBlocks || base >= n) return;
    const uint32_t len = (n - base < sliceElems) ? (n - base) : sliceElems;

    TPipe pipe;
    TBuf<TPosition::VECCALC> accBuf, valBuf, rawBuf, ctlBuf;
    pipe.InitBuffer(accBuf, kChunk * sizeof(float));
    pipe.InitBuffer(valBuf, kChunk * sizeof(float));
    pipe.InitBuffer(rawBuf, kChunk * sizeof(T));
    pipe.InitBuffer(ctlBuf, kCtl * sizeof(int32_t));
    LocalTensor<float>   acc = accBuf.Get<float>();
    LocalTensor<float>   val = valBuf.Get<float>();
    LocalTensor<T>       raw = rawBuf.Get<T>();
    LocalTensor<int32_t> ctl = ctlBuf.Get<int32_t>();

    const uint32_t flagOff = blk * kFlagStride;
    __gm__ int32_t *selfPub  = (__gm__ int32_t *)selfAddr + kPubBase + flagOff;
    __gm__ int32_t *selfPriv = (__gm__ int32_t *)selfAddr + kPrivBase + flagOff;

    // (1) bump this block's private counter -- device-side, so replays advance it
    const int32_t seq = ReadFresh(selfPriv, ctl) + 1;
    WriteFlush(selfPriv, seq, ctl);
    const uint32_t slot = (uint32_t)(seq & 1);

    // Slot offsets are computed in BYTES from a f32-sized layout, so one mapped
    // staging buffer serves either dtype.
    const uint32_t slotByteOff = kDataByte + slot * cap * kSlotByte;
    __gm__ T *selfSlot = (__gm__ T *)(selfAddr + slotByteOff) + base;

    // (2) stage this block's slice into its own slot (no conversion: the wire
    // format is the input format)
    GlobalTensor<T> xG, slotG;
    for (uint32_t off = 0; off < len; off += kChunk) {
        const uint32_t cnt  = (len - off < kChunk) ? (len - off) : kChunk;
        const uint32_t cntA = CeilTo(cnt, kAlign);
        xG.SetGlobalBuffer((__gm__ T *)xAddr + base + off, cntA);
        slotG.SetGlobalBuffer(selfSlot + off, cntA);
        DataCopy(raw, xG, cntA);
        PipeBarrier<PIPE_ALL>();
        DataCopy(slotG, raw, cntA);
        PipeBarrier<PIPE_ALL>();
    }

    // (3) push the staged data out of cache BEFORE publishing the flag.  Order
    // matters: a peer that sees the flag must be able to see the data, so the
    // flush of the data has to complete first.
    for (uint32_t off = 0; off < len; off += kAlign) {
        GlobalTensor<T> line;
        line.SetGlobalBuffer(selfSlot + off, kAlign);
        DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(line);
    }
    PipeBarrier<PIPE_ALL>();

    // (4) publish, then wait for everyone's MATCHING BLOCK.  `spinLimit` exists so
    // a desynchronised run fails instead of wedging the card -- a hung kernel on
    // this box needs the orphan-process cleanup, which is far worse than a wrong
    // answer.
    WriteFlush(selfPub, seq, ctl);

#define MG_WAIT_PEER(IDX, PTR)                                                    \
    if ((IDX) < world && (PTR) != selfAddr) {                                     \
        __gm__ int32_t *pub = (__gm__ int32_t *)(PTR) + kPubBase + flagOff;        \
        uint32_t spins = 0;                                                       \
        while (ReadFresh(pub, ctl) < seq) {                                       \
            if (++spins > spinLimit) break;                                       \
        }                                                                         \
    }
    MG_WAIT_PEER(0u, p0)
    MG_WAIT_PEER(1u, p1)
    MG_WAIT_PEER(2u, p2)
    MG_WAIT_PEER(3u, p3)
    MG_WAIT_PEER(4u, p4)
    MG_WAIT_PEER(5u, p5)
    MG_WAIT_PEER(6u, p6)
    MG_WAIT_PEER(7u, p7)
#undef MG_WAIT_PEER

    // (5) everyone's slice is ready: read and sum IN F32.  Pointers are consumed
    // UNROLLED, never through a runtime-indexed array -- that silently collapses to
    // index 0 and cost a debugging round in D16.
    for (uint32_t off = 0; off < len; off += kChunk) {
        const uint32_t cnt  = (len - off < kChunk) ? (len - off) : kChunk;
        const uint32_t cntA = CeilTo(cnt, kAlign);

#define MG_ACC_PEER(IDX, PTR)                                                     \
        if ((IDX) < world) {                                                      \
            GlobalTensor<T> src;                                                  \
            src.SetGlobalBuffer(                                                  \
                (__gm__ T *)((PTR) + slotByteOff) + base + off, cntA);            \
            DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE,             \
                                     DcciDst::CACHELINE_OUT>(src);                \
            DataCopy(raw, src, cntA);                                             \
            PipeBarrier<PIPE_ALL>();                                              \
            if constexpr (sizeof(T) == sizeof(float)) {                           \
                if ((IDX) == 0) { Adds(acc, raw, (T)0, cntA); }                   \
                else            { Add(acc, acc, raw, cntA); }                     \
            } else {                                                              \
                Cast(val, raw, RoundMode::CAST_NONE, cntA);                       \
                PipeBarrier<PIPE_ALL>();                                          \
                if ((IDX) == 0) { Adds(acc, val, 0.0f, cntA); }                   \
                else            { Add(acc, acc, val, cntA); }                     \
            }                                                                     \
            PipeBarrier<PIPE_ALL>();                                              \
        }
        MG_ACC_PEER(0u, p0)
        MG_ACC_PEER(1u, p1)
        MG_ACC_PEER(2u, p2)
        MG_ACC_PEER(3u, p3)
        MG_ACC_PEER(4u, p4)
        MG_ACC_PEER(5u, p5)
        MG_ACC_PEER(6u, p6)
        MG_ACC_PEER(7u, p7)
#undef MG_ACC_PEER

        GlobalTensor<T> outG;
        outG.SetGlobalBuffer((__gm__ T *)outAddr + base + off, cntA);
        // One rounding, on the way out: CAST_RINT is round-to-nearest-even, which
        // is what torch's f32 -> bf16 does.
        if constexpr (sizeof(T) == sizeof(float)) {
            DataCopy(outG, acc, cntA);
        } else {
            Cast(raw, acc, RoundMode::CAST_RINT, cntA);
            PipeBarrier<PIPE_ALL>();
            DataCopy(outG, raw, cntA);
        }
        PipeBarrier<PIPE_ALL>();
    }
}
}  // namespace

#define MG_PEER_ALLREDUCE_ENTRY(NAME, T)                                          \
    extern "C" __global__ __aicore__ void NAME(                                   \
        GM_ADDR p0, GM_ADDR p1, GM_ADDR p2, GM_ADDR p3,                           \
        GM_ADDR p4, GM_ADDR p5, GM_ADDR p6, GM_ADDR p7,                           \
        GM_ADDR selfAddr, GM_ADDR xAddr, GM_ADDR outAddr,                         \
        uint32_t n, uint32_t world, uint32_t cap, uint32_t spinLimit,             \
        uint32_t sliceElems)                                                      \
    {                                                                             \
        PeerAllReduceBody<T>(p0, p1, p2, p3, p4, p5, p6, p7, selfAddr, xAddr,     \
                             outAddr, n, world, cap, spinLimit, sliceElems);      \
    }

MG_PEER_ALLREDUCE_ENTRY(peer_allreduce_f32, float)
MG_PEER_ALLREDUCE_ENTRY(peer_allreduce_bf16, bfloat16_t)
