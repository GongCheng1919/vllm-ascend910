// peer_read_sum_f32 -- the smallest possible self-written cross-rank kernel.
//
// WHY THIS EXISTS (P6.5 D15).  CANN's own MC2 fused matmul+all-reduce microbenches
// green on this box and then hangs inside vLLM, in graph mode AND in eager.  That
// killed the plan of pricing a fused mid-group GEMM + all-reduce with a vendor op,
// and it put a question mark over the whole fused-op line: is an operator that does
// cross-rank communication from inside the kernel usable here at all?
//
// This kernel answers the narrowest version of that question, with the fewest
// moving parts that can fail:
//
//   * NO HCCL.  No `Hccl` object, no `InitV2`, no comm-resource allocation, no task
//     queue -- i.e. none of the machinery that MC2 died in.  Peer HBM is reached
//     directly, which `aclrtDeviceEnablePeerAccess` makes legal and which measured
//     canAccessPeer=1 for every pair of the four TP devices, including the pair that
//     straddles the {0-3}/{4-7} topology boundary.
//   * NO in-kernel barrier yet.  Ranks are synchronised by the CALLER for this first
//     milestone, so a failure here can only mean "a kernel cannot read peer memory",
//     not "our barrier is wrong".  The in-kernel barrier is the next milestone and
//     is what graph capture actually needs.
//
// Semantics: out[i] = sum over r in [0, world) of peers[r][i].  Every rank passes
// the same peer pointer list (its own included), so every rank computes the same
// sum -- an all-reduce, done by reading instead of by messaging.
//
// The pointer list is a fixed 8 slots because kernel arguments are positional;
// slots >= world are never dereferenced.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
// 4096 floats = 16 KB per UB buffer, two buffers -> 32 KB of the 192 KB UB.
// Deliberately small: this milestone is about "does the read work", not bandwidth.
constexpr uint32_t kChunk = 4096;
// float32 DataCopy moves whole 32 B blocks, so every count must be a multiple of 8.
constexpr uint32_t kAlign = 8;

__aicore__ inline uint32_t CeilTo(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }
}  // namespace

extern "C" __global__ __aicore__ void peer_read_sum_f32(
    GM_ADDR p0, GM_ADDR p1, GM_ADDR p2, GM_ADDR p3,
    GM_ADDR p4, GM_ADDR p5, GM_ADDR p6, GM_ADDR p7,
    GM_ADDR outAddr, uint32_t n, uint32_t world)
{
    const uint32_t cores = GetBlockNum();
    const uint32_t bid   = GetBlockIdx();

    // Split on a 32 B boundary so no core ever issues an unaligned DataCopy.
    const uint32_t perCore = CeilTo(CeilTo(n, kAlign) / cores, kAlign);
    const uint32_t begin   = bid * perCore;
    if (begin >= n) return;
    const uint32_t myN = (begin + perCore <= n) ? perCore : (n - begin);

    TPipe pipe;
    TBuf<TPosition::VECCALC> accBuf, tmpBuf;
    pipe.InitBuffer(accBuf, kChunk * sizeof(float));
    pipe.InitBuffer(tmpBuf, kChunk * sizeof(float));
    LocalTensor<float> acc = accBuf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();

    GlobalTensor<float> outG;
    outG.SetGlobalBuffer((__gm__ float *)outAddr + begin, myN);

    for (uint32_t off = 0; off < myN; off += kChunk) {
        const uint32_t cnt = (myN - off < kChunk) ? (myN - off) : kChunk;
        // The tail may read/write up to 7 floats past `n`; the caller sizes every
        // buffer to a multiple of 8 floats so that stays inside the allocation.
        const uint32_t cntA = CeilTo(cnt, kAlign);

        // The peer pointers are consumed by an UNROLLED sequence, never through a
        // runtime-indexed array.  The first version wrote
        //     GM_ADDR peers[8] = {p0, ...}; ... peers[r]
        // and every rank got world * (its OWN value): the runtime index collapsed
        // to peers[0].  It looked exactly like "peer reads do not work" -- reading
        // one pointer at a time (world == 1) passed, because that path only ever
        // touches index 0.  Do not reintroduce the array.
#define MG_ACC_PEER(IDX, PTR)                                                     \
        if ((IDX) < world) {                                                      \
            GlobalTensor<float> src;                                              \
            src.SetGlobalBuffer((__gm__ float *)(PTR) + begin + off, cntA);       \
            DataCopy(tmp, src, cntA);                                             \
            /* PIPE_ALL on every step: this is the correctness milestone, and a */ \
            /* missing MTE2->V barrier would look exactly like a bad peer read  */ \
            /* (P6.5 D3 / the AIV WAR hazard note).  Tighten later.             */ \
            PipeBarrier<PIPE_ALL>();                                              \
            if ((IDX) == 0) { Adds(acc, tmp, 0.0f, cntA); }                       \
            else            { Add(acc, acc, tmp, cntA); }                         \
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

        DataCopy(outG[off], acc, cntA);
        PipeBarrier<PIPE_ALL>();
    }
}
