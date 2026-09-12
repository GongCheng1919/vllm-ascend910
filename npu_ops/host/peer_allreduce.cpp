// Host side of the in-kernel-barrier all-reduce (P6.5 D17).
//
//   npu.peer_allreduce(Tensor x, Tensor staging, int[] ptrs, int self_idx,
//                      int cap, int spin_limit) -> Tensor
//
// `staging` is this rank's mapped buffer (see peer_ipc.cpp); `ptrs` are every
// rank's staging base in THIS rank's address space, `self_idx` says which entry is
// ours. The layout the kernel expects is documented in the kernel and reproduced by
// `run_peer_staging_elems` below so the two cannot drift.
//
// x may be float32 or bfloat16; `cap` is counted in float-sized elements either
// way, so ONE staging buffer -- exported once, see the 507899 trap in D16 -- serves
// both dtypes.
//
// No `dist.barrier()` is needed around this op -- that is the entire point. It is
// what makes the op legal inside a captured graph, where a host-side collective
// cannot appear.

#include "utils.h"
#include "aclrtlaunch_peer_allreduce_f32.h"
#include "aclrtlaunch_peer_allreduce_bf16.h"

namespace ascendc_path {

namespace {
// These MUST match kernel/peer_allreduce_f32.cpp -- the flag areas are read by
// PEER kernels, so a mismatch is silent cross-rank corruption, not a crash.
constexpr int64_t kDataBase   = 1024;  // float elements
constexpr int64_t kAlign      = 8;     // floats per 32 B block
constexpr int64_t kMaxBlocks  = 40;    // AIV cores on 910B4
constexpr int64_t kSliceBytes = 512;   // slice alignment: two blocks must never
                                       // share a cache line, or one block's flush
                                       // races another block's staging
constexpr size_t  kMaxPeers   = 8;

// Smallest slice worth giving a block.  Below this the per-block barrier (a spin
// on every peer's flag) costs more than the bytes it guards, so more blocks would
// make the op slower rather than faster.
constexpr int64_t kMinSliceBytes = 4096;

// Slice size and block count are derived from `n` and the dtype ALONE,
// deliberately: block b waits on its peers' block b, so every rank must cut
// identical slices -- and they have to agree on that without communicating.
int64_t slice_elems_for(int64_t n, int64_t esize)
{
    const int64_t align = kSliceBytes / esize;
    const int64_t even = (n + kMaxBlocks - 1) / kMaxBlocks;
    const int64_t aligned = (even + align - 1) / align * align;
    return std::max<int64_t>(kMinSliceBytes / esize, aligned);
}
}  // namespace

// Elements a staging buffer needs for a given per-slot capacity (two slots).
int64_t run_peer_staging_elems(int64_t cap)
{
    TORCH_CHECK(cap % kAlign == 0, "peer_allreduce: cap must be a multiple of ",
                kAlign, ", got ", cap);
    return kDataBase + 2 * cap;
}

at::Tensor run_peer_allreduce(const at::Tensor &x, const at::Tensor &staging,
                              std::vector<int64_t> ptrs, int64_t self_idx,
                              int64_t cap, int64_t spin_limit)
{
    TORCH_CHECK(x.dim() == 1 && x.is_contiguous(),
                "peer_allreduce: x must be a contiguous 1-D tensor");
    const bool is_bf16 = x.scalar_type() == at::kBFloat16;
    TORCH_CHECK(is_bf16 || x.scalar_type() == at::kFloat,
                "peer_allreduce: x must be float32 or bfloat16, got ",
                x.scalar_type());
    // bf16 is the format the model actually uses (the mid-group GEMM emits bf16),
    // and it halves what crosses the fabric.  The reduction still accumulates in
    // f32 inside the kernel, so this path is MORE accurate than a bf16 collective,
    // not less.
    const int64_t esize = is_bf16 ? 2 : 4;
    TORCH_CHECK(staging.scalar_type() == at::kFloat && staging.is_contiguous(),
                "peer_allreduce: staging must be contiguous float32");
    const int64_t world = static_cast<int64_t>(ptrs.size());
    TORCH_CHECK(world >= 1 && world <= (int64_t)kMaxPeers,
                "peer_allreduce: world must be 1..", kMaxPeers);
    TORCH_CHECK(self_idx >= 0 && self_idx < world, "peer_allreduce: bad self_idx");

    const int64_t n = x.size(0);
    // DataCopy moves whole 32 B blocks: 8 floats, or 16 bf16.
    const int64_t nalign = 32 / esize;
    TORCH_CHECK(n % nalign == 0, "peer_allreduce: n must be a multiple of ", nalign,
                " for ", x.scalar_type(), ", got ", n);
    // `cap` is counted in FLOAT-sized elements for both dtypes, so one mapped
    // staging buffer serves either without re-exporting an IPC key.
    TORCH_CHECK(n * esize <= cap * 4, "peer_allreduce: n=", n, " (", n * esize,
                " B) exceeds slot capacity ", cap, " (", cap * 4, " B)");
    TORCH_CHECK(staging.numel() >= run_peer_staging_elems(cap),
                "peer_allreduce: staging too small for cap=", cap);

    // The peer addresses must be pairwise distinct. Identical values mean the IPC
    // mapping did not happen and every "peer" read would silently return LOCAL
    // memory -- the failure that cost a round in D16, and one that produces a
    // plausible wrong answer rather than an error.
    for (int64_t i = 0; i < world; ++i) {
        for (int64_t j = i + 1; j < world; ++j) {
            TORCH_CHECK(ptrs[i] != ptrs[j],
                        "peer_allreduce: ptrs[", i, "] == ptrs[", j,
                        "] -- peer buffers were not IPC-mapped");
        }
    }

    at::Tensor out = at::empty_like(x);

    void *p[kMaxPeers];
    for (size_t i = 0; i < kMaxPeers; ++i) {
        p[i] = reinterpret_cast<void *>(ptrs[i < (size_t)world ? i : 0]);
    }

    const uint32_t uN = static_cast<uint32_t>(n);
    const uint32_t uW = static_cast<uint32_t>(world);
    const uint32_t uC = static_cast<uint32_t>(cap);
    const uint32_t uS = static_cast<uint32_t>(spin_limit);
    // Two constraints from EXEC_KERNEL_CMD, both of which fail at COMPILE time in
    // ways that point at utils.h rather than here:
    //   * `blockdim` is pasted into the lambda's capture list, so it must be a
    //     named variable -- a literal `1u` becomes `[acl_stream, 1u, ...]`;
    //   * ConvertTypes takes non-const lvalue REFERENCES, so every argument must
    //     be an lvalue -- a `reinterpret_cast<void*>(...)` rvalue does not bind.
    // One block per slice; the barrier is per block (see the kernel header).
    const int64_t slice = slice_elems_for(n, esize);
    const uint32_t uSlice = static_cast<uint32_t>(slice);
    const uint32_t blockDim = static_cast<uint32_t>(
        std::min<int64_t>(kMaxBlocks, (n + slice - 1) / slice));
    void *selfPtr = reinterpret_cast<void *>(ptrs[self_idx]);
    if (is_bf16) {
        EXEC_KERNEL_CMD(peer_allreduce_bf16, blockDim,
                        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                        selfPtr, x, out,
                        uN, uW, uC, uS, uSlice);
    } else {
        EXEC_KERNEL_CMD(peer_allreduce_f32, blockDim,
                        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                        selfPtr, x, out,
                        uN, uW, uC, uS, uSlice);
    }
    return out;
}

}  // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("peer_staging_elems(int cap) -> int");
    m.def("peer_allreduce(Tensor x, Tensor staging, int[] ptrs, int self_idx, "
          "int cap, int spin_limit) -> Tensor");
}
}

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("peer_allreduce", TORCH_FN(ascendc_path::run_peer_allreduce));
}
}

namespace {
TORCH_LIBRARY_IMPL(npu, CompositeExplicitAutograd, m)
{
    m.impl("peer_staging_elems", TORCH_FN(ascendc_path::run_peer_staging_elems));
}
}
