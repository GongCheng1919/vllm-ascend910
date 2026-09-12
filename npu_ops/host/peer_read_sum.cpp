// Host side of the peer-access all-reduce probe (P6.5 D15 follow-up).
//
// Registers two ops:
//   npu.enable_peer_access(int[] devices) -> ()      one-time, per rank
//   npu.peer_read_sum(Tensor self_buf, int[] ptrs) -> Tensor
//
// `ptrs` are raw device addresses of every rank's staging buffer, gathered by the
// caller (torch.distributed all_gather of data_ptr()).  Handing raw pointers across
// processes is only sound because every rank has called `enable_peer_access` for
// the others first; without that the kernel faults rather than reading garbage.
//
// The point of the whole exercise is that this path touches NONE of the HCCL
// comm-resource machinery that CANN's MC2 hangs in inside vLLM.

#include "utils.h"
#include "acl/acl.h"
#include "aclrtlaunch_peer_read_sum_f32.h"

namespace ascendc_path {

namespace {
constexpr uint32_t kAivNum = 40;   // 910B4: 2 vector cores per AI core
constexpr int64_t  kAlign  = 8;    // floats per 32 B block
constexpr size_t   kMaxPeers = 8;
}  // namespace

void run_enable_peer_access(std::vector<int64_t> devices)
{
    for (int64_t d : devices) {
        aclError e = aclrtDeviceEnablePeerAccess(static_cast<int32_t>(d), 0);
        // Re-enabling an already-enabled peer is not an error we care about; every
        // other status is, because a silent failure here turns into a device fault
        // deep inside the kernel where it is far harder to read.
        TORCH_CHECK(e == ACL_SUCCESS,
                    "aclrtDeviceEnablePeerAccess(", d, ") failed with ", (int)e);
    }
}

at::Tensor run_peer_read_sum(const at::Tensor &self_buf, std::vector<int64_t> ptrs)
{
    TORCH_CHECK(self_buf.dim() == 1, "peer_read_sum: self_buf must be 1-D");
    TORCH_CHECK(self_buf.scalar_type() == at::kFloat, "peer_read_sum: float32 only");
    TORCH_CHECK(self_buf.is_contiguous(), "peer_read_sum: self_buf must be contiguous");
    const int64_t world = static_cast<int64_t>(ptrs.size());
    TORCH_CHECK(world >= 1 && world <= (int64_t)kMaxPeers,
                "peer_read_sum: world must be 1..", kMaxPeers, ", got ", world);

    const int64_t n = self_buf.size(0);
    // The kernel's tail rounds counts up to 8 floats, so the buffers it reads and
    // writes must be a multiple of 8 long.  Enforced rather than padded here: a
    // silent pad would make the caller's `data_ptr()` exchange describe a different
    // length than the kernel walks.
    TORCH_CHECK(n % kAlign == 0,
                "peer_read_sum: n must be a multiple of ", kAlign, ", got ", n);

    at::Tensor out = at::empty_like(self_buf);

    void *p[kMaxPeers];
    for (size_t i = 0; i < kMaxPeers; ++i) {
        p[i] = (i < (size_t)world) ? reinterpret_cast<void *>(ptrs[i])
                                   : reinterpret_cast<void *>(ptrs[0]);
    }

    const int64_t blocks = (n + kAlign - 1) / kAlign;
    const uint32_t blockDim = static_cast<uint32_t>(
        std::max<int64_t>(1, std::min<int64_t>(kAivNum, blocks)));

    const uint32_t uN = static_cast<uint32_t>(n);
    const uint32_t uW = static_cast<uint32_t>(world);
    EXEC_KERNEL_CMD(peer_read_sum_f32, blockDim,
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                    out, uN, uW);
    return out;
}

}  // namespace ascendc_path

// Anonymous namespace: TORCH_LIBRARY_FRAGMENT expands to a static initialiser whose
// generated name restarts at 0 per translation unit, so a named namespace would
// collide with the other host files and the linker would keep only one.
namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("enable_peer_access(int[] devices) -> ()");
    m.def("peer_read_sum(Tensor self_buf, int[] ptrs) -> Tensor");
}
}

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("peer_read_sum", TORCH_FN(ascendc_path::run_peer_read_sum));
}
}

namespace {
TORCH_LIBRARY_IMPL(npu, CompositeExplicitAutograd, m)
{
    m.impl("enable_peer_access", TORCH_FN(ascendc_path::run_enable_peer_access));
}
}
