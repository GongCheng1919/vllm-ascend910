// Cross-process device-memory mapping for the peer all-reduce probe (P6.5 D15).
//
// WHY THIS FILE EXISTS -- the first attempt was silently wrong.  `data_ptr()` was
// all-gathered and handed to the kernel as a peer address; `aclrtDeviceEnablePeerAccess`
// returned success, the kernel ran, and every rank got `world * (its own value)`.
// The reason: all four ranks' buffers had the IDENTICAL numeric address
// (0x12c041200000).  There is no unified virtual address space across devices here,
// so a raw peer pointer resolves to LOCAL memory -- no fault, no warning, just the
// wrong answer.  That is the same failure shape as
// `datacopypad-ispad-reads-real-data`: a read that is in-bounds and plausible and
// simply from the wrong place.
//
// The fix is a real mapping step.  ACL's IPC memory API returns a key for a device
// allocation; an importing process turns that key into an address VALID IN ITS OWN
// address space, with ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS asking for
// direct peer access rather than a copy.  This is also why CANN's own
// `distribute_barrier` reads remote addresses out of `HcclOpResParam` instead of
// using raw pointers -- that struct carries mapped addresses.  We get the same
// thing without any HCCL involvement.
//
// Ops registered:
//   npu.ipc_get_bare_tgid()                  -> int      pid ACL wants for whitelisting
//   npu.ipc_export(Tensor buf, int[] pids)   -> str       key, peers whitelisted
//   npu.ipc_import(str key)                  -> int       locally-valid device address

#include "utils.h"
#include "acl/acl.h"

#include <cstring>
#include <string>
#include <vector>

// torch_npu bundles its own copy of acl_rt.h
// (torch_npu/include/third_party/acl/inc/acl/acl_rt.h) and it wins the include
// path, but that copy predates the IPC memory API -- CANN 8.5's real header has
// these, torch_npu's does not, so the build fails with "was not declared in this
// scope".  The symbols ARE exported by libascendcl (verified with nm -D), so
// declare them here rather than fighting the include order.  Signatures copied
// verbatim from
// /usr/local/Ascend/cann-8.5.0/aarch64-linux/include/acl/acl_rt.h.
extern "C" {
aclError aclrtDeviceGetBareTgid(int32_t *pid);
aclError aclrtIpcMemGetExportKey(void *devPtr, size_t size, char *key, size_t len,
                                 uint64_t flags);
aclError aclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flags);
aclError aclrtIpcMemSetImportPid(const char *key, int32_t *pid, size_t num);
aclError aclrtIpcMemClose(const char *key);
}

#ifndef ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT
#define ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT 0x0UL
#endif
#ifndef ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS
#define ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS 0x1UL
#endif

namespace ascendc_path {

namespace {
// The API takes a caller-provided buffer and a length; 256 B is far beyond the
// key text ACL produces and keeps the exchange a fixed-size byte blob, which is
// what lets the caller all-gather it as a uint8 tensor instead of pickling.
constexpr size_t kKeyLen = 256;
}  // namespace

int64_t run_ipc_get_bare_tgid()
{
    int32_t pid = -1;
    aclError e = aclrtDeviceGetBareTgid(&pid);
    TORCH_CHECK(e == ACL_SUCCESS, "aclrtDeviceGetBareTgid failed with ", (int)e);
    return static_cast<int64_t>(pid);
}

std::string run_ipc_export(const at::Tensor &buf, std::vector<int64_t> pids)
{
    TORCH_CHECK(buf.is_contiguous(), "ipc_export: buf must be contiguous");
    std::vector<char> key(kKeyLen, 0);
    const size_t bytes = static_cast<size_t>(buf.numel()) * buf.element_size();
    aclError e = aclrtIpcMemGetExportKey(buf.data_ptr(), bytes, key.data(), kKeyLen,
                                         ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT);
    TORCH_CHECK(e == ACL_SUCCESS, "aclrtIpcMemGetExportKey failed with ", (int)e);

    // Whitelist the importing processes.  Without this the import is rejected --
    // and the default export flag is deliberately kept (rather than
    // DISABLE_PID_VALIDATION) so a missing pid fails loudly instead of opening
    // the allocation to every process on the box.
    std::vector<int32_t> p;
    p.reserve(pids.size());
    for (int64_t v : pids) p.push_back(static_cast<int32_t>(v));
    if (!p.empty()) {
        e = aclrtIpcMemSetImportPid(key.data(), p.data(), p.size());
        TORCH_CHECK(e == ACL_SUCCESS, "aclrtIpcMemSetImportPid failed with ", (int)e);
    }
    return std::string(key.data(), strnlen(key.data(), kKeyLen));
}

void run_ipc_close(std::string key)
{
    // Exported keys are a finite, PROCESS-OUTLIVING resource: a run that exports
    // one per shape and dies without closing leaves the next run's very first
    // aclrtIpcMemGetExportKey failing with 507899.  Callers must close.
    aclError e = aclrtIpcMemClose(key.c_str());
    TORCH_CHECK(e == ACL_SUCCESS, "aclrtIpcMemClose failed with ", (int)e);
}

int64_t run_ipc_import(std::string key)
{
    void *ptr = nullptr;
    aclError e = aclrtIpcMemImportByKey(&ptr, key.c_str(),
                                        ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
    TORCH_CHECK(e == ACL_SUCCESS, "aclrtIpcMemImportByKey failed with ", (int)e);
    TORCH_CHECK(ptr != nullptr, "aclrtIpcMemImportByKey returned null");
    return reinterpret_cast<int64_t>(ptr);
}

}  // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("ipc_get_bare_tgid() -> int");
    m.def("ipc_export(Tensor buf, int[] pids) -> str");
    m.def("ipc_import(str key) -> int");
    m.def("ipc_close(str key) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(npu, CompositeExplicitAutograd, m)
{
    m.impl("ipc_get_bare_tgid", TORCH_FN(ascendc_path::run_ipc_get_bare_tgid));
    m.impl("ipc_export", TORCH_FN(ascendc_path::run_ipc_export));
    m.impl("ipc_import", TORCH_FN(ascendc_path::run_ipc_import));
    m.impl("ipc_close", TORCH_FN(ascendc_path::run_ipc_close));
}
}
