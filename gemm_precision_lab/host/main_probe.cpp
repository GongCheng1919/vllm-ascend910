// Probe: which ACL step hangs on this host?
#include <cstdio>
#include "acl/acl.h"

#define STEP(s, expr)                                        \
    do {                                                     \
        fprintf(stderr, "[probe] %s ...\n", s);              \
        fflush(stderr);                                      \
        aclError e = (expr);                                 \
        fprintf(stderr, "[probe] %s -> %d\n", s, (int)e);    \
        fflush(stderr);                                      \
        if (e != ACL_SUCCESS) return 1;                      \
    } while (0)

int main() {
    STEP("aclInit", aclInit(nullptr));
    STEP("aclrtSetDevice(0)", aclrtSetDevice(0));
    aclrtContext ctx = nullptr;
    STEP("aclrtCreateContext", aclrtCreateContext(&ctx, 0));
    aclrtStream stream = nullptr;
    STEP("aclrtCreateStream", aclrtCreateStream(&stream));
    void* d = nullptr;
    STEP("aclrtMalloc 1MB", aclrtMalloc(&d, 1 << 20, ACL_MEM_MALLOC_HUGE_FIRST));
    STEP("aclrtFree", aclrtFree(d));
    STEP("aclrtDestroyStream", aclrtDestroyStream(stream));
    STEP("aclrtDestroyContext", aclrtDestroyContext(ctx));
    STEP("aclrtResetDevice", aclrtResetDevice(0));
    STEP("aclFinalize", aclFinalize());
    fprintf(stderr, "[probe] ALL OK\n");
    return 0;
}
