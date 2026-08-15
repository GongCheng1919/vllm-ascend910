#pragma once
// Host harness for the mid-group activation quantiser.
//
// Checks the kernel against a CPU model of `fakequant_lab/quant.py`, then times
// it.  The point of the timing is NOT to optimise this kernel -- it is meant to
// be fused into the preceding rmsnorm/activation -- but to put a number on the
// gap the roadmap §7 flagged: the GEMM latency table does not include getting
// the activation into int4 planes.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "acl_utils.h"
#include "bf16.h"
#include "data_utils.h"

namespace qa {

constexpr uint32_t kFallbackBlocks = 20;

// (blockDim, stream, x, a_hi, a_lo, a_scale, a_ksum, M, K)
using LaunchFn = std::function<void(uint32_t, aclrtStream, void*, void*, void*,
                                    void*, void*, uint32_t, uint32_t)>;

// bf16 tiny; torch.finfo(torch.bfloat16).tiny.
constexpr float kTinyBf16 = 1.17549435e-38f;

// Round-half-to-even, matching torch.round / RoundMode::CAST_RINT.
inline float RintHalfEven(float v) { return std::nearbyint(v); }

// The CPU model of quant.py for one group; see midgroup_quant_a.inc's header.
inline void ReferenceQuantGroup(const std::vector<uint16_t>& x, size_t off, uint32_t gk,
                                std::vector<int8_t>& q, uint16_t& scale_bf16,
                                int32_t& ksum_neg)
{
    float amax = 0.0f;
    for (uint32_t k = 0; k < gk; ++k) amax = std::max(amax, std::fabs(BF16ToFP32(x[off + k])));
    float sf = amax / 127.0f;
    if (sf == 0.0f) sf = kTinyBf16;
    scale_bf16 = FP32ToBF16(sf);
    const float s = BF16ToFP32(scale_bf16);

    int64_t acc = 0;
    for (uint32_t k = 0; k < gk; ++k) {
        float v = RintHalfEven(BF16ToFP32(x[off + k]) / s);
        v = std::min(127.0f, std::max(-128.0f, v));
        const int32_t qi = static_cast<int32_t>(v);
        q[k] = static_cast<int8_t>(qi);
        acc += qi;
    }
    ksum_neg = static_cast<int32_t>(-acc);
}

inline uint32_t QueryAiCoreNum()
{
    int64_t v = 0;
    if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_CUBE_CORE_NUM, &v) == ACL_SUCCESS && v > 0) {
        return static_cast<uint32_t>(v);
    }
    return kFallbackBlocks;
}

inline int Run(int argc, char** argv, uint32_t gk, const char* opName, LaunchFn launch)
{
    Args args = ParseArgs(argc, argv);
    const uint32_t M = static_cast<uint32_t>(args.rows);
    const uint32_t K = static_cast<uint32_t>(args.k);
    if (M == 0 || K == 0 || (K % gk) || (K % 2)) {
        std::cerr << "shape constraint: K multiple of GK=" << gk << " and even\n";
        return 1;
    }
    PrintHeader(opName, args);

    const uint32_t numG = K / gk;
    const size_t xCount = (size_t)M * K;
    const size_t planeBytes = (size_t)M * (K / 2);
    const size_t sCount = (size_t)numG * M;

    // Activation-like values: a few outliers, otherwise small.  The scale is a
    // per-group max, so a flat distribution would hide clamping behaviour.
    std::vector<uint16_t> hX = GenerateBF16(xCount, args.seed, -3.0f, 3.0f);

    std::vector<int8_t> hHiRef(planeBytes), hLoRef(planeBytes);
    std::vector<uint16_t> hSRef(sCount);
    std::vector<int32_t> hKRef(sCount);
    {
        std::vector<int8_t> q(gk);
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t g = 0; g < numG; ++g) {
                uint16_t s;
                int32_t kn;
                ReferenceQuantGroup(hX, (size_t)m * K + (size_t)g * gk, gk, q, s, kn);
                hSRef[(size_t)g * M + m] = s;
                hKRef[(size_t)g * M + m] = kn;
                // Pack: even index in the LOW nibble, matching PackInt4.
                const size_t base = (size_t)m * (K / 2) + (size_t)g * (gk / 2);
                for (uint32_t j = 0; j < gk / 2; ++j) {
                    const int32_t q0 = q[2 * j], q1 = q[2 * j + 1];
                    const int32_t hi0 = q0 >> 4, hi1 = q1 >> 4;
                    const int32_t lo0 = (q0 & 15) - 8, lo1 = (q1 & 15) - 8;
                    hHiRef[base + j] = static_cast<int8_t>((hi0 & 0xF) | ((hi1 & 0xF) << 4));
                    hLoRef[base + j] = static_cast<int8_t>((lo0 & 0xF) | ((lo1 & 0xF) << 4));
                }
            }
        }
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtContext ctx = nullptr;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateContext(&ctx, 0));
    ACL_CHECK(aclrtCreateStream(&stream));

    const uint32_t coreNum = (args.blocks > 0) ? (uint32_t)args.blocks : QueryAiCoreNum();
    const uint64_t items = (uint64_t)M * numG;
    // AIV-only kernel: blockDim counts VECTOR cores, of which there are 2 per
    // AI core.  (The GEMM's blockDim counts AI cores -- different unit, same
    // parameter name.)
    const uint32_t aivTotal = coreNum * 2;
    const uint32_t blockDim = (uint32_t)std::max<uint64_t>(
        1, std::min<uint64_t>(aivTotal, items));

    void *dX = nullptr, *dHi = nullptr, *dLo = nullptr, *dS = nullptr, *dK = nullptr;
    const size_t xBytes = xCount * sizeof(uint16_t);
    const size_t sBytes = sCount * sizeof(uint16_t);
    const size_t kBytes = sCount * sizeof(int32_t);
    ACL_CHECK(aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dHi, planeBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dLo, planeBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dS, sBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dK, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(dX, xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    std::cout << "[launch]     blockDim=" << blockDim << " AIV (of " << aivTotal
              << ")  M=" << M << " K=" << K
              << " GK=" << gk << " groups=" << numG << " items=" << items << "\n"
              << std::flush;

    auto run_once = [&]() { launch(blockDim, stream, dX, dHi, dLo, dS, dK, M, K); };

    bool overall = true;
    if (args.check) {
        ACL_CHECK(aclrtMemset(dHi, planeBytes, 0, planeBytes));
        ACL_CHECK(aclrtMemset(dLo, planeBytes, 0, planeBytes));
        run_once();
        ACL_CHECK(aclrtSynchronizeStream(stream));

        std::vector<int8_t> hHi(planeBytes), hLo(planeBytes);
        std::vector<uint16_t> hS(sCount);
        std::vector<int32_t> hK(sCount);
        ACL_CHECK(aclrtMemcpy(hHi.data(), planeBytes, dHi, planeBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hLo.data(), planeBytes, dLo, planeBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hS.data(), sBytes, dS, sBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(hK.data(), kBytes, dK, kBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        auto cmp = [&](const char* what, const void* a, const void* b, size_t n) {
            const size_t bad = [&] {
                size_t c = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (((const uint8_t*)a)[i] != ((const uint8_t*)b)[i]) ++c;
                }
                return c;
            }();
            std::cout << "[check]      " << what << (bad ? "  FAIL  " : "  PASS  ")
                      << "byte mismatches=" << bad << " / " << n << "\n";
            if (bad) overall = false;
        };
        std::cout << "[debug]      first 4 scales  dev / ref (bits, value):\n";
        for (int i = 0; i < 4 && (size_t)i < sCount; ++i) {
            std::cout << "             [" << i << "] 0x" << std::hex << hS[i]
                      << " / 0x" << hSRef[i] << std::dec
                      << "   " << BF16ToFP32(hS[i]) << " / " << BF16ToFP32(hSRef[i])
                      << "   ksum " << hK[i] << " / " << hKRef[i] << "\n";
        }
        std::cout << "[debug]      first 8 a_hi bytes dev / ref: ";
        for (int i = 0; i < 8; ++i) {
            std::cout << (int)(uint8_t)hHi[i] << "/" << (int)(uint8_t)hHiRef[i] << " ";
        }
        std::cout << "\n" << std::flush;
        cmp("a_hi   ", hHi.data(), hHiRef.data(), planeBytes);
        cmp("a_lo   ", hLo.data(), hLoRef.data(), planeBytes);
        cmp("a_scale", hS.data(), hSRef.data(), sBytes);
        cmp("a_ksum ", hK.data(), hKRef.data(), kBytes);
    }

    if (args.profile) {
        for (int i = 0; i < args.warmup; ++i) run_once();
        ACL_CHECK(aclrtSynchronizeStream(stream));
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < args.repeat; ++i) run_once();
        ACL_CHECK(aclrtSynchronizeStream(stream));
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double us =
            std::chrono::duration<double, std::micro>(t1 - t0).count() / args.repeat;
        std::cout << "[perf]       " << us << " us/iter\n";
        std::cout << "[csv]        " << opName << "," << M << "," << K << "," << gk
                  << "," << us << "\n";
    }

    aclrtFree(dK);
    aclrtFree(dS);
    aclrtFree(dLo);
    aclrtFree(dHi);
    aclrtFree(dX);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();

    std::cout << "[status]     " << (overall ? "PASS" : "FAIL") << "\n";
    return overall ? 0 : 1;
}

}  // namespace qa
