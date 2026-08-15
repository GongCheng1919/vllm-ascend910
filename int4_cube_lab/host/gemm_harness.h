#pragma once
// Shared host harness for the per-channel low-bit GEMM kernels.
// The two mains differ only in QBITS and the aclrtlaunch_* symbol they pass in.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>

#include "acl/acl.h"
#include "acl_utils.h"
#include "data_utils.h"
#include "gemm_ref.h"

namespace harness {

constexpr uint32_t kTileN = 128;
constexpr uint32_t kInnerKB = 256;      // must match the kernel's INNER_KB
constexpr uint32_t kFallbackBlocks = 20;
constexpr double   kMinSnrDb = 40.0;
// 910B4 L2 is 168 MiB.  Benchmarking one weight tensor in a loop serves it from
// L2, which real inference never does — 64 distinct layers stream from HBM.
// --cold rotates over enough copies to blow past L2 between reuses.
constexpr size_t   kColdFootprint = 1024ull * 1024 * 1024;
constexpr uint32_t kMaxRotate = 16;

// (blockDim, stream, a, a_scale, w, w_scale, y, workspace, M, N, K)
using LaunchFn = std::function<void(uint32_t, aclrtStream, void*, void*, void*,
                                    void*, void*, void*, uint32_t, uint32_t, uint32_t)>;

inline uint32_t QueryAiCoreNum()
{
    int64_t v = 0;
    // MIX_AIC_1_2 blockDim counts AIC (cube) blocks.
    if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_CUBE_CORE_NUM, &v) == ACL_SUCCESS && v > 0) {
        return static_cast<uint32_t>(v);
    }
    if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_AICORE_CORE_NUM, &v) == ACL_SUCCESS && v > 0) {
        return static_cast<uint32_t>(v);
    }
    return kFallbackBlocks;
}

inline int Run(int argc, char** argv, uint32_t qbits, uint32_t tileM,
               const char* opName, LaunchFn launch)
{
    const uint32_t elemsPerByte = 8 / qbits;
    const uint32_t kAlign = kInnerKB * elemsPerByte;   // 512 for int4, 256 for int8

    Args args = ParseArgs(argc, argv);
    const uint32_t M = static_cast<uint32_t>(args.rows);
    const uint32_t N = static_cast<uint32_t>(args.cols);
    const uint32_t K = static_cast<uint32_t>(args.k);

    if (M == 0 || N == 0 || K == 0 || (N % kTileN) || (K % kAlign)) {
        std::cerr << "shape constraint: N multiple of 128; K multiple of "
                  << kAlign << " (int" << qbits << " build)\n";
        return 1;
    }
    // M is rounded up to the tile height.  16 is the cube fractal height, so a
    // decode-shaped M=1 is padded to 16 by any cube kernel, not just this one;
    // FLOPS below are still computed from the TRUE M.
    const uint32_t Mp = ((M + tileM - 1) / tileM) * tileM;
    if (K / elemsPerByte > 65535u) {
        std::cerr << "K/elemsPerByte must fit in uint16 (Nd2Nz srcDValue)\n";
        return 1;
    }
    PrintHeader(opName, args);

    const size_t aCount  = static_cast<size_t>(Mp) * K;
    const size_t wCount  = static_cast<size_t>(N) * K;
    const size_t yCount  = static_cast<size_t>(Mp) * N;
    const size_t yRealCount = static_cast<size_t>(M) * N;
    const size_t numTiles = static_cast<size_t>(Mp / tileM) * (N / kTileN);

    // Values live in [-8,7] for both builds so the two kernels can be compared
    // on bit-identical inputs.
    std::vector<int8_t> hA = GenerateInt8(aCount, args.seed, -8, 7);
    std::vector<int8_t> hW = GenerateInt8(wCount, args.seed + 101, -8, 7);
    std::vector<uint16_t> hAScale = GenerateBF16(Mp, args.seed + 202, 0.0005f, 0.0020f);
    std::vector<uint16_t> hWScale = GenerateBF16(N, args.seed + 303, 0.0005f, 0.0020f);

    std::vector<uint16_t> hYRef;
    if (args.check || args.stress > 0) {
        std::cout << "[debug]      CPU reference (" << M << "x" << N << "x" << K
                  << ")...\n" << std::flush;
        ReferencePerchannelGemm(hA, hAScale, hW, hWScale, hYRef, M, N, K);  // real rows only
        std::cout << "[debug]      CPU reference done\n" << std::flush;
    }

    // Device-side payload: packed for int4, raw for int8.
    const std::vector<int8_t> aDev = (qbits == 4) ? PackInt4(hA, Mp, K) : hA;
    const std::vector<int8_t> wDev = (qbits == 4) ? PackInt4(hW, N, K) : hW;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtContext ctx = nullptr;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateContext(&ctx, 0));
    ACL_CHECK(aclrtCreateStream(&stream));

    const uint32_t coreNum = (args.blocks > 0) ? static_cast<uint32_t>(args.blocks)
                                               : QueryAiCoreNum();
    const uint32_t blockDim = static_cast<uint32_t>(
        std::max<size_t>(1, std::min<size_t>(coreNum, numTiles)));

    const size_t aBytes  = aDev.size();
    const size_t wBytes  = wDev.size();
    const size_t asBytes = static_cast<size_t>(Mp) * sizeof(uint16_t);
    const size_t wsclBytes = static_cast<size_t>(N) * sizeof(uint16_t);
    const size_t yBytes  = yCount * sizeof(uint16_t);
    const size_t wsBytes = static_cast<size_t>(blockDim) * tileM * kTileN * sizeof(int32_t);

    const uint32_t rotate = args.cold
        ? std::min<uint32_t>(kMaxRotate,
              std::max<uint32_t>(2, static_cast<uint32_t>(
                  (kColdFootprint + wBytes - 1) / wBytes)))
        : 1;

    void *dA = nullptr, *dAScale = nullptr, *dWScale = nullptr;
    void *dY = nullptr, *dWs = nullptr;
    std::vector<void*> dW(rotate, nullptr);
    ACL_CHECK(aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dAScale, asBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    for (uint32_t i = 0; i < rotate; ++i) {
        ACL_CHECK(aclrtMalloc(&dW[i], wBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ACL_CHECK(aclrtMalloc(&dWScale, wsclBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dWs, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(dA, aBytes, aDev.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dAScale, asBytes, hAScale.data(), asBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    for (uint32_t i = 0; i < rotate; ++i) {
        ACL_CHECK(aclrtMemcpy(dW[i], wBytes, wDev.data(), wBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ACL_CHECK(aclrtMemcpy(dWScale, wsclBytes, hWScale.data(), wsclBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t rotIdx = 0;
    auto run_once = [&]() {
        void* w = dW[rotIdx];
        if (rotate > 1) rotIdx = (rotIdx + 1) % rotate;
        launch(blockDim, stream, dA, dAScale, w, dWScale, dY, dWs, Mp, N, K);
    };

    std::cout << "[launch]     blockDim=" << blockDim
              << " (aicore=" << coreNum << ")  tileM=" << tileM
              << "  M=" << M << (Mp != M ? " -> padded " : " ") << (Mp != M ? std::to_string(Mp) : std::string())
              << "  tiles=" << numTiles
              << "  workspace=" << (double(wsBytes) / (1024.0 * 1024.0)) << " MiB"
              << (rotate > 1 ? "  cold(x" + std::to_string(rotate) + " W copies)" : "")
              << "\n"
              << std::flush;

    bool overall = true;

    auto check_now = [&]() {
        std::vector<uint16_t> hY(yCount);
        ACL_CHECK(aclrtMemcpy(hY.data(), yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        hY.resize(yRealCount);              // padded rows carry no reference
        auto r = CompareBF16(hY, hYRef, 2.0e-2f, 8.0e-2f);
        // atol/rtol alone is vacuous when |y| ~ 1e-3: a completely wrong output
        // still lands inside atol.  An int32-exact GEMM rounded to bf16 scores
        // ~80 dB, so anything under 40 dB is a broken kernel, not rounding.
        if (r.snr_db < kMinSnrDb) r.pass = false;
        return r;
    };

    if (args.check || args.stress > 0) {
        const int iters = std::max(1, args.stress);
        int passes = 0;
        CompareReport worst{true, 0, 0.0, 0.0, 999.0, SIZE_MAX};
        for (int i = 0; i < iters; ++i) {
            ACL_CHECK(aclrtMemset(dY, yBytes, 0, yBytes));
            run_once();
            ACL_CHECK(aclrtSynchronizeStream(stream));
            auto rep = check_now();
            if (rep.pass) {
                ++passes;
            } else if (worst.pass || rep.mismatches > worst.mismatches) {
                worst = rep;
            }
            if (iters > 1) {
                std::cout << "[stress]     run " << (i + 1) << "/" << iters << "  "
                          << (rep.pass ? "PASS" : "FAIL")
                          << "  mismatches=" << rep.mismatches << "\n" << std::flush;
            } else {
                PrintCheck("y (bf16)", rep);
            }
        }
        if (iters > 1) {
            std::cout << "[stress]     " << passes << "/" << iters << " runs PASS\n";
            if (passes < iters) PrintCheck("y (worst run)", worst);
        }
        overall = overall && (passes == iters);
    }

    if (args.profile) {
        // --nosync queues every repeat and synchronizes once, so the reported
        // time is kernel time rather than per-launch host round-trip.  At the
        // small end (512^3) the round-trip dominates by ~100x.
        double avgUs;
        if (args.nosync) {
            for (int i = 0; i < args.warmup; ++i) run_once();
            ACL_CHECK(aclrtSynchronizeStream(stream));
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < args.repeat; ++i) run_once();
            ACL_CHECK(aclrtSynchronizeStream(stream));
            const auto t1 = std::chrono::high_resolution_clock::now();
            avgUs = std::chrono::duration<double, std::micro>(t1 - t0).count() / args.repeat;
        } else {
            avgUs = BenchmarkUs([&]() {
                run_once();
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }, args.warmup, args.repeat);
        }
        const double ops = 2.0 * double(M) * double(N) * double(K);
        PrintTime(avgUs, -1.0, ops / avgUs / 1.0e6);
        std::cout << "[csv]        " << opName << "," << M << "," << N << "," << K
                  << "," << avgUs << "," << (ops / avgUs / 1.0e6) << "\n";
    }

    PrintStatus(overall);

    aclrtFree(dWs);
    aclrtFree(dY);
    aclrtFree(dWScale);
    for (uint32_t i = 0; i < rotate; ++i) aclrtFree(dW[i]);
    aclrtFree(dAScale);
    aclrtFree(dA);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return overall ? 0 : 1;
}

}  // namespace harness
