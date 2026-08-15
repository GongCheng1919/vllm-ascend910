#pragma once
// Shared host harness for the W4A8 per-channel GEMM kernels.
//
// Differs from gemm_harness.h in three ways, all of them consequences of the
// MSD split (see kernels/perchannel_w4a8_gemm.inc):
//   * A is generated over the FULL int8 range and shipped as two packed int4
//     planes, a_hi and a_lo.  Their combined size equals an int8 A exactly.
//   * the kernel also takes w_ksum[N] = sum_k W[n,k], a weight-side constant.
//   * the workspace tile is 2*TILE_M rows tall, not TILE_M.
//
// The CPU reference is the plain int8 x int4 GEMM: if the split, the cube path
// and the +8*ksum correction are all right, the kernel reproduces it exactly.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl_utils.h"
#include "data_utils.h"
#include "gemm_ref.h"

namespace w4a8 {

constexpr uint32_t kTileN = 128;
constexpr uint32_t kFallbackBlocks = 20;
constexpr double   kMinSnrDb = 40.0;
constexpr size_t   kColdFootprint = 1024ull * 1024 * 1024;
constexpr uint32_t kMaxRotate = 16;

// (blockDim, stream, a_hi, a_lo, a_scale, w, w_scale, w_ksum, y, workspace, M, N, K)
using LaunchFn = std::function<void(uint32_t, aclrtStream, void*, void*, void*,
                                    void*, void*, void*, void*, void*,
                                    uint32_t, uint32_t, uint32_t)>;

inline uint32_t QueryAiCoreNum()
{
    int64_t v = 0;
    if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_CUBE_CORE_NUM, &v) == ACL_SUCCESS && v > 0) {
        return static_cast<uint32_t>(v);
    }
    if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_AICORE_CORE_NUM, &v) == ACL_SUCCESS && v > 0) {
        return static_cast<uint32_t>(v);
    }
    return kFallbackBlocks;
}

inline int Run(int argc, char** argv, uint32_t tileM, uint32_t innerK,
               const char* opName, LaunchFn launch)
{
    Args args = ParseArgs(argc, argv);
    const uint32_t M = static_cast<uint32_t>(args.rows);
    const uint32_t N = static_cast<uint32_t>(args.cols);
    const uint32_t K = static_cast<uint32_t>(args.k);

    if (M == 0 || N == 0 || K == 0 || (N % kTileN) || (K % innerK)) {
        std::cerr << "shape constraint: N multiple of 128; K multiple of "
                  << innerK << " (tileM=" << tileM << " build)\n";
        return 1;
    }
    const uint32_t Mp = ((M + tileM - 1) / tileM) * tileM;
    if (K / 2 > 65535u) {
        std::cerr << "K/2 must fit in uint16 (Nd2Nz srcDValue)\n";
        return 1;
    }
    PrintHeader(opName, args);

    const size_t aCount = static_cast<size_t>(Mp) * K;
    const size_t wCount = static_cast<size_t>(N) * K;
    const size_t yCount = static_cast<size_t>(Mp) * N;
    const size_t yRealCount = static_cast<size_t>(M) * N;
    const size_t numTiles = static_cast<size_t>(Mp / tileM) * (N / kTileN);

    // A spans the full int8 range on purpose: the MSD split has to be exact
    // there, not merely on the [-8,7] subrange the W4A4 lab used.  W is int4.
    std::vector<int8_t> hA = GenerateInt8(aCount, args.seed, -128, 127);
    std::vector<int8_t> hW = GenerateInt8(wCount, args.seed + 101, -8, 7);
    std::vector<uint16_t> hAScale = GenerateBF16(Mp, args.seed + 202, 0.0005f, 0.0020f);
    std::vector<uint16_t> hWScale = GenerateBF16(N, args.seed + 303, 0.0005f, 0.0020f);

    std::vector<uint16_t> hYRef;
    if (args.check || args.stress > 0) {
        std::cout << "[debug]      CPU reference (" << M << "x" << N << "x" << K
                  << ")...\n" << std::flush;
        ReferencePerchannelGemm(hA, hAScale, hW, hWScale, hYRef, M, N, K);
        std::cout << "[debug]      CPU reference done\n" << std::flush;
    }

    std::vector<int8_t> hAHi, hALo;
    SplitInt8ToInt4Planes(hA, Mp, K, hAHi, hALo);
    const std::vector<int8_t> wDev = PackInt4(hW, N, K);
    const std::vector<int32_t> hKSum = WeightKSum(hW, N, K);

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

    const size_t aPlaneBytes = hAHi.size();
    const size_t wBytes      = wDev.size();
    const size_t asBytes     = static_cast<size_t>(Mp) * sizeof(uint16_t);
    const size_t wsclBytes   = static_cast<size_t>(N) * sizeof(uint16_t);
    const size_t ksumBytes   = static_cast<size_t>(N) * sizeof(int32_t);
    const size_t yBytes      = yCount * sizeof(uint16_t);
    // 2*tileM rows: the cube tile stacks the hi and lo planes.
    const size_t wspBytes = static_cast<size_t>(blockDim) * 2 * tileM * kTileN * sizeof(int32_t);

    const uint32_t rotate = args.cold
        ? std::min<uint32_t>(kMaxRotate,
              std::max<uint32_t>(2, static_cast<uint32_t>(
                  (kColdFootprint + wBytes - 1) / wBytes)))
        : 1;

    void *dAHi = nullptr, *dALo = nullptr, *dAScale = nullptr;
    void *dWScale = nullptr, *dKSum = nullptr, *dY = nullptr, *dWsp = nullptr;
    std::vector<void*> dW(rotate, nullptr);
    ACL_CHECK(aclrtMalloc(&dAHi, aPlaneBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dALo, aPlaneBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dAScale, asBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    for (uint32_t i = 0; i < rotate; ++i) {
        ACL_CHECK(aclrtMalloc(&dW[i], wBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ACL_CHECK(aclrtMalloc(&dWScale, wsclBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dKSum, ksumBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dWsp, wspBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(dAHi, aPlaneBytes, hAHi.data(), aPlaneBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dALo, aPlaneBytes, hALo.data(), aPlaneBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dAScale, asBytes, hAScale.data(), asBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    for (uint32_t i = 0; i < rotate; ++i) {
        // All rotate copies hold the same weight, so one w_ksum covers them all.
        ACL_CHECK(aclrtMemcpy(dW[i], wBytes, wDev.data(), wBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ACL_CHECK(aclrtMemcpy(dWScale, wsclBytes, hWScale.data(), wsclBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dKSum, ksumBytes, hKSum.data(), ksumBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t rotIdx = 0;
    auto run_once = [&]() {
        void* w = dW[rotIdx];
        if (rotate > 1) rotIdx = (rotIdx + 1) % rotate;
        launch(blockDim, stream, dAHi, dALo, dAScale, w, dWScale, dKSum, dY, dWsp, Mp, N, K);
    };

    std::cout << "[launch]     blockDim=" << blockDim
              << " (aicore=" << coreNum << ")  tileM=" << tileM
              << " (cube tile " << (2 * tileM) << "x" << kTileN << ")"
              << "  M=" << M << (Mp != M ? " -> padded " : " ")
              << (Mp != M ? std::to_string(Mp) : std::string())
              << "  tiles=" << numTiles
              << "  workspace=" << (double(wspBytes) / (1024.0 * 1024.0)) << " MiB"
              << (rotate > 1 ? "  cold(x" + std::to_string(rotate) + " W copies)" : "")
              << "\n"
              << std::flush;

    bool overall = true;

    auto check_now = [&]() {
        std::vector<uint16_t> hY(yCount);
        ACL_CHECK(aclrtMemcpy(hY.data(), yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        hY.resize(yRealCount);
        auto r = CompareBF16(hY, hYRef, 2.0e-2f, 8.0e-2f);
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

    aclrtFree(dWsp);
    aclrtFree(dY);
    aclrtFree(dKSum);
    aclrtFree(dWScale);
    for (uint32_t i = 0; i < rotate; ++i) aclrtFree(dW[i]);
    aclrtFree(dAScale);
    aclrtFree(dALo);
    aclrtFree(dAHi);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return overall ? 0 : 1;
}

}  // namespace w4a8
