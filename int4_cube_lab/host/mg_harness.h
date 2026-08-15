#pragma once
// Host harness for the mid-group W4A8 GEMM kernels.
//
// P0 probe mode: the per-group scales are DEGENERATE -- as[g,m] = as[m] and
// ws[g,n] = ws[n] for every g.  The kernel then computes the same value as the
// per-channel W4A8 kernel (up to fp32 accumulation order), so the existing CPU
// reference still applies and any correctness failure is a pipeline bug rather
// than a quantisation-model mismatch.  What is NOT degenerate is the amount of
// work: every group still does its own Fixpipe, CrossCore handshake, scale
// load and fp32 rescale+accumulate, so the measured cost is the real
// mid-group cost.  P3 will feed real per-group scales through the same kernel.
//
// P3 (default): REAL per-group scales, real per-group zero points, and a real
// mid-group reference (ReferenceMidGroupGemm).  This is what actually verifies
// the maths.  `--degenerate` restores the P0 probe path above, which is kept
// only so the P0/P3 cost numbers (midgroup_cost_v4 / v5) stay reproducible --
// it is NOT a correctness mode for asym, because with w_zero = 0 the whole
// zero-point term drops out of the result.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl_utils.h"
#include "data_utils.h"
#include "bf16.h"
#include "gemm_ref.h"

namespace mg {

constexpr uint32_t kTileN = 128;
constexpr uint32_t kFallbackBlocks = 20;
constexpr double   kMinSnrDb = 40.0;
constexpr size_t   kColdFootprint = 1024ull * 1024 * 1024;
constexpr uint32_t kMaxRotate = 16;

// (blockDim, stream, a_hi, a_lo, a_scale, w, w_scale, w_ksum, w_zero, a_ksum,
//  y, workspace, M, N, K)
using LaunchFn = std::function<void(uint32_t, aclrtStream, void*, void*, void*,
                                    void*, void*, void*, void*, void*, void*, void*,
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

inline int Run(int argc, char** argv, uint32_t tileM, uint32_t gk, uint32_t innerK,
               const char* opName, LaunchFn launch, bool asym = false)
{
    Args args = ParseArgs(argc, argv);
    const uint32_t M = static_cast<uint32_t>(args.rows);
    const uint32_t N = static_cast<uint32_t>(args.cols);
    const uint32_t K = static_cast<uint32_t>(args.k);

    if (M == 0 || N == 0 || K == 0 || (N % kTileN) || (K % gk) || (K % innerK)) {
        std::cerr << "shape constraint: N multiple of 128; K multiple of GK=" << gk
                  << " and of innerK=" << innerK << "\n";
        return 1;
    }
    const uint32_t Mp = ((M + tileM - 1) / tileM) * tileM;
    if (K / 2 > 65535u) {
        std::cerr << "K/2 must fit in uint16 (Nd2Nz srcDValue)\n";
        return 1;
    }
    PrintHeader(opName, args);

    const uint32_t numG = K / gk;
    const size_t aCount = static_cast<size_t>(Mp) * K;
    const size_t wCount = static_cast<size_t>(N) * K;
    const size_t yCount = static_cast<size_t>(Mp) * N;
    const size_t yRealCount = static_cast<size_t>(M) * N;
    const size_t numTiles = static_cast<size_t>(Mp / tileM) * (N / kTileN);

    // --load-dir: the inputs come from fakequant_lab, which quantised them with
    // its own primitives.  That makes the comparison a real seam check between
    // the algorithm and the kernel, rather than between the kernel and a
    // reference that shares this file's conventions.
    const bool loading = !args.load_dir.empty();
    std::vector<int8_t> hA = loading
        ? ReadBin<int8_t>(args.load_dir + "/a_q.bin", aCount)
        : GenerateInt8(aCount, args.seed, -128, 127);
    std::vector<int8_t> hW = loading
        ? ReadBin<int8_t>(args.load_dir + "/w_q.bin", wCount)
        : GenerateInt8(wCount, args.seed + 101, -8, 7);

    std::vector<int8_t> hAHi, hALo;
    SplitInt8ToInt4Planes(hA, Mp, K, hAHi, hALo);
    const std::vector<int8_t> wDev = PackInt4(hW, N, K);
    const std::vector<int32_t> hKSum = WeightKSumPerGroup(hW, N, K, gk);

    // a_ksum is always the true negated row sum: it is cheap, and feeding it
    // real data means the extra MulAddDst always runs on representative
    // operands even in --degenerate cost runs.
    const std::vector<int32_t> hAKsumG = ActKSumNegPerGroup(hA, Mp, K, gk);

    std::vector<uint16_t> hAScaleG(static_cast<size_t>(numG) * Mp);
    std::vector<uint16_t> hWScaleG(static_cast<size_t>(numG) * N);
    std::vector<uint16_t> hWZeroG(static_cast<size_t>(numG) * N, 0);

    if (loading) {
        hAScaleG = ReadBin<uint16_t>(args.load_dir + "/a_scale_g.bin", (size_t)numG * Mp);
        hWScaleG = ReadBin<uint16_t>(args.load_dir + "/w_scale_g.bin", (size_t)numG * N);
        if (asym) {
            hWZeroG = ReadBin<uint16_t>(args.load_dir + "/w_zero_g.bin", (size_t)numG * N);
        }
    } else if (args.degenerate) {
        // P0 probe path: every group repeats one per-channel scale and the zero
        // point is 0, so the per-channel reference still applies.
        const std::vector<uint16_t> hAScale =
            GenerateBF16(Mp, args.seed + 202, 0.0005f, 0.0020f);
        const std::vector<uint16_t> hWScale =
            GenerateBF16(N, args.seed + 303, 0.0005f, 0.0020f);
        for (uint32_t g = 0; g < numG; ++g) {
            std::copy(hAScale.begin(), hAScale.end(), hAScaleG.begin() + (size_t)g * Mp);
            std::copy(hWScale.begin(), hWScale.end(), hWScaleG.begin() + (size_t)g * N);
        }
    } else {
        // Real mid-group: an independent scale per (group, row/channel).
        hAScaleG = GenerateBF16(static_cast<size_t>(numG) * Mp, args.seed + 202,
                                0.0005f, 0.0020f);
        hWScaleG = GenerateBF16(static_cast<size_t>(numG) * N, args.seed + 303,
                                0.0005f, 0.0020f);
        if (asym) {
            // Integer code-unit zero points in [-8,7], the range quant.py's
            // `zero = qmin - round(mn/scale)` actually produces for int4.
            const std::vector<int8_t> z =
                GenerateInt8(static_cast<size_t>(numG) * N, args.seed + 505, -8, 7);
            for (size_t i = 0; i < z.size(); ++i) {
                hWZeroG[i] = FP32ToBF16(static_cast<float>(z[i]));
            }
        }
    }

    std::vector<uint16_t> hYRef;
    if (args.check || args.stress > 0) {
        std::cout << "[debug]      CPU reference (" << M << "x" << N << "x" << K
                  << (args.degenerate ? ", degenerate" : ", mid-group")
                  << (asym && !args.degenerate ? ", asym" : "") << ")...\n" << std::flush;
        if (args.degenerate) {
            std::vector<uint16_t> aS(hAScaleG.begin(), hAScaleG.begin() + Mp);
            std::vector<uint16_t> wS(hWScaleG.begin(), hWScaleG.begin() + N);
            ReferencePerchannelGemm(hA, aS, hW, wS, hYRef, M, N, K);
        } else {
            ReferenceMidGroupGemm(hA, hAScaleG, hW, hWScaleG,
                                  asym ? hWZeroG : std::vector<uint16_t>{},
                                  hYRef, M, N, K, gk);
        }
        std::cout << "[debug]      CPU reference done\n" << std::flush;
    }

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
    const size_t asBytes     = hAScaleG.size() * sizeof(uint16_t);
    const size_t wsclBytes   = hWScaleG.size() * sizeof(uint16_t);
    const size_t ksumBytes   = hKSum.size() * sizeof(int32_t);
    const size_t akBytes     = hAKsumG.size() * sizeof(int32_t);
    const size_t yBytes      = yCount * sizeof(uint16_t);
    // two slots per block: the kernel double-buffers the flush target
    const size_t wspBytes = static_cast<size_t>(blockDim) * 2 * 2 * tileM * kTileN * sizeof(int32_t);

    const uint32_t rotate = args.cold
        ? std::min<uint32_t>(kMaxRotate,
              std::max<uint32_t>(2, static_cast<uint32_t>(
                  (kColdFootprint + wBytes - 1) / wBytes)))
        : 1;

    void *dAHi = nullptr, *dALo = nullptr, *dAScale = nullptr;
    void *dWScale = nullptr, *dKSum = nullptr, *dY = nullptr, *dWsp = nullptr;
    void *dWZero = nullptr, *dAKsum = nullptr;
    std::vector<void*> dW(rotate, nullptr);
    ACL_CHECK(aclrtMalloc(&dAHi, aPlaneBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dALo, aPlaneBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dAScale, asBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    for (uint32_t i = 0; i < rotate; ++i) {
        ACL_CHECK(aclrtMalloc(&dW[i], wBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ACL_CHECK(aclrtMalloc(&dWScale, wsclBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dKSum, ksumBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dWZero, wsclBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dAKsum, akBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dWsp, wspBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(dAHi, aPlaneBytes, hAHi.data(), aPlaneBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dALo, aPlaneBytes, hALo.data(), aPlaneBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dAScale, asBytes, hAScaleG.data(), asBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    for (uint32_t i = 0; i < rotate; ++i) {
        ACL_CHECK(aclrtMemcpy(dW[i], wBytes, wDev.data(), wBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ACL_CHECK(aclrtMemcpy(dWScale, wsclBytes, hWScaleG.data(), wsclBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dKSum, ksumBytes, hKSum.data(), ksumBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dWZero, wsclBytes, hWZeroG.data(), wsclBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dAKsum, akBytes, hAKsumG.data(), akBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t rotIdx = 0;
    auto run_once = [&]() {
        void* w = dW[rotIdx];
        if (rotate > 1) rotIdx = (rotIdx + 1) % rotate;
        launch(blockDim, stream, dAHi, dALo, dAScale, w, dWScale, dKSum,
               dWZero, dAKsum, dY, dWsp, Mp, N, K);
    };

    std::cout << "[launch]     blockDim=" << blockDim
              << " (aicore=" << coreNum << ")  tileM=" << tileM
              << " (cube tile " << (2 * tileM) << "x" << kTileN << ")"
              << "  GK=" << gk << " groups=" << numG
              << "  M=" << M << (Mp != M ? " -> padded " : " ")
              << (Mp != M ? std::to_string(Mp) : std::string())
              << "  tiles=" << numTiles
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

    // --dump writes everything the seam check needs: the exact inputs, the CPU
    // reference and the device output, so fakequant_lab can recompute y with its
    // own quantisation primitives and compare.  Written before the timing loop
    // so a dump run stays cheap.
    if (args.dump) {
        run_once();
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<uint16_t> hY(yCount);
        ACL_CHECK(aclrtMemcpy(hY.data(), yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        hY.resize(yRealCount);
        const std::string d = args.out_dir;
        std::string mk = "mkdir -p '" + d + "'";
        if (std::system(mk.c_str()) != 0) { std::cerr << "mkdir failed\n"; return 1; }
        WriteBin(d + "/a_q.bin", hA);
        WriteBin(d + "/w_q.bin", hW);
        WriteBin(d + "/a_scale_g.bin", hAScaleG);
        WriteBin(d + "/w_scale_g.bin", hWScaleG);
        WriteBin(d + "/w_zero_g.bin", hWZeroG);
        WriteBin(d + "/y_dev.bin", hY);
        std::ofstream meta(d + "/meta.txt");
        meta << "M " << M << "\nMp " << Mp << "\nN " << N << "\nK " << K
             << "\ngk " << gk << "\nasym " << (asym ? 1 : 0) << "\n";
        meta.close();
        std::cout << "[dump]       wrote inputs + y_dev to " << d << "\n" << std::flush;
    }

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
    aclrtFree(dAKsum);
    aclrtFree(dWZero);
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

}  // namespace mg
