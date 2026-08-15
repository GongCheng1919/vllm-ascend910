// Shared driver for the three gemm_precision_lab ops (bf16 / int8 / int4).
// Each main_<op>.cpp instantiates RunGemm with its launch function + dtype.
//
// Output type OutT: uint16_t (bf16) for gemm_bf16, int32_t for gemm_int8 / gemm_int4.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "acl_utils.h"
#include "data_utils.h"
#include "gemm_common.h"

namespace {

constexpr uint32_t kBlockDim910B4 = 16;

inline void CheckGemmShape(const Args& args, GemmDType dt, uint32_t tileM = 128, uint32_t tileN = 128)
{
    const int M = args.rows, N = args.cols, K = args.k;
    if (M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "M/N/K must be positive\n";
        std::exit(1);
    }
    // 手写流水 kernel: M/N 需为 tile 尺寸的倍数, K 512 倍数 (bf16 的 PHYSICAL_K=512; int8/int4 1024 是其倍数)
    if ((M % tileM) != 0 || (N % tileN) != 0 || (K % 512) != 0) {
        std::cerr << "kernel requires M%" << tileM << "==0, N%" << tileN << "==0, K%512==0 for "
                  << GemmDTypeName(dt) << " (got M=" << M << " N=" << N << " K=" << K << ")\n";
        std::exit(1);
    }
}

template <typename OutT>
inline bool CheckOut(const std::vector<OutT>& got, const std::vector<OutT>& ref, const char* name)
{
    if constexpr (std::is_same_v<OutT, uint16_t>) {
        const CompareReport r = CompareBF16(got, ref, /*atol=*/2.0f, /*rtol=*/1e-2f);
        PrintCheck(name, r);
        if (!r.pass) {
            size_t fb = r.first_bad_idx;
            std::cerr << "[dump] first_bad=" << fb << " got:";
            for (size_t i = fb; i < fb + 8 && i < got.size(); ++i) std::cerr << " 0x" << std::hex << got[i] << std::dec;
            std::cerr << "\n[dump] first_bad=" << fb << " ref:";
            for (size_t i = fb; i < fb + 8 && i < ref.size(); ++i) std::cerr << " 0x" << std::hex << ref[i] << std::dec;
            std::cerr << "\n";
        }
        return r.pass;
    } else {
        size_t mismatches = 0;
        size_t firstBad = SIZE_MAX;
        int64_t maxAbs = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const int64_t d = static_cast<int64_t>(got[i]) - static_cast<int64_t>(ref[i]);
            if (d != 0) {
                if (firstBad == SIZE_MAX) firstBad = i;
                mismatches++;
            }
            if (std::llabs(d) > maxAbs) maxAbs = std::llabs(d);
        }
        CompareReport r{};
        r.pass = (mismatches == 0);
        r.mismatches = mismatches;
        r.max_abs_diff = static_cast<double>(maxAbs);
        r.first_bad_idx = firstBad;
        PrintCheck(name, r);
        if (!r.pass) {
            // DEBUG: dump the first few elements of the bad region and the reference
            size_t region = got.size() / 2;  // [region:] is the second-core slice
            std::cerr << "[dump] bad-region got:";
            for (size_t i = region; i < region + 6 && i < got.size(); ++i) std::cerr << " " << got[i];
            std::cerr << "\n[dump] bad-region ref:";
            for (size_t i = region; i < region + 6 && i < ref.size(); ++i) std::cerr << " " << ref[i];
            std::cerr << "\n";
        }
        return r.pass;
    }
}

}  // namespace

// T: input storage type (uint16_t bf16 / int8_t / uint8_t packed int4)
// OutT: output type (uint16_t bf16 / int32_t)
// LaunchFn: void(uint32_t blockDim, aclrtStream stream, void* a, void* b, void* c,
//                void* workspace, uint32_t M, uint32_t N, uint32_t K)
// GenRef:   std::vector<OutT>(const std::vector<T>& a, const std::vector<T>& b, int M, int N, int K)
template <typename T, typename OutT, GemmDType DT, bool PACKED, typename LaunchFn, typename GenRef>
int RunGemm(int argc, char** argv, const char* opName,
            LaunchFn launch, GenRef genRef,
            uint32_t tileM = 128, uint32_t tileN = 128)
{
    constexpr GemmDType dt = DT;
    constexpr bool packedInput = PACKED;
    Args args = ParseArgs(argc, argv);
    CheckGemmShape(args, dt, tileM, tileN);
    PrintHeader(opName, args);

    const uint32_t M = static_cast<uint32_t>(args.rows);
    const uint32_t N = static_cast<uint32_t>(args.cols);
    const uint32_t K = static_cast<uint32_t>(args.k);
    const size_t aCount = static_cast<size_t>(M) * K;
    const size_t bCount = static_cast<size_t>(N) * K;
    const size_t cCount = static_cast<size_t>(M) * N;

    // ---- 手写流水 kernel (用户 mid_group_gemm_fwd_lab 结构): tileM x tileN tile 流 ----
    const uint32_t numTiles = (M / tileM) * (N / tileN);
    const uint32_t dimUsed = std::max<uint32_t>(1, std::min(kBlockDim910B4, numTiles));
    const size_t wsCount = static_cast<size_t>(dimUsed) * tileM * tileN;

    // ---- host data + CPU reference ----
    std::vector<T> hA, hB;
    std::vector<OutT> hRef;
    if constexpr (dt == GemmDType::BF16) {
        auto a16 = GenerateBF16(aCount, args.seed, 1.0f);
        auto b16 = GenerateBF16(bCount, args.seed + 101, 1.0f);
        hA.assign(reinterpret_cast<T*>(a16.data()), reinterpret_cast<T*>(a16.data()) + aCount);
        hB.assign(reinterpret_cast<T*>(b16.data()), reinterpret_cast<T*>(b16.data()) + bCount);
        if (args.check) hRef = genRef(a16, b16, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    } else {
        auto a8 = GenerateInt8(aCount, args.seed, -8, 7);  // 与用户 mid_group_gemm_fwd_lab host 一致
        auto b8 = GenerateInt8(bCount, args.seed + 101, -8, 7);
        if constexpr (packedInput) {
            // int4: restrict to nibble range [-8,7], then pack 2 per byte
            for (auto& v : a8) v = static_cast<int8_t>((static_cast<int>(v) % 16 + 16) % 16 - 8);
            for (auto& v : b8) v = static_cast<int8_t>((static_cast<int>(v) % 16 + 16) % 16 - 8);
            std::vector<uint8_t> aPack = PackInt4(a8);
            std::vector<uint8_t> bPack = PackInt4(b8);
            hA.assign(reinterpret_cast<T*>(aPack.data()), reinterpret_cast<T*>(aPack.data()) + aPack.size());
            hB.assign(reinterpret_cast<T*>(bPack.data()), reinterpret_cast<T*>(bPack.data()) + bPack.size());
            if (args.check) hRef = genRef(aPack, bPack, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
        } else {
            hA.assign(reinterpret_cast<T*>(a8.data()), reinterpret_cast<T*>(a8.data()) + aCount);
            hB.assign(reinterpret_cast<T*>(b8.data()), reinterpret_cast<T*>(b8.data()) + bCount);
            if (args.check) hRef = genRef(a8, b8, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
        }
    }

    // int4 (packedInput): 打包后字节 = 元素数/2
    const size_t aBytes = (packedInput ? aCount / 2 : aCount) * sizeof(T);
    const size_t bBytes = (packedInput ? bCount / 2 : bCount) * sizeof(T);
    const size_t cBytes = cCount * sizeof(OutT);
    const size_t wsBytes = wsCount * sizeof(int32_t);

    // ---- ACL setup ----
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtContext ctx;
    aclrtStream stream;
    ACL_CHECK(aclrtCreateContext(&ctx, 0));
    ACL_CHECK(aclrtCreateStream(&stream));

    void* dA = nullptr; void* dB = nullptr; void* dC = nullptr;
    void* dWs = nullptr;
    ACL_CHECK(aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dWs, wsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dB, bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- launch ----
    auto run_once = [&]() {
        launch(dimUsed, stream, dA, dB, dC, dWs,
               static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K));
    };
    run_once();
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // ---- check ----
    bool overall = true;
    if (args.check) {
        std::vector<OutT> hC(cCount);
        ACL_CHECK(aclrtMemcpy(hC.data(), cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        overall = CheckOut<OutT>(hC, hRef, "c");
        // DEBUG: workspace (Fixpipe 输出，AIC 写 / AIV 读) 落盘后对比每个 block
        // 最后一次分到的 tile 的期望值 (AIV 对 int8/int4 是纯直通，workspace
        // 应该精确等于 hRef 里那个 tile 的切片)。用来区分：
        //   workspace 本身就错   -> AIC 侧 (Mmad/Fixpipe/寻址) 的 bug
        //   workspace 是对的但 c 错 -> AIV 侧 (读 workspace 的时序/竞争) 的 bug
        if constexpr (std::is_same_v<OutT, int32_t>) {
            const uint32_t numNTilesDbg = N / tileN;
            std::vector<int32_t> hWs(wsCount);
            ACL_CHECK(aclrtMemcpy(hWs.data(), wsBytes, dWs, wsBytes, ACL_MEMCPY_DEVICE_TO_HOST));
            uint32_t blocksBad = 0, blocksChecked = 0;
            for (uint32_t b = 0; b < dimUsed; ++b) {
                // 该 block 最后一次分到的 tileId = b + k*dimUsed 中最大的 < numTiles 者
                if (b >= numTiles) continue;
                uint32_t lastTileId = b;
                while (lastTileId + dimUsed < numTiles) lastTileId += dimUsed;
                const uint32_t mTile = lastTileId / numNTilesDbg;
                const uint32_t nTile = lastTileId % numNTilesDbg;
                size_t mismatches = 0;
                int firstBadI = -1, firstBadJ = -1;
                int32_t gotV = 0, refV = 0;
                for (uint32_t i = 0; i < tileM; ++i) {
                    for (uint32_t j = 0; j < tileN; ++j) {
                        const size_t wsIdx = (size_t)b * tileM * tileN + (size_t)i * tileN + j;
                        const size_t refIdx = (size_t)(mTile * tileM + i) * N + (nTile * tileN + j);
                        if (hWs[wsIdx] != static_cast<int32_t>(hRef[refIdx])) {
                            if (firstBadI < 0) {
                                firstBadI = (int)i; firstBadJ = (int)j;
                                gotV = hWs[wsIdx]; refV = static_cast<int32_t>(hRef[refIdx]);
                            }
                            mismatches++;
                        }
                    }
                }
                blocksChecked++;
                if (mismatches > 0) {
                    blocksBad++;
                    std::cerr << "[ws-check] block=" << b << " lastTile=(" << mTile << "," << nTile
                              << ") mismatches=" << mismatches << "/" << (tileM * tileN)
                              << " first_bad=(" << firstBadI << "," << firstBadJ
                              << ") ws=" << gotV << " ref=" << refV << "\n";
                }
            }
            std::cerr << "[ws-check] summary: " << blocksBad << "/" << blocksChecked
                      << " blocks have wrong workspace (AIC-side bug if >0; "
                      << "AIV/timing-side bug if 0 but final c still wrong)\n";
        }
    }

    // ---- bench ----
    if (args.profile) {
        const double avgUs = BenchmarkUs([&]() {
            run_once();
            ACL_CHECK(aclrtSynchronizeStream(stream));
        }, args.warmup, args.repeat);
        const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        const double tflops = ops / avgUs / 1.0e6;
        PrintTime(avgUs, -1.0, tflops);
        std::cout << "[tflops]     " << std::fixed << std::setprecision(1) << tflops
                  << " TFLOPS  (M=" << M << " N=" << N << " K=" << K
                  << ", " << GemmDTypeName(dt) << ", " << args.repeat << " iters)\n";
    }

    PrintStatus(overall);

    aclrtFree(dA); aclrtFree(dB); aclrtFree(dC); aclrtFree(dWs);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return overall ? 0 : 1;
}
