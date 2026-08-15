// main_gemm.cpp — host harness for the three raw GEMM kernels.
// Usage:
//   gemm_test --op s4s4|s8s8|bf16 --m 128 --n 5120 --k 5120 [--check] [--perf] [--warmup 10] [--repeat 100] [--seed 7]
//   gemm_test --sweep                              (runs the full shape matrix for all three ops)
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <cstdint>
#include <cstdlib>

#include "acl/acl.h"
#include "aclrtlaunch_gemm_s4s4.h"
#include "aclrtlaunch_gemm_s8s8.h"
#include "aclrtlaunch_gemm_bf16.h"
#include "ref_gemm.h"

#define ACL_CHECK(expr)                                                     \
    do {                                                                    \
        aclError __e = (expr);                                              \
        if (__e != ACL_SUCCESS) {                                           \
            std::cerr << "ACL error " << __e << " at " << __FILE__ << ":"  \
                      << __LINE__ << " in " << #expr << std::endl;          \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

struct Args {
    std::string op = "s8s8";
    uint32_t m = 128, n = 5120, k = 5120;
    bool check = false, perf = false, sweep = false;
    int warmup = 10, repeat = 100;
    uint32_t seed = 7;
};

static Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--op") a.op = next();
        else if (s == "--m") a.m = (uint32_t)std::stoul(next());
        else if (s == "--n") a.n = (uint32_t)std::stoul(next());
        else if (s == "--k") a.k = (uint32_t)std::stoul(next());
        else if (s == "--check") a.check = true;
        else if (s == "--perf") a.perf = true;
        else if (s == "--sweep") a.sweep = true;
        else if (s == "--warmup") a.warmup = std::stoi(next());
        else if (s == "--repeat") a.repeat = std::stoi(next());
        else if (s == "--seed") a.seed = (uint32_t)std::stoul(next());
        else {
            std::cerr << "unknown arg: " << s << "\n";
            std::exit(1);
        }
    }
    return a;
}

static const char* OpName(const std::string& op) {
    if (op == "s4s4") return "s4s4";
    if (op == "s8s8") return "s8s8";
    return "bf16";
}

static void RunOp(const std::string& op, uint32_t M, uint32_t N, uint32_t K, const Args& args,
                  aclrtStream stream, bool verbose) {
    // shape validation
    if (M % 16 != 0 || N % 128 != 0 || K % 256 != 0) {
        std::cerr << "shape must satisfy M%16==0, N%128==0, K%256==0 (got " << M << "x" << N << "x" << K << ")\n";
        std::exit(1);
    }

    std::mt19937 rng(args.seed);
    std::uniform_int_distribution<int> u4(-8, 7);
    std::uniform_int_distribution<int> u8(-127, 127);
    std::uniform_real_distribution<float> uf(-2.0f, 2.0f);

    const bool isInt = (op != "bf16");
    const size_t aElems = (size_t)M * K;
    const size_t bElems = (size_t)K * N;
    const size_t cElems = (size_t)M * N;

    std::vector<uint8_t> hA, hB;
    std::vector<int32_t> hCrefI;
    std::vector<float> hCrefF;
    std::vector<uint16_t> hABf, hBBf;

    if (op == "s4s4") {
        std::vector<int8_t> a(aElems), b(bElems);
        for (auto& v : a) v = (int8_t)u4(rng);
        for (auto& v : b) v = (int8_t)u4(rng);
        PackInt4(a, hA);
        PackInt4(b, hB);
        hCrefI = RefGemmInt("s4s4", a, b, M, N, K);
    } else if (op == "s8s8") {
        hA.resize(aElems); hB.resize(bElems);
        for (size_t i = 0; i < aElems; i++) hA[i] = (uint8_t)(int8_t)u8(rng);
        for (size_t i = 0; i < bElems; i++) hB[i] = (uint8_t)(int8_t)u8(rng);
        std::vector<int8_t> a(aElems), b(bElems);
        for (size_t i = 0; i < aElems; i++) a[i] = (int8_t)hA[i];
        for (size_t i = 0; i < bElems; i++) b[i] = (int8_t)hB[i];
        hCrefI = RefGemmInt("s8s8", a, b, M, N, K);
    } else {
        hABf.resize(aElems); hBBf.resize(bElems);
        for (auto& v : hABf) v = FP32ToBF16(uf(rng));
        for (auto& v : hBBf) v = FP32ToBF16(uf(rng));
        hCrefF = RefGemmBf16(hABf, hBBf, M, N, K);
    }

    size_t aBytes = hA.size() * sizeof(uint8_t);
    size_t bBytes = hB.size() * sizeof(uint8_t);
    size_t cBytes = cElems * (isInt ? sizeof(int32_t) : sizeof(float));
    if (!isInt) { aBytes = hABf.size() * sizeof(uint16_t); bBytes = hBBf.size() * sizeof(uint16_t); }

    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    ACL_CHECK(aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    if (isInt) {
        ACL_CHECK(aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(dB, bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        ACL_CHECK(aclrtMemcpy(dA, aBytes, hABf.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(dB, bBytes, hBBf.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    const uint32_t blockDim = (N / 128 < 20) ? (N / 128) : 20;
    auto launch = [&]() {
        if (op == "s4s4") ACL_CHECK(aclrtlaunch_gemm_s4s4(blockDim, stream, dA, dB, dC, M, N, K));
        else if (op == "s8s8") ACL_CHECK(aclrtlaunch_gemm_s8s8(blockDim, stream, dA, dB, dC, M, N, K));
        else ACL_CHECK(aclrtlaunch_gemm_bf16(blockDim, stream, dA, dB, dC, M, N, K));
    };

    launch();
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // ---- correctness ----
    bool pass = true;
    if (args.check) {
        if (isInt) {
            std::vector<int32_t> hC(cElems);
            ACL_CHECK(aclrtMemcpy(hC.data(), cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST));
            int64_t worst = 0;
            size_t bad = 0;
            for (size_t i = 0; i < cElems; i++) {
                int64_t d = std::llabs((int64_t)hC[i] - (int64_t)hCrefI[i]);
                if (d > worst) worst = d;
                if (d != 0) bad++;
            }
            pass = (bad == 0);
            if (verbose)
                std::cout << "[check] " << OpName(op) << " M" << M << " N" << N << " K" << K
                          << " bad=" << bad << "/" << cElems << " worst=" << worst
                          << (pass ? "  PASS" : "  FAIL") << std::endl;
        } else {
            std::vector<float> hC(cElems);
            ACL_CHECK(aclrtMemcpy(hC.data(), cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST));
            double maxAbs = 0, maxRel = 0;
            size_t bad = 0;
            for (size_t i = 0; i < cElems; i++) {
                double err = std::fabs((double)hC[i] - (double)hCrefF[i]);
                double mag = std::fabs((double)hCrefF[i]);
                maxAbs = std::max(maxAbs, err);
                if (mag > 0) maxRel = std::max(maxRel, err / mag);
                if (err > 0.05 * std::max(1.0, mag)) bad++;
            }
            pass = (bad == 0);
            if (verbose)
                std::cout << "[check] bf16 M" << M << " N" << N << " K" << K
                          << " bad=" << bad << "/" << cElems
                          << " maxAbs=" << maxAbs << " maxRel=" << maxRel
                          << (pass ? "  PASS" : "  FAIL") << std::endl;
        }
    }

    // ---- perf ----
    if (args.perf) {
        for (int i = 0; i < args.warmup; i++) launch();
        ACL_CHECK(aclrtSynchronizeStream(stream));
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < args.repeat; i++) launch();
        ACL_CHECK(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double tflops = 2.0 * (double)M * N * K * args.repeat / sec / 1e12;
        double us = sec / args.repeat * 1e6;
        if (verbose)
            std::cout << "[perf] " << OpName(op) << " M" << M << " N" << N << " K" << K
                      << " t=" << us << " us  " << tflops << " TFLOPS  (" << pass << ")" << std::endl;
        std::cout << OpName(op) << "," << M << "," << N << "," << K << "," << us << "," << tflops << std::endl;
    }

    ACL_CHECK(aclrtFree(dA));
    ACL_CHECK(aclrtFree(dB));
    ACL_CHECK(aclrtFree(dC));
}

int main(int argc, char** argv) {
    Args args = ParseArgs(argc, argv);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtContext ctx;
    aclrtStream stream;
    ACL_CHECK(aclrtCreateContext(&ctx, 0));
    ACL_CHECK(aclrtCreateStream(&stream));

    if (args.sweep) {
        const std::vector<uint32_t> Ms = {16, 32, 64, 128};
        const std::vector<std::pair<uint32_t, uint32_t>> NKs = {{512, 512}, {5120, 5120}, {5120, 27648}};
        std::vector<std::string> ops = {"s4s4", "s8s8", "bf16"};
        // warmup the device first
        for (uint32_t i = 0; i < 5; i++) {
            // trivial launch to settle clocks
        }
        for (const auto& op : ops) {
            for (auto M : Ms) {
                for (auto [N, K] : NKs) {
                    RunOp(op, M, N, K, args, stream, /*verbose=*/true);
                }
            }
        }
    } else {
        RunOp(args.op, args.m, args.n, args.k, args, stream, /*verbose=*/true);
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtDestroyContext(ctx));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
