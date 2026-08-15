#pragma once
// Host-side helpers for the gemm_precision_lab:
//   * ComputeGemmTiling  — TCubeTiling via matmul_tiling::MultiCoreMatmulTiling
//   * int4 pack/unpack   — ND [.., K] with 2 nibbles per byte (even k -> low nibble)
//   * CPU references     — C = A @ B^T in fp32/int32, cast to bf16

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "adv_api/matmul/bmm_tiling.h"
#include "bf16.h"

enum class GemmDType { BF16 = 0, INT8 = 1, INT4 = 2 };

inline const char* GemmDTypeName(GemmDType dt)
{
    switch (dt) {
        case GemmDType::BF16: return "bf16";
        case GemmDType::INT8: return "int8";
        case GemmDType::INT4: return "int4";
    }
    return "?";
}

// Tile-flow plan for C = A[M,K] @ B^T[N,K] -> C[M,N].
// M 方向 tile 流: 输出按 tileM 行一段划分 tilesM 段, 20 核 grid-stride 交错领取
// (负载均衡, 最后一批最多差 1 段)。N 方向全宽 (C 行 stride = N, 与 tiling 一致)。
// M 按 tile 粒度 padding: A/C 的行数 pad 到 tilesM*tileM (pad 行补 0)。
struct GemmPlan {
    AscendC::tiling::TCubeTiling tileTiling;  // 单段 tiling (SetDim(1), 形状 tileM x N)
    int32_t tileM = 0;   // 每段 M 行 (128 或 M)
    int32_t tilesM = 0;  // ceil(M/tileM)
    int32_t padM = 0;    // tilesM*tileM
    int32_t dimUsed = 0; // min(coreNum, tilesM)
};

inline GemmPlan ComputeGemmPlan(GemmDType dt, int M, int N, int K, int coreNum)
{
    using namespace matmul_tiling;

    DataType aDt = DataType::DT_BF16;
    if (dt == GemmDType::INT8) aDt = DataType::DT_INT8;
    if (dt == GemmDType::INT4) aDt = DataType::DT_INT4;

    GemmPlan plan;
    plan.tileM = std::min(128, M);  // 16 对齐 (M 为 16 倍数; M>=128 时取 128)
    plan.tilesM = (M + plan.tileM - 1) / plan.tileM;
    plan.padM = plan.tilesM * plan.tileM;
    plan.dimUsed = std::min(coreNum, plan.tilesM);

    MultiCoreMatmulTiling mm;
    mm.SetDim(1);
    mm.SetAType(TPosition::GM, CubeFormat::ND, aDt, false);
    mm.SetBType(TPosition::GM, CubeFormat::ND, aDt, true);
    mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16);
    mm.SetShape(plan.tileM, N, K);
    mm.SetOrgShape(plan.tileM, N, K);
    mm.SetBufferSpace(512 * 1024, 128 * 1024, 192 * 1024);

    AscendC::tiling::TCubeTiling& t = plan.tileTiling;
    if (mm.GetTiling(t) != 0) {
        std::cerr << "[tiling] GetTiling failed: " << GemmDTypeName(dt)
                  << " M=" << M << " N=" << N << " K=" << K << " cores=" << coreNum << "\n";
        std::exit(1);
    }
    std::fprintf(stderr,
                 "[plan] %s M=%d N=%d K=%d tileM=%d tilesM=%d padM=%d dimUsed=%d "
                 "tileTiling sM=%d sN=%d sK=%d bM=%d bN=%d bK=%d dA1=%d dB1=%d stM=%d stN=%d stKa=%d stKb=%d\n",
                 GemmDTypeName(dt), M, N, K, plan.tileM, plan.tilesM, plan.padM, plan.dimUsed,
                 t.singleCoreM, t.singleCoreN, t.singleCoreK, t.baseM, t.baseN, t.baseK,
                 t.depthA1, t.depthB1, t.stepM, t.stepN, t.stepKa, t.stepKb);
    return plan;
}

// ---- int4 ND packing: byte contains elements 2j (low nibble) and 2j+1 (high nibble).
inline std::vector<uint8_t> PackInt4(const std::vector<int8_t>& vals)
{
    std::vector<uint8_t> packed(vals.size() / 2);
    for (size_t j = 0; j < packed.size(); ++j) {
        const int lo = static_cast<int>(vals[2 * j]) & 0xF;
        const int hi = static_cast<int>(vals[2 * j + 1]) & 0xF;
        packed[j] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return packed;
}

// DEBUG: nibble 交换版 (even k -> high nibble)
inline std::vector<uint8_t> PackInt4Swapped(const std::vector<int8_t>& vals)
{
    std::vector<uint8_t> packed(vals.size() / 2);
    for (size_t j = 0; j < packed.size(); ++j) {
        const int lo = static_cast<int>(vals[2 * j]) & 0xF;
        const int hi = static_cast<int>(vals[2 * j + 1]) & 0xF;
        packed[j] = static_cast<uint8_t>((lo << 4) | hi);
    }
    return packed;
}

inline int UnpackInt4Nibble(uint8_t byte, bool high)
{
    int v = high ? ((byte >> 4) & 0xF) : (byte & 0xF);
    return (v >= 8) ? v - 16 : v;  // sign-extend
}

// ---- CPU references (C = A @ B^T, fp32/int32 accumulate -> bf16 out) ----
inline std::vector<uint16_t> RefBf16Gemm(const std::vector<uint16_t>& a,
                                         const std::vector<uint16_t>& b,
                                         int M, int N, int K)
{
    std::vector<uint16_t> c(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(BF16ToFP32(a[static_cast<size_t>(m) * K + k])) *
                       static_cast<double>(BF16ToFP32(b[static_cast<size_t>(n) * K + k]));
            }
            c[static_cast<size_t>(m) * N + n] = FP32ToBF16(static_cast<float>(acc));
        }
    }
    return c;
}

inline std::vector<int32_t> RefIntGemm(const std::vector<int8_t>& a,
                                        const std::vector<int8_t>& b,
                                        int M, int N, int K)
{
    std::vector<int32_t> c(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int64_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int64_t>(a[static_cast<size_t>(m) * K + k]) *
                       static_cast<int64_t>(b[static_cast<size_t>(n) * K + k]);
            }
            c[static_cast<size_t>(m) * N + n] = static_cast<int32_t>(acc);
        }
    }
    return c;
}

inline std::vector<int32_t> RefInt4Gemm(const std::vector<uint8_t>& aPacked,
                                        const std::vector<uint8_t>& bPacked,
                                        int M, int N, int K)
{
    std::vector<int32_t> c(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int64_t acc = 0;
            for (int k = 0; k < K; ++k) {
                const size_t byteIdx = (static_cast<size_t>(m) * K + k) / 2;
                const bool high = (k & 1) != 0;
                const int av = UnpackInt4Nibble(aPacked[byteIdx], high);
                const size_t bByteIdx = (static_cast<size_t>(n) * K + k) / 2;
                const int bv = UnpackInt4Nibble(bPacked[bByteIdx], high);
                acc += static_cast<int64_t>(av) * bv;
            }
            c[static_cast<size_t>(m) * N + n] = static_cast<int32_t>(acc);
        }
    }
    return c;
}
