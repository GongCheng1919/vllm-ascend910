#include "gemm_ref.h"

#include <cstdlib>
#include <iostream>

#include "bf16.h"

std::vector<int8_t> PackInt4(const std::vector<int8_t>& v, size_t rows, size_t cols)
{
    if ((cols % 2) != 0) {
        std::cerr << "PackInt4: cols must be even\n";
        std::exit(1);
    }
    std::vector<int8_t> packed(rows * cols / 2);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; c += 2) {
            const uint8_t lo = static_cast<uint8_t>(v[r * cols + c]) & 0x0F;
            const uint8_t hi = static_cast<uint8_t>(v[r * cols + c + 1]) & 0x0F;
            packed[(r * cols + c) / 2] = static_cast<int8_t>(lo | (hi << 4));
        }
    }
    return packed;
}

void SplitInt8ToInt4Planes(const std::vector<int8_t>& a, size_t rows, size_t cols,
                           std::vector<int8_t>& hi_packed,
                           std::vector<int8_t>& lo_packed)
{
    if ((cols % 2) != 0) {
        std::cerr << "SplitInt8ToInt4Planes: cols must be even\n";
        std::exit(1);
    }
    hi_packed.assign(rows * cols / 2, 0);
    lo_packed.assign(rows * cols / 2, 0);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; c += 2) {
            const int8_t e0 = a[r * cols + c];
            const int8_t e1 = a[r * cols + c + 1];
            const uint8_t h0 = static_cast<uint8_t>(e0 >> 4) & 0x0F;
            const uint8_t h1 = static_cast<uint8_t>(e1 >> 4) & 0x0F;
            const uint8_t l0 = static_cast<uint8_t>(e0 ^ 8) & 0x0F;
            const uint8_t l1 = static_cast<uint8_t>(e1 ^ 8) & 0x0F;
            hi_packed[(r * cols + c) / 2] = static_cast<int8_t>(h0 | (h1 << 4));
            lo_packed[(r * cols + c) / 2] = static_cast<int8_t>(l0 | (l1 << 4));
        }
    }
}

std::vector<int32_t> WeightKSum(const std::vector<int8_t>& w, size_t rows, size_t cols)
{
    std::vector<int32_t> ksum(rows, 0);
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < static_cast<int64_t>(rows); ++n) {
        int32_t acc = 0;
        const int8_t* row = w.data() + static_cast<size_t>(n) * cols;
        for (size_t k = 0; k < cols; ++k) acc += static_cast<int32_t>(row[k]);
        ksum[n] = acc;
    }
    return ksum;
}

std::vector<int32_t> WeightKSumPerGroup(const std::vector<int8_t>& w,
                                        size_t rows, size_t cols, size_t gk)
{
    if (gk == 0 || (cols % gk) != 0) {
        std::cerr << "WeightKSumPerGroup: cols must be a multiple of gk\n";
        std::exit(1);
    }
    const size_t numG = cols / gk;
    std::vector<int32_t> ksum(numG * rows, 0);
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < static_cast<int64_t>(rows); ++n) {
        const int8_t* row = w.data() + static_cast<size_t>(n) * cols;
        for (size_t g = 0; g < numG; ++g) {
            int32_t acc = 0;
            for (size_t k = g * gk; k < (g + 1) * gk; ++k) acc += static_cast<int32_t>(row[k]);
            ksum[g * rows + static_cast<size_t>(n)] = acc;
        }
    }
    return ksum;
}

void ReferencePerchannelGemm(
    const std::vector<int8_t>& a_q,
    const std::vector<uint16_t>& a_scale_bf16,
    const std::vector<int8_t>& w_q,
    const std::vector<uint16_t>& w_scale_bf16,
    std::vector<uint16_t>& y_bf16,
    uint32_t M, uint32_t N, uint32_t K)
{
    y_bf16.assign(static_cast<size_t>(M) * N, 0);

#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < static_cast<int64_t>(M); ++m) {
        const float as = BF16ToFP32(a_scale_bf16[m]);
        const int8_t* arow = a_q.data() + static_cast<size_t>(m) * K;
        for (uint32_t n = 0; n < N; ++n) {
            const int8_t* wrow = w_q.data() + static_cast<size_t>(n) * K;
            int32_t acc = 0;
            for (uint32_t k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(arow[k]) * static_cast<int32_t>(wrow[k]);
            }
            const float ws = BF16ToFP32(w_scale_bf16[n]);
            y_bf16[static_cast<size_t>(m) * N + n] =
                FP32ToBF16(static_cast<float>(acc) * as * ws);
        }
    }
}

std::vector<int32_t> ActKSumNegPerGroup(const std::vector<int8_t>& a,
                                        size_t rows, size_t cols, size_t gk)
{
    const size_t G = cols / gk;
    std::vector<int32_t> out(G * rows, 0);
    for (size_t g = 0; g < G; ++g) {
        for (size_t m = 0; m < rows; ++m) {
            const int8_t* row = a.data() + m * cols + g * gk;
            int32_t s = 0;
            for (size_t k = 0; k < gk; ++k) s += static_cast<int32_t>(row[k]);
            out[g * rows + m] = -s;
        }
    }
    return out;
}

void ReferenceMidGroupGemm(
    const std::vector<int8_t>& a_q,
    const std::vector<uint16_t>& a_scale_g,
    const std::vector<int8_t>& w_q,
    const std::vector<uint16_t>& w_scale_g,
    const std::vector<uint16_t>& w_zero_g,
    std::vector<uint16_t>& y_bf16,
    uint32_t M, uint32_t N, uint32_t K, uint32_t gk)
{
    y_bf16.assign(static_cast<size_t>(M) * N, 0);
    const uint32_t G = K / gk;
    const bool asym = !w_zero_g.empty();

#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < static_cast<int64_t>(M); ++m) {
        const int8_t* arow = a_q.data() + static_cast<size_t>(m) * K;
        for (uint32_t n = 0; n < N; ++n) {
            const int8_t* wrow = w_q.data() + static_cast<size_t>(n) * K;
            float acc = 0.0f;
            for (uint32_t g = 0; g < G; ++g) {
                const size_t base = static_cast<size_t>(g) * gk;
                // Subtract the zero point from every weight, which is the
                // DEFINITION.  The kernel instead uses the rank-1 identity
                // (-wz * sum_k A); doing the same here would make the check
                // circular -- a sign error in the identity would cancel out.
                const int32_t wz = !asym ? 0 : static_cast<int32_t>(
                    BF16ToFP32(w_zero_g[static_cast<size_t>(g) * N + n]));
                int32_t dot = 0;
                for (uint32_t k = 0; k < gk; ++k) {
                    dot += static_cast<int32_t>(arow[base + k]) *
                           (static_cast<int32_t>(wrow[base + k]) - wz);
                }
                acc += static_cast<float>(dot) *
                       BF16ToFP32(w_scale_g[static_cast<size_t>(g) * N + n]) *
                       BF16ToFP32(a_scale_g[static_cast<size_t>(g) * M + m]);
            }
            y_bf16[static_cast<size_t>(m) * N + n] = FP32ToBF16(acc);
        }
    }
}
