#include "gemm_ref.h"

#include <algorithm>
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

namespace {
// Write one int4 value at a PADDED element position.  The destination starts
// zeroed, so every position never written is a zero -- which is exactly the
// padding contract (mg_kgeom.h).
inline void PutNibble(std::vector<int8_t>& out, size_t elemPos, int8_t v)
{
    const size_t b = elemPos >> 1;
    const uint8_t nib = static_cast<uint8_t>(v) & 0x0F;
    if (elemPos & 1u) {
        out[b] = static_cast<int8_t>((static_cast<uint8_t>(out[b]) & 0x0F) | (nib << 4));
    } else {
        out[b] = static_cast<int8_t>((static_cast<uint8_t>(out[b]) & 0xF0) | nib);
    }
}
}  // namespace

std::vector<int8_t> PackInt4Grouped(const std::vector<int8_t>& v, size_t rows,
                                    uint32_t K, uint32_t gk)
{
    const uint32_t al   = mgk::kCubeKElemsInt4;
    const uint32_t G    = mgk::NumGroups(K, gk);
    const uint32_t Kpad = mgk::KPadElems(K, gk, al);
    std::vector<int8_t> packed(rows * (Kpad / 2), 0);
    for (size_t r = 0; r < rows; ++r) {
        const size_t rowPad = r * Kpad;                // in padded elements
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t off  = mgk::GroupOffsetElems(g, gk, al);
            const uint32_t real = mgk::GroupRealElems(g, K, gk);
            for (uint32_t k = 0; k < real; ++k) {
                PutNibble(packed, rowPad + off + k, v[r * K + (size_t)g * gk + k]);
            }
        }
    }
    return packed;
}

void SplitInt8ToInt4PlanesGrouped(const std::vector<int8_t>& a, size_t rows,
                                  uint32_t K, uint32_t gk,
                                  std::vector<int8_t>& hi_packed,
                                  std::vector<int8_t>& lo_packed)
{
    const uint32_t al   = mgk::kCubeKElemsInt4;
    const uint32_t G    = mgk::NumGroups(K, gk);
    const uint32_t Kpad = mgk::KPadElems(K, gk, al);
    // The MSD encoding is a == 16*hi + lo + 8, so a ZERO activation is NOT two
    // zero nibbles: it is hi=0, lo=-8 (nibble 0x8).  Initialising the lo plane
    // to 0x88 therefore makes every slot that is never written decode to a == 0
    // exactly, which keeps the padding self-consistent instead of relying on the
    // weight's pad also being zero.  (It is -- but a correctness argument that
    // needs BOTH operands to cooperate is one bug away from being wrong.)
    hi_packed.assign(rows * (Kpad / 2), 0);
    lo_packed.assign(rows * (Kpad / 2), static_cast<int8_t>(0x88));
    for (size_t r = 0; r < rows; ++r) {
        const size_t rowPad = r * Kpad;
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t off  = mgk::GroupOffsetElems(g, gk, al);
            const uint32_t real = mgk::GroupRealElems(g, K, gk);
            for (uint32_t k = 0; k < real; ++k) {
                const int8_t e = a[r * K + (size_t)g * gk + k];
                PutNibble(hi_packed, rowPad + off + k, static_cast<int8_t>(e >> 4));
                PutNibble(lo_packed, rowPad + off + k, static_cast<int8_t>(e ^ 8));
            }
        }
    }
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
    if (gk == 0) {
        std::cerr << "WeightKSumPerGroup: gk must be non-zero\n";
        std::exit(1);
    }
    // G = ceil: the final group may be short.  Summing only its real elements is
    // the same as summing its padded range, because the pad is zero.
    const size_t numG = mgk::NumGroups(static_cast<uint32_t>(cols),
                                       static_cast<uint32_t>(gk));
    std::vector<int32_t> ksum(numG * rows, 0);
#pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < static_cast<int64_t>(rows); ++n) {
        const int8_t* row = w.data() + static_cast<size_t>(n) * cols;
        for (size_t g = 0; g < numG; ++g) {
            const size_t end = std::min(cols, (g + 1) * gk);
            int32_t acc = 0;
            for (size_t k = g * gk; k < end; ++k) acc += static_cast<int32_t>(row[k]);
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
    const size_t G = mgk::NumGroups(static_cast<uint32_t>(cols),
                                    static_cast<uint32_t>(gk));
    std::vector<int32_t> out(G * rows, 0);
    for (size_t g = 0; g < G; ++g) {
        const size_t real = std::min(cols, (g + 1) * gk) - g * gk;
        for (size_t m = 0; m < rows; ++m) {
            const int8_t* row = a.data() + m * cols + g * gk;
            int32_t s = 0;
            for (size_t k = 0; k < real; ++k) s += static_cast<int32_t>(row[k]);
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
    uint32_t M, uint32_t N, uint32_t K, uint32_t gk, uint32_t mp)
{
    y_bf16.assign(static_cast<size_t>(M) * N, 0);
    const uint32_t G = mgk::NumGroups(K, gk);      // the last group may be short
    const uint32_t Mp = mp ? mp : M;               // a_scale's row stride
    const bool asym = !w_zero_g.empty();

#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < static_cast<int64_t>(M); ++m) {
        const int8_t* arow = a_q.data() + static_cast<size_t>(m) * K;
        for (uint32_t n = 0; n < N; ++n) {
            const int8_t* wrow = w_q.data() + static_cast<size_t>(n) * K;
            float acc = 0.0f;
            for (uint32_t g = 0; g < G; ++g) {
                const size_t base = static_cast<size_t>(g) * gk;
                const uint32_t real = mgk::GroupRealElems(g, K, gk);
                // Subtract the zero point from every weight, which is the
                // DEFINITION.  The kernel instead uses the rank-1 identity
                // (-wz * sum_k A); doing the same here would make the check
                // circular -- a sign error in the identity would cancel out.
                const int32_t wz = !asym ? 0 : static_cast<int32_t>(
                    BF16ToFP32(w_zero_g[static_cast<size_t>(g) * N + n]));
                int32_t dot = 0;
                for (uint32_t k = 0; k < real; ++k) {
                    dot += static_cast<int32_t>(arow[base + k]) *
                           (static_cast<int32_t>(wrow[base + k]) - wz);
                }
                acc += static_cast<float>(dot) *
                       BF16ToFP32(w_scale_g[static_cast<size_t>(g) * N + n]) *
                       BF16ToFP32(a_scale_g[static_cast<size_t>(g) * Mp + m]);
            }
            y_bf16[static_cast<size_t>(m) * N + n] = FP32ToBF16(acc);
        }
    }
}
