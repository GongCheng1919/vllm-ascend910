#include "ref_gemm.h"
#include <cstring>
#include <cmath>

float BF16ToFP32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t FP32ToBF16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if (std::isnan(f)) {
        return uint16_t((u >> 16) | 0x40);
    }
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    return uint16_t((u + rounding_bias) >> 16);
}

void PackInt4(const std::vector<int8_t>& src, std::vector<uint8_t>& dst) {
    dst.resize((src.size() + 1) / 2);
    for (size_t i = 0; i < src.size(); i += 2) {
        uint8_t lo = uint8_t(src[i] & 0xF);
        uint8_t hi = (i + 1 < src.size()) ? uint8_t(src[i + 1] & 0xF) : 0;
        dst[i / 2] = lo | (hi << 4);
    }
}

std::vector<int32_t> RefGemmInt(const std::string& op, const std::vector<int8_t>& a,
                                const std::vector<int8_t>& b, uint32_t M, uint32_t N, uint32_t K) {
    std::vector<int32_t> c((size_t)M * N, 0);
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            int64_t acc = 0;
            for (uint32_t k = 0; k < K; k++) {
                acc += (int64_t)a[(size_t)m * K + k] * (int64_t)b[(size_t)k * N + n];
            }
            c[(size_t)m * N + n] = (int32_t)acc;
        }
    }
    return c;
}

std::vector<float> RefGemmBf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                               uint32_t M, uint32_t N, uint32_t K) {
    std::vector<float> c((size_t)M * N, 0.0f);
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            double acc = 0.0;
            for (uint32_t k = 0; k < K; k++) {
                acc += (double)BF16ToFP32(a[(size_t)m * K + k]) * (double)BF16ToFP32(b[(size_t)k * N + n]);
            }
            c[(size_t)m * N + n] = (float)acc;
        }
    }
    return c;
}
