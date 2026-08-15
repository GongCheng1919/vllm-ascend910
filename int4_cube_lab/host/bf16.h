#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

// Host-side BF16 <-> FP32 helpers.
// BF16 layout: sign(1) | exp(8) | mantissa(7) -- same exp range as FP32, truncated mantissa.

inline float BF16ToFP32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Round-to-nearest-even (IEEE 754 default), matches typical NPU rounding.
inline uint16_t FP32ToBF16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Handle NaN: preserve NaN bit pattern by setting low mantissa bit
    if (std::isnan(f)) {
        return uint16_t((u >> 16) | 0x40);
    }
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    return uint16_t((u + rounding_bias) >> 16);
}

// Convenience: round-trip a fp32 scalar through bf16 (matches NPU bf16 storage).
inline float RoundTripBF16(float f) {
    return BF16ToFP32(FP32ToBF16(f));
}
