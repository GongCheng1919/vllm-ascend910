// ref_gemm.h/.cpp — CPU references for the three raw GEMM dtypes.
#pragma once

#include <cstdint>
#include <vector>
#include <string>

// op: "s4s4" | "s8s8" | "bf16"
// A: [M,K], B: [K,N] (row-major); returns C: [M,N] in int32 (int ops) or fp32 (bf16).
// Input vectors hold the *unpacked* logical elements:
//   s4s4 -> int8_t in [-8, 7] (packing to nibbles happens on the host for device)
//   s8s8 -> int8_t
//   bf16 -> uint16_t (bf16 bits)
std::vector<int32_t> RefGemmInt(const std::string& op, const std::vector<int8_t>& a,
                                const std::vector<int8_t>& b, uint32_t M, uint32_t N, uint32_t K);

std::vector<float> RefGemmBf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                               uint32_t M, uint32_t N, uint32_t K);

// Host-side int4 packing: 2 x [-8,7] -> 1 byte (low nibble first).
void PackInt4(const std::vector<int8_t>& src, std::vector<uint8_t>& dst);

// Host-side unpacking for result comparison (from device int32 C).
float BF16ToFP32(uint16_t b);
uint16_t FP32ToBF16(float f);
