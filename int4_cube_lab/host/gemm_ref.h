#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mg_kgeom.h"

// Pack unpacked int4 values (each in [-8,7], stored one per int8_t) into the
// device layout: 2 elements per byte along the last axis, even index in the low
// nibble.  `cols` must be even.
std::vector<int8_t> PackInt4(const std::vector<int8_t>& v, size_t rows, size_t cols);

// ---------------------------------------------------------------------------
// Ragged-K variants.  See kernels/mg_kgeom.h for the layout and for why the
// zero padding is exact.  The UNPACKED inputs are always [rows, K] with the
// REAL K -- only the device payload carries padding, so the CPU reference never
// sees it and a kernel that mishandles the pad is caught rather than excused.
// ---------------------------------------------------------------------------

// Pack [rows, K] into the grouped padded layout: G = ceil(K/GK) groups, group g
// at element offset g*ceil(GK/64), its tail zero-filled to a fractal boundary.
// Result is rows * (KPadElems(K,GK)/2) bytes.
std::vector<int8_t> PackInt4Grouped(const std::vector<int8_t>& v, size_t rows,
                                    uint32_t K, uint32_t gk);

// MSD split into two packed int4 planes, in the same grouped padded layout.
void SplitInt8ToInt4PlanesGrouped(const std::vector<int8_t>& a, size_t rows,
                                  uint32_t K, uint32_t gk,
                                  std::vector<int8_t>& hi_packed,
                                  std::vector<int8_t>& lo_packed);

// Most-significant-digit split of int8 activations into two packed int4 planes:
//   hi[k] = a >> 4          in [-8,7]
//   lo[k] = (a & 15) ^ 8    in [-8,7]
// so that a == 16*hi + lo + 8 exactly, for every int8 value.  Both planes come
// back in the same packed layout as PackInt4.  The trailing +8 leaves a term
// 8 * sum_k W[n,k] on the output, which is what WeightKSum supplies.
void SplitInt8ToInt4Planes(const std::vector<int8_t>& a, size_t rows, size_t cols,
                           std::vector<int8_t>& hi_packed,
                           std::vector<int8_t>& lo_packed);

// ksum[n] = sum_k W[n,k], over the UNPACKED weight values.
std::vector<int32_t> WeightKSum(const std::vector<int8_t>& w, size_t rows, size_t cols);

// Group-major per-group weight k-sum: ksum[g*rows + n] = sum_{k in group g} W[n,k].
// Layout matches the mid-group kernel's [G, N] with G = ceil(cols/gk); a short
// final group sums only its real elements (the padded ones are zero anyway, so
// this agrees with summing over the padded range -- see kernels/mg_kgeom.h).
std::vector<int32_t> WeightKSumPerGroup(const std::vector<int8_t>& w,
                                        size_t rows, size_t cols, size_t gk);

// C[m,n] = sum_k A[m,k] * W[n,k], then * a_scale[m] * w_scale[n], rounded to bf16.
// A and W are the UNPACKED value arrays (valid for both the int4 and int8 build).
void ReferencePerchannelGemm(
    const std::vector<int8_t>& a_q,
    const std::vector<uint16_t>& a_scale_bf16,
    const std::vector<int8_t>& w_q,
    const std::vector<uint16_t>& w_scale_bf16,
    std::vector<uint16_t>& y_bf16,
    uint32_t M, uint32_t N, uint32_t K);

// The real mid-group reference (this is what P3 is verified against; the
// degenerate-scale path above only ever proved the pipeline, not the maths):
//
//   y[m,n] = sum_g as[g,m] * ws[g,n] * ( sum_{k in g} A[m,k] * (W[n,k] - wz[g,n]) )
//
// Scales are GROUP-MAJOR, matching the kernel: a_scale_g is [G,M], w_scale_g and
// w_zero_g are [G,N], with G = ceil(K/gk) -- the final group may be short.
// `w_zero_g` empty means symmetric.
// Accumulation across groups is fp32, per group is int32, exactly as the kernel
// does it -- so a mismatch here is a maths bug, not a rounding difference.
//
// The zero point is applied as -wz * sum_{k in g} A[m,k] rather than by
// subtracting it from every weight, which is the identity the kernel exploits;
// computing it the other way here would make the check circular.
// `mp` is the row stride of a_scale_g / a_ksum -- the TILE_M-PADDED row count the
// kernel is launched with, which is NOT M whenever the caller pads.  Passing M
// here (the old signature's only option) silently reads the wrong scale for every
// row once Mp != M, and shows up as a whole-tensor mismatch that looks like a
// kernel bug.  Defaults to M so the aligned call sites are unchanged.
void ReferenceMidGroupGemm(
    const std::vector<int8_t>& a_q,
    const std::vector<uint16_t>& a_scale_g,
    const std::vector<int8_t>& w_q,
    const std::vector<uint16_t>& w_scale_g,
    const std::vector<uint16_t>& w_zero_g,
    std::vector<uint16_t>& y_bf16,
    uint32_t M, uint32_t N, uint32_t K, uint32_t gk, uint32_t mp = 0);

// akneg[g*M + m] = -sum_{k in group g} A[m,k], group-major, int32, G = ceil.
// This is what the activation quantise kernel will emit alongside a_scale; it is
// int32 and not bf16 because at GK=1024 the sum reaches ~1.3e5.
std::vector<int32_t> ActKSumNegPerGroup(const std::vector<int8_t>& a,
                                        size_t rows, size_t cols, size_t gk);
