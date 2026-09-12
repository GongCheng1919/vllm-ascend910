// Host-only gate for the ragged-K layout.  No NPU needed; runs in ~1 s.
//
// WHAT IT PROVES.  That the zero padding described in kernels/mg_kgeom.h is
// EXACT, not approximate -- by decoding the packed device payload exactly the
// way the kernel will and comparing the int32 per-group partial against the
// definition (ReferenceMidGroupGemm's `sum_k A*(W-wz)`) bit for bit.
//
// WHY IT EXISTS.  The kernel change for ragged K is mechanical, but its
// correctness rests entirely on an algebraic claim: a padded slot contributes 0
// to the Mmad AND 0 to both rank-1 correction terms, so nothing downstream has
// to remember how many elements were real.  That claim is cheap to check here
// and expensive to debug on device (a wrong pad shows up as a plausible-looking
// SNR, not a crash).  Run this before touching the .inc.
//
// The formula being reproduced is midgroup_w4a8_gemm.inc:369 --
//   acc += ((16*C_hi + C_lo) + 8*w_ksum[g,n] - wz[g,n]*a_ksum[g,m]) * ws * as
// with a_ksum stored NEGATED, and the Mmad running over the PADDED range.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "bf16.h"
#include "gemm_ref.h"
#include "mg_kgeom.h"

namespace {

int failures = 0;

// Read back one int4 value from the packed layout, sign-extended from 4 bits.
int32_t GetNibble(const std::vector<int8_t>& p, size_t elemPos)
{
    const uint8_t b = static_cast<uint8_t>(p[elemPos >> 1]);
    const uint8_t n = (elemPos & 1u) ? (b >> 4) : (b & 0x0F);
    return (n & 0x8u) ? (static_cast<int32_t>(n) - 16) : static_cast<int32_t>(n);
}

uint32_t Rng(uint32_t& s) { s = s * 1664525u + 1013904223u; return s >> 8; }

void Case(uint32_t M, uint32_t N, uint32_t K, uint32_t gk, bool asym)
{
    const uint32_t al   = mgk::kCubeKElemsInt4;
    const uint32_t G    = mgk::NumGroups(K, gk);
    const uint32_t Kpad = mgk::KPadElems(K, gk, al);

    uint32_t seed = 12345u + K * 7u + gk;
    std::vector<int8_t> A((size_t)M * K), W((size_t)N * K);
    for (auto& v : A) v = static_cast<int8_t>(static_cast<int32_t>(Rng(seed) % 256u) - 128);
    for (auto& v : W) v = static_cast<int8_t>(static_cast<int32_t>(Rng(seed) % 16u) - 8);

    std::vector<uint16_t> as((size_t)G * M), ws((size_t)G * N), wz((size_t)G * N, 0);
    for (auto& v : as) v = FP32ToBF16(0.0005f + 0.0015f * (Rng(seed) % 1000u) / 1000.0f);
    for (auto& v : ws) v = FP32ToBF16(0.0005f + 0.0015f * (Rng(seed) % 1000u) / 1000.0f);
    if (asym) {
        for (auto& v : wz)
            v = FP32ToBF16(static_cast<float>(static_cast<int32_t>(Rng(seed) % 16u) - 8));
    }

    // The device payload, in the grouped padded layout.
    std::vector<int8_t> hi, lo;
    SplitInt8ToInt4PlanesGrouped(A, M, K, gk, hi, lo);
    const std::vector<int8_t> wq = PackInt4Grouped(W, N, K, gk);
    const std::vector<int32_t> wksum = WeightKSumPerGroup(W, N, K, gk);
    const std::vector<int32_t> aksneg = ActKSumNegPerGroup(A, M, K, gk);

    // ---- 1. every PADDED slot must decode to a == 0 and w == 0 ----
    size_t padSlots = 0;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t off  = mgk::GroupOffsetElems(g, gk, al);
            const uint32_t real = mgk::GroupRealElems(g, K, gk);
            const uint32_t pad  = mgk::GroupPadElems(g, K, gk, al);
            for (uint32_t k = real; k < pad; ++k) {
                const size_t pos = (size_t)m * Kpad + off + k;
                const int32_t a = 16 * GetNibble(hi, pos) + GetNibble(lo, pos) + 8;
                if (a != 0) {
                    printf("  FAIL pad activation decodes to %d at m=%u g=%u k=%u\n",
                           a, m, g, k);
                    ++failures;
                }
                ++padSlots;
            }
            // and the REAL slots must round-trip the MSD identity exactly
            for (uint32_t k = 0; k < real; ++k) {
                const size_t pos = (size_t)m * Kpad + off + k;
                const int32_t a = 16 * GetNibble(hi, pos) + GetNibble(lo, pos) + 8;
                if (a != A[(size_t)m * K + (size_t)g * gk + k]) {
                    printf("  FAIL MSD round-trip m=%u g=%u k=%u: %d vs %d\n", m, g, k, a,
                           A[(size_t)m * K + (size_t)g * gk + k]);
                    ++failures;
                }
            }
        }
    }
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t off  = mgk::GroupOffsetElems(g, gk, al);
            const uint32_t real = mgk::GroupRealElems(g, K, gk);
            const uint32_t pad  = mgk::GroupPadElems(g, K, gk, al);
            for (uint32_t k = real; k < pad; ++k) {
                if (GetNibble(wq, (size_t)n * Kpad + off + k) != 0) {
                    printf("  FAIL pad weight non-zero at n=%u g=%u k=%u\n", n, g, k);
                    ++failures;
                }
            }
        }
    }

    // ---- 2. the kernel's int32 per-group partial == the definition ----
    // Kernel path: Mmad over the PADDED range, then the two rank-1 corrections.
    // Definition:  sum over the REAL range of A*(W-wz).  These must agree in
    // int32, with no tolerance, or the padding is not free.
    size_t exact = 0;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t g = 0; g < G; ++g) {
                const uint32_t off  = mgk::GroupOffsetElems(g, gk, al);
                const uint32_t real = mgk::GroupRealElems(g, K, gk);
                const uint32_t pad  = mgk::GroupPadElems(g, K, gk, al);
                const int32_t wzi = static_cast<int32_t>(BF16ToFP32(wz[(size_t)g * N + n]));

                int32_t mmad = 0;                       // the cube's int32 output
                for (uint32_t k = 0; k < pad; ++k) {    // PADDED range, as the cube sees it
                    const size_t ap = (size_t)m * Kpad + off + k;
                    const size_t wp = (size_t)n * Kpad + off + k;
                    mmad += (16 * GetNibble(hi, ap) + GetNibble(lo, ap)) * GetNibble(wq, wp);
                }
                const int32_t kern = mmad + 8 * wksum[(size_t)g * N + n]
                                     - wzi * (-aksneg[(size_t)g * M + m]);

                int32_t def = 0;                        // the definition, REAL range
                for (uint32_t k = 0; k < real; ++k) {
                    def += static_cast<int32_t>(A[(size_t)m * K + (size_t)g * gk + k]) *
                           (static_cast<int32_t>(W[(size_t)n * K + (size_t)g * gk + k]) - wzi);
                }
                if (kern != def) {
                    printf("  FAIL group partial m=%u n=%u g=%u: kernel %d != def %d\n",
                           m, n, g, kern, def);
                    ++failures;
                } else {
                    ++exact;
                }
            }
        }
    }

    // ---- 3. and the full bf16 output matches the reference bit for bit ----
    std::vector<uint16_t> yRef;
    ReferenceMidGroupGemm(A, as, W, ws, asym ? wz : std::vector<uint16_t>{}, yRef,
                          M, N, K, gk);
    size_t nonzero = 0;
    for (uint16_t v : yRef) nonzero += (v != 0);
    if (nonzero == 0) { printf("  FAIL reference is all zeros\n"); ++failures; }

    printf("M=%-4u N=%-4u K=%-6u GK=%-5u %-5s G=%-3u lastGK=%-5u Kpad=%-6u "
           "pad=%zu slots, %zu partials exact\n",
           M, N, K, gk, asym ? "asym" : "sym", G, mgk::GroupRealElems(G - 1, K, gk),
           Kpad, padSlots, exact);
}

// The aligned case must be BYTE-IDENTICAL to the pre-ragged packers, or every
// number P0..P6 measured stops being comparable to anything measured after this
// change.  Kpad == K there, so the grouped path must degenerate exactly.
void Regression(uint32_t rows, uint32_t K, uint32_t gk)
{
    uint32_t seed = 999u;
    std::vector<int8_t> A((size_t)rows * K), W((size_t)rows * K);
    for (auto& v : A) v = static_cast<int8_t>(static_cast<int32_t>(Rng(seed) % 256u) - 128);
    for (auto& v : W) v = static_cast<int8_t>(static_cast<int32_t>(Rng(seed) % 16u) - 8);

    if (mgk::KPadElems(K, gk, mgk::kCubeKElemsInt4) != K) {
        printf("  FAIL regression shape K=%u is not actually aligned\n", K);
        ++failures;
        return;
    }
    const bool wSame = (PackInt4Grouped(W, rows, K, gk) == PackInt4(W, rows, K));
    std::vector<int8_t> hiG, loG, hiO, loO;
    SplitInt8ToInt4PlanesGrouped(A, rows, K, gk, hiG, loG);
    SplitInt8ToInt4Planes(A, rows, K, hiO, loO);
    const bool aSame = (hiG == hiO) && (loG == loO);
    if (!wSame || !aSame) {
        printf("  FAIL K=%u: weight %s, activation %s\n", K,
               wSame ? "same" : "DIFFERS", aSame ? "same" : "DIFFERS");
        ++failures;
    } else {
        printf("K=%-6u GK=%-5u byte-identical to the pre-ragged packers\n", K, gk);
    }
}

}  // namespace

int main()
{
    printf("== regression: aligned shapes must not move a single byte ==\n");
    Regression(4, 5120, 1024);
    Regression(4, 27648, 1024);
    Regression(4, 2048, 1024);
    Regression(4, 1280, 1024);       // ragged GROUP, aligned fractals -> no padding
    printf("\n");

    printf("== QwQ-32B row-parallel K shards (GK=1024): ragged at the GROUP level ==\n");
    for (uint32_t tp : {1u, 2u, 4u, 8u}) Case(3, 128, 5120 / tp, 1024, true);
    for (uint32_t tp : {2u, 4u, 8u}) Case(3, 128, 27648 / tp, 1024, true);

    printf("\n== pathological K: a short last group, down to n=1 ==\n");
    for (uint32_t K : {1057u, 1279u, 1025u, 33u, 1u, 65u, 63u, 2049u})
        Case(3, 128, K, 1024, true);
    Case(3, 128, 1057, 1024, false);       // sym path too
    Case(17, 256, 777, 1024, true);        // ragged M and a single short group

    printf("\n== odd K: the last real element lands in a low nibble ==\n");
    for (uint32_t K : {1023u, 1057u, 35u, 1u}) Case(2, 128, K, 1024, true);

    printf("\n== unaligned GK: correct, though it pads every group ==\n");
    Case(3, 128, 100, 33, true);
    Case(3, 128, 1057, 96, true);

    printf("\n%s (failures=%d)\n", failures ? "*** FAIL ***" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
