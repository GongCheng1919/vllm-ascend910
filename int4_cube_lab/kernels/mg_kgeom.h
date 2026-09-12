#ifndef MG_KGEOM_H
#define MG_KGEOM_H
// ---------------------------------------------------------------------------
// The ragged-K group geometry for the mid-group W4A8 kernels.
//
// THE CONTRACT.  K, M and N are ARBITRARY.  Callers never align anything:
// K = 1057 with a 33-element last group, or an odd K, are legal inputs.  What
// the hardware actually requires is that every operand the cube touches is a
// whole number of FRACTALS -- 16 rows x C0, and for int4 C0 = 32 bytes = 64
// elements (`kNzBlockSize = kK0B * kBlock` in midgroup_w4a8_gemm.inc).  So the
// op absorbs the misalignment in its OWN packed layout by zero-padding each
// quant group up to a fractal boundary.  Nothing above the op sees this.
//
// WHY ZERO PADDING IS EXACT, NOT APPROXIMATE.  A padded element has q_a = 0 and
// q_w = 0, so it contributes 0 to the int32 dot product AND 0 to both rank-1
// correction terms (`w_ksum = sum q_w`, `a_ksum = -sum q_a`).  That is why the
// sums may be taken over the padded range without bookkeeping: no code needs to
// remember how many elements were real.  The one quantity that must be computed
// on the REAL range is the group's scale (amax over the real elements) -- and
// even there, appending zeros cannot raise an amax of absolute values, so the
// quantiser gets the same scale whether it masks the tail or zeroes it.
//
// THE LAYOUT.  G = ceil(K / GK) groups.  Group g occupies
// `GroupPadElems(g)` elements starting at `GroupOffsetElems(g)`:
//
//     g <  G-1 : real GK,          padded to ceil(GK / align)
//     g == G-1 : real K-(G-1)*GK,  padded to ceil(that / align)
//
// so the offset is simply g * ceil(GK/align) for every g.  Kpad = the sum.
// When `GK % align == 0` -- the only regime worth shipping -- only the LAST
// group can pad, and by at most align-1 elements (<1% of a row).  An unaligned
// GK is still CORRECT here, it just pads every group; that is a statement about
// efficiency, not about validity, and the two must not be conflated.
//
// Included by both the kernel (.inc, device side) and the host harness / op, so
// the AIC, the AIV, the packer and the CPU reference cannot disagree about where
// a group starts.  A disagreement between AIC and AIV about G is a DEADLOCK, not
// a wrong number, which is why there is exactly one definition of it.
// ---------------------------------------------------------------------------

// The device compiler needs its own qualifier on functions called from
// __aicore__ code; the host build wants none.  The includer picks.
#ifndef MG_KGEOM_FN
#define MG_KGEOM_FN inline constexpr
#endif

namespace mgk {

// int4 packs 2 elements per byte and C0 is 32 bytes, so the cube's K-direction
// granularity is 64 elements.  This is the ONLY alignment the op imposes on
// itself, and it never reaches the caller.
enum : unsigned int { kCubeKElemsInt4 = 64u };

MG_KGEOM_FN unsigned int CeilDiv(unsigned int a, unsigned int b)
{
    return (a + b - 1u) / b;
}

MG_KGEOM_FN unsigned int CeilTo(unsigned int a, unsigned int b)
{
    return CeilDiv(a, b) * b;
}

// Number of quant groups: the LAST one may be short.
MG_KGEOM_FN unsigned int NumGroups(unsigned int K, unsigned int GK)
{
    return CeilDiv(K, GK);
}

// Padded width of a full (non-final) group.  Also the group stride, for every g.
MG_KGEOM_FN unsigned int GroupStrideElems(unsigned int GK, unsigned int align)
{
    return CeilTo(GK, align);
}

// Real element count of group g.
MG_KGEOM_FN unsigned int GroupRealElems(unsigned int g, unsigned int K, unsigned int GK)
{
    const unsigned int G = NumGroups(K, GK);
    return (g + 1u < G) ? GK : (K - (G - 1u) * GK);
}

// Padded element count of group g -- a whole number of fractals.
MG_KGEOM_FN unsigned int GroupPadElems(unsigned int g, unsigned int K, unsigned int GK,
                                       unsigned int align)
{
    return CeilTo(GroupRealElems(g, K, GK), align);
}

// Element offset of group g inside the padded row.
MG_KGEOM_FN unsigned int GroupOffsetElems(unsigned int g, unsigned int GK, unsigned int align)
{
    return g * GroupStrideElems(GK, align);
}

// Padded row length in elements.  This -- not K -- is what the packed tensors
// are sized by, and what Nd2Nz uses as srcDValue (after the /2 for int4).
MG_KGEOM_FN unsigned int KPadElems(unsigned int K, unsigned int GK, unsigned int align)
{
    const unsigned int G = NumGroups(K, GK);
    return (G - 1u) * GroupStrideElems(GK, align) + GroupPadElems(G - 1u, K, GK, align);
}

// ---- slice geometry, in BYTES (int4: 2 elements per byte) ----
// A group's padded byte count need not be a multiple of the L0 slice, so the
// LAST slice of a group can be short -- always by a multiple of C0 (32 B),
// which is what keeps LoadData2D's repeatTimes and Mmad's k legal.

MG_KGEOM_FN unsigned int SlicesInGroup(unsigned int groupPadBytes, unsigned int innerKB)
{
    return CeilDiv(groupPadBytes, innerKB);
}

MG_KGEOM_FN unsigned int LastSliceBytes(unsigned int groupPadBytes, unsigned int innerKB)
{
    const unsigned int n = SlicesInGroup(groupPadBytes, innerKB);
    return groupPadBytes - (n - 1u) * innerKB;
}

// Bytes of slice s within a group of `groupPadBytes` bytes.
MG_KGEOM_FN unsigned int SliceBytes(unsigned int s, unsigned int groupPadBytes,
                                    unsigned int innerKB)
{
    return (s + 1u < SlicesInGroup(groupPadBytes, innerKB))
               ? innerKB
               : LastSliceBytes(groupPadBytes, innerKB);
}

}  // namespace mgk

#endif  // MG_KGEOM_H
