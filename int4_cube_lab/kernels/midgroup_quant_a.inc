// midgroup_quant_a.inc -- per-token mid-group INT8 activation quantiser + MSD split.
//
// The including .cpp must define, before the include:
//   GK_CFG       quantisation group size along K, in ELEMENTS
//   KERNEL_NAME  the extern "C" __global__ symbol to emit
//
// Contract (G = ceil(K / GK) groups; group-major outputs, matching the GEMM):
//   x:       [M,  K]      bf16   -- the activation, ARBITRARY K
//   a_hi:    [Mp, Kpad/2] int4 packed  -- high nibbles, in [-8,7]
//   a_lo:    [Mp, Kpad/2] int4 packed  -- low nibbles biased by -8, in [-8,7]
//   a_scale: [G, Mp]      bf16
//   a_ksum:  [G, Mp]      int32  -- NEGATED sum_{k in g} A_q[m,k]  (see GEMM D6)
//
// RAGGED K.  The final group holds `real` in [1, GK] elements and is emitted in
// the grouped PADDED layout of mg_kgeom.h -- zero-filled up to a whole int4
// fractal.  This kernel gets that for free: it zeroes the group's UB tail before
// the vector work, after which every op below stays FULL WIDTH and unchanged.
//   - amax: appending zeros cannot raise a max of absolute values -> same scale
//   - q = rint(0/s) = 0, and -sum_k q is unchanged -> same a_ksum
//   - MSD of q=0 is hi=0, lo=-8, which is precisely the encoding the host packer
//     writes into padded slots (`0x88`), so kernel and packer agree bit for bit.
//
// Mp >= M is the GEMM's TILE_M-padded row count, and it is a multiple of kBr.
// Producing the padded rows HERE, as zeros, is what keeps the caller free of any
// shape-dependent work: rows [M, Mp) quantise to scale=tiny / q=0 / ksum=0, so
// their output rows are finite and get sliced off.  Padding them on the host
// instead costs four `constant_pad_nd` launches per GEMM (a_hi/a_lo/a_scale/
// a_ksum) -- 33% of decode device time in the engine (P4_E2E.md D6, D10).
//
// ---------------------------------------------------------------------------
// STRUCTURE (adopted from nitro's `midgroup_rowq.cpp`)
//
// A job is a tile of kBr rows x one quant group, and the vector work runs on
// kSub-row sub-tiles, so every vector instruction covers kSub*GK = 8192 elements.
// The first version worked one (row, group) at a time -- 1024 elements per
// instruction -- and measured a FLAT 0.94 us per work item independent of the
// 3 KB it touched: at M=1024/K=27648 that was 650 us for 85 MB, ~129 GB/s
// against the ~1 TB/s this lab's memory-bound kernels reach.  Ablation put only
// 40 us of it on the stores and ~35 us on scalar round trips; the rest was ~19
// vector instructions x their fixed issue cost, paid per 1024 elements.  Same
// AIV instruction-issue wall P0 hit in the GEMM (P0_COST §2.5), same fix: make
// each instruction cover more data.
//
// Batching also makes the per-group outputs contiguous: kBr scales and kBr ksums
// per job go out as ONE DataCopyPad each, instead of a 2-byte and a 4-byte
// scattered write per row.
//
// ---------------------------------------------------------------------------
// SEMANTICS are `fakequant_lab/quant.py` verbatim.  This is where we deviate
// from nitro on purpose: nitro quantises with `inv = 127/rowMax` (the UNROUNDED
// max) while storing `bf16(rowMax/127)`, so the scale it quantises with is not
// the scale it stores.  quant.py divides by the ROUNDED scale, and P2's accuracy
// numbers were produced that way, so this kernel must too:
//
//   scale = bf16(max|x| / 127)            <- rounded to bf16 BEFORE use
//   scale = tiny(bf16) where max|x| == 0  <- keeps zero groups at 0, not NaN
//   q     = clamp(rint(x / float(scale)), -128, 127)
//
// `rint` is round-half-to-even, which is what torch.round does.  The divide is a
// real Div, not a multiply by a reciprocal: 1/s is inexact for a general bf16
// scale and would flip round-half-even decisions relative to torch.
//
// Known deviation: the zero-scale guard is `Maxs(scale, tiny)` rather than
// `where(scale == 0, tiny, scale)`.  These differ only when the fp32 scale is
// SUBNORMAL -- a group whose largest |x| is itself a bf16 denormal (< 1.5e-36).
// All-zero groups, the case the guard exists for, give exactly tiny either way.
//
// The MSD split is done in floating point rather than with bit twiddling:
//
//   hi = floor(q / 16)        == q >> 4          (arithmetic shift == floor)
//   lo = q - 16*hi - 8        == (q & 15) - 8    (host's `(q&15)^8` as a value)
//
// Both land in [-8,7] and are exact in half, so `Cast<int4b_t>` (vconv_f162s4r)
// does the nibble packing in one instruction -- no gather/compaction needed.

#include "kernel_operator.h"

// The ragged-K group geometry, shared verbatim with the GEMM, the packer and the
// CPU reference -- one definition, so nothing can disagree about where a group
// starts or how wide the last one is.
#define MG_KGEOM_FN __aicore__ inline constexpr
#include "mg_kgeom.h"

using namespace AscendC;

namespace {

#ifndef GK_CFG
#define GK_CFG 1024
#endif

constexpr uint32_t GK   = GK_CFG;
constexpr uint32_t kBr  = 16;                 // rows per DMA job
constexpr uint32_t kSub = 8;                  // rows per vector sub-tile
constexpr uint32_t kSubPerJob = kBr / kSub;
constexpr uint32_t kSubElems  = kSub * GK;    // 8192 -- the vector op width
constexpr uint32_t kJobElems  = kBr * GK;
constexpr uint32_t kBlkF = 64;                // fp32 elems per vector repeat
constexpr uint32_t kSmall = 64;               // small per-row vectors

constexpr float kQMax     = 127.0f;
// torch.finfo(torch.bfloat16).tiny -- same exponent range as fp32, so exact.
constexpr float kTinyBf16 = 1.17549435e-38f;

// GK a multiple of 64 keeps the fold/reduce repeat counts exact AND makes the
// group's padded span equal GK, so only the FINAL group is ever short.
static_assert(GK % kBlkF == 0, "GK must be a multiple of 64");
static_assert(kBr % kSub == 0, "kBr must be a multiple of kSub");

}  // namespace

extern "C" __global__ __aicore__
void KERNEL_NAME(GM_ADDR x, GM_ADDR a_hi, GM_ADDR a_lo,
                 GM_ADDR a_scale, GM_ADDR a_ksum,
                 uint32_t M, uint32_t Mp, uint32_t K)
{
    if ASCEND_IS_AIC { return; }

    const uint32_t numG    = mgk::NumGroups(K, GK);
    // The output planes are Kpad/2 bytes per row, not K/2: the final group is
    // padded up to a fractal (mg_kgeom.h).
    const uint32_t KbPad   = mgk::KPadElems(K, GK, mgk::kCubeKElemsInt4) / 2;
    // Jobs cover the PADDED rows; only the load is clamped to the real ones.
    const uint32_t rowJobs = (Mp + kBr - 1) / kBr;
    const uint64_t jobs    = (uint64_t)rowJobs * numG;
    // AIV-only build: GetSubBlockIdx() is always 0 here, so GetBlockIdx() /
    // GetBlockNum() already enumerate the vector cores.  (The GEMM is a mix
    // kernel and must use blockIdx*2 + subBlockIdx -- using that idiom here
    // yields only EVEN ids, and half the work silently never runs.)
    const uint32_t aivId  = GetBlockIdx();
    const uint32_t aivNum = GetBlockNum();

    GlobalTensor<bfloat16_t> xGm, sGm;
    GlobalTensor<int8_t> hiGm, loGm;
    GlobalTensor<int32_t> ksGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), (uint64_t)M * K);
    hiGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a_hi), (uint64_t)Mp * KbPad);
    loGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(a_lo), (uint64_t)Mp * KbPad);
    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a_scale), (uint64_t)numG * Mp);
    ksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a_ksum), (uint64_t)numG * Mp);

    TPipe pipe;
    TBuf<QuePosition::VECCALC> bX, bHi4, bLo4, bF0, bF1, bHalf,
                               bMax, bScF, bScBf, bScAll, bSBlk, bKs, bKsI;
    pipe.InitBuffer(bX,    kJobElems * sizeof(bfloat16_t));   // 32 KB
    pipe.InitBuffer(bHi4,  kJobElems / 2);                    //  8 KB
    pipe.InitBuffer(bLo4,  kJobElems / 2);                    //  8 KB
    pipe.InitBuffer(bF0,   kSubElems * sizeof(float));        // 32 KB
    pipe.InitBuffer(bF1,   kSubElems * sizeof(float));        // 32 KB
    pipe.InitBuffer(bHalf, kSubElems * sizeof(half));         // 16 KB
    pipe.InitBuffer(bMax,  kSmall * sizeof(float));
    pipe.InitBuffer(bScF,  kSmall * sizeof(float));
    pipe.InitBuffer(bScBf, kSmall * sizeof(bfloat16_t));
    pipe.InitBuffer(bScAll, kSmall * sizeof(float));
    pipe.InitBuffer(bSBlk, kSmall * 8 * sizeof(float));
    pipe.InitBuffer(bKs,   kSmall * sizeof(float));
    pipe.InitBuffer(bKsI,  kBr * sizeof(int32_t));

    auto ubX    = bX.Get<bfloat16_t>();
    auto ubHi4  = bHi4.Get<int8_t>();
    auto ubLo4  = bLo4.Get<int8_t>();
    auto ubF0   = bF0.Get<float>();      // lossless bf16->fp32 of the sub-tile
    auto ubF1   = bF1.Get<float>();      // abs / q / hi scratch
    auto ubI1   = bF1.Get<int32_t>();    // int32 view of the same bytes
    auto ubHalf = bHalf.Get<half>();
    auto ubMax  = bMax.Get<float>();
    auto ubScF  = bScF.Get<float>();
    auto ubScBf = bScBf.Get<bfloat16_t>();
    auto ubScAll = bScAll.Get<float>();   // kBr rounded scales, fp32, 32 B aligned
    auto ubSBlk = bSBlk.Get<float>();
    auto ubKs   = bKs.Get<float>();
    auto ubKsI  = bKsI.Get<int32_t>();

    event_t evM2V   = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
    event_t evVMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
    event_t evVMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
    event_t evMte3V = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());

    // Fold a GK-wide row down to its first 64 floats, then one reduce per row.
    const BinaryRepeatParams foldPar{1, 1, 1, 0, 8, 0};
    // src1 is one 8-float Brcb block re-read by every block and every repeat.
    const BinaryRepeatParams bcastPar{1, 1, 0, 8, 8, 0};

    for (uint64_t job = aivId; job < jobs; job += aivNum) {
        const uint32_t rj   = static_cast<uint32_t>(job / numG);
        const uint32_t g    = static_cast<uint32_t>(job % numG);
        const uint32_t row0 = rj * kBr;
        // Store every padded row of this job; load only the rows x actually has.
        // Mp is a multiple of kBr, so `rowsStore` is kBr except when a caller
        // passes Mp == M with a ragged M (the lab harness does).
        const uint32_t rowsStore = (Mp - row0 < kBr) ? (Mp - row0) : kBr;
        const uint32_t rowsLoad  = (row0 >= M) ? 0u
                                              : ((M - row0 < kBr) ? (M - row0) : kBr);
        // This group's real width, and the padded width that gets stored.
        const uint32_t gReal = mgk::GroupRealElems(g, K, GK);
        const uint32_t gPad  = mgk::GroupPadElems(g, K, GK, mgk::kCubeKElemsInt4);
        const bool     shortG = (gReal < GK);

        // Rows the load does not cover -- the tail of a ragged job and every row
        // of a fully-padded one -- are zeroed here.  Stored padded rows come out
        // as scale=tiny / q=0 / ksum=0, which contributes nothing to the GEMM;
        // rows beyond `rowsStore` are quantised too (cheaper than masking) but
        // never stored, so they only have to be finite.
        //
        // A SHORT group zeroes the whole job buffer for the same reason: the
        // elements past `gReal` must read as 0 so the full-width vector ops below
        // stay untouched.  Once per K sweep, so its cost is irrelevant.
        if (rowsLoad < kBr || shortG) {
            // half's zero bit pattern is bf16's, and there is no scalar bf16
            // conversion on this backend -- duplicate through the half view.
            Duplicate(bX.Get<half>(), static_cast<half>(0.0f), kJobElems);
        }

        // V -> MTE2 on ubX, covering two write-after-* hazards that
        // `PipeBarrier<PIPE_V>` cannot: the zero fill just above (V) racing the
        // load below (MTE2), and the PREVIOUS job's reads of ubX (V) racing this
        // job's load.  Both were latent while the CALLER padded M to a multiple
        // of kBr -- the fill branch never ran, and the previous job's V work was
        // long enough to hide the second.  Moving the padding into this kernel
        // put a ragged job on the decode hot path and the first one surfaced at
        // once, as exactly (M mod kBr) wrong rows.
        SetFlag<HardEvent::V_MTE2>(evVMte2);
        WaitFlag<HardEvent::V_MTE2>(evVMte2);

        if (rowsLoad > 0) {
            const uint64_t xOff = (uint64_t)row0 * K + (uint64_t)g * GK;
            if (!shortG) {
                DataCopyPadExtParams<bfloat16_t> ipad{false, 0, 0,
                                                      static_cast<bfloat16_t>(0.0f)};
                DataCopyExtParams ip{static_cast<uint16_t>(rowsLoad),
                                     static_cast<uint32_t>(GK * sizeof(bfloat16_t)),
                                     static_cast<uint32_t>((K - GK) * sizeof(bfloat16_t)),
                                     0u, 0u};
                DataCopyPad(ubX, xGm[xOff], ip, ipad);
            } else {
                // A short group reads gReal < GK elements per row but must still
                // land them GK apart in UB, so the UB rows keep the sub-tile
                // layout the vector ops assume.  One burst per row rather than a
                // strided multi-row copy: dstStride's unit (bytes vs 32 B blocks)
                // differs by direction, and a row-pitch bug here would show up as
                // a few wrong ROWS -- the same tell as the M-padding race this
                // kernel already carries a barrier for.  It costs <=16 extra
                // MTE3-class issues on ONE group out of G.
                //
                // isPad MUST be true here.  A DataCopyPad whose blockLen is not a
                // multiple of 32 B still moves a whole number of 32 B blocks, so
                // with isPad=false the hardware brings in the elements that FOLLOW
                // the group in GM -- real activation values, not zeros.  Their
                // effect is invisible in the scale (an amax over the same
                // distribution rarely moves) and fatal in a_ksum and in the packed
                // planes: exactly the symptom this cost a debug cycle to find.
                // isPad + rightPadding zero-fills the block tail instead.
                const uint32_t realBytes = gReal * sizeof(bfloat16_t);
                const uint32_t rightPadElems =
                    (mgk::CeilTo(realBytes, 32u) - realBytes) / sizeof(bfloat16_t);
                DataCopyPadExtParams<bfloat16_t> ipad1{
                    true, 0, static_cast<uint8_t>(rightPadElems),
                    static_cast<bfloat16_t>(0.0f)};
                DataCopyExtParams ip1{1u, realBytes, 0u, 0u, 0u};
                for (uint32_t r = 0; r < rowsLoad; ++r) {
                    DataCopyPad(ubX[r * GK], xGm[xOff + (uint64_t)r * K], ip1, ipad1);
                }
            }
        }
        SetFlag<HardEvent::MTE2_V>(evM2V);
        WaitFlag<HardEvent::MTE2_V>(evM2V);

        // Only the sub-tiles that are stored.  At decode (M=1, Mp=16) that is
        // still both sub-tiles, because the padded rows have to be written.
        const uint32_t subTiles = (rowsStore + kSub - 1) / kSub;
        for (uint32_t t = 0; t < subTiles; ++t) {
            const uint32_t off = t * kSubElems;
            const uint32_t r0  = t * kSub;

            Cast(ubF0, ubX[off], RoundMode::CAST_NONE, kSubElems);
            PipeBarrier<PIPE_V>();
            Abs(ubF1, ubF0, kSubElems);
            PipeBarrier<PIPE_V>();

            // per-row absmax: chunked Max fold, then one WholeReduceMax
            for (uint32_t r = 0; r < kSub; ++r) {
                Max(ubF1[r * GK], ubF1[r * GK + kBlkF], ubF1[r * GK],
                    kBlkF, GK / kBlkF - 1, foldPar);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceMax(ubMax, ubF1, kBlkF, kSub, 1, 1, GK / 8,
                           ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();

            // scale = bf16(max/127), floored at bf16-tiny.  All vector ops on the
            // kSub reduced values -- no GetValue, so the vector pipe never drains.
            Muls(ubScF, ubMax, 1.0f / kQMax, kSmall);
            PipeBarrier<PIPE_V>();
            Maxs(ubScF, ubScF, kTinyBf16, kSmall);
            PipeBarrier<PIPE_V>();
            // Round-trip through bf16 at OFFSET 0: a bf16 tensor indexed by the
            // sub-tile (r0 = 8) would sit at byte 16, and vector destinations must
            // be 32 B aligned -- that misalignment faults the core, it does not
            // silently misbehave.  The rounded scale is carried to the store in
            // fp32 (offset r0*4 = 32 B, aligned) and cast once after the loop.
            Cast(ubScBf, ubScF, RoundMode::CAST_ROUND, kSub);
            PipeBarrier<PIPE_V>();
            Cast(ubScF, ubScBf, RoundMode::CAST_NONE, kSub);
            PipeBarrier<PIPE_V>();
            Adds(ubScAll[r0], ubScF, 0.0f, kSub);
            PipeBarrier<PIPE_V>();
            // one 32 B block per row, all 8 lanes holding that row's scale
            Brcb(ubSBlk, ubScF, 1, {1, 8});
            PipeBarrier<PIPE_V>();

            // q = rint(x / s) per row, then clamp
            for (uint32_t r = 0; r < kSub; ++r) {
                Div(ubF1[r * GK], ubF0[r * GK], ubSBlk[r * 8],
                    kBlkF, GK / kBlkF, bcastPar);
            }
            PipeBarrier<PIPE_V>();
            Cast(ubI1, ubF1, RoundMode::CAST_RINT, kSubElems);
            PipeBarrier<PIPE_V>();
            Mins(ubI1, ubI1, 127, kSubElems);
            PipeBarrier<PIPE_V>();
            Maxs(ubI1, ubI1, -128, kSubElems);
            PipeBarrier<PIPE_V>();
            Cast(ubF0, ubI1, RoundMode::CAST_NONE, kSubElems);   // exact, |q|<=128
            PipeBarrier<PIPE_V>();

            // -sum_k q per row (exact in fp32: |sum| <= 128*GK << 2^24)
            Muls(ubF1, ubF0, -1.0f, kSubElems);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < kSub; ++r) {
                Add(ubF1[r * GK], ubF1[r * GK + kBlkF], ubF1[r * GK],
                    kBlkF, GK / kBlkF - 1, foldPar);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum(ubKs, ubF1, kBlkF, kSub, 1, 1, GK / 8);
            PipeBarrier<PIPE_V>();
            Cast(ubKsI[r0], ubKs, RoundMode::CAST_RINT, kSub);
            PipeBarrier<PIPE_V>();

            // MSD split: hi = floor(q/16) into ubF1, then lo = q - 16*hi - 8
            Muls(ubF1, ubF0, 1.0f / 16.0f, kSubElems);
            PipeBarrier<PIPE_V>();
            Cast(ubI1, ubF1, RoundMode::CAST_FLOOR, kSubElems);
            PipeBarrier<PIPE_V>();
            Cast(ubF1, ubI1, RoundMode::CAST_NONE, kSubElems);
            PipeBarrier<PIPE_V>();
            Cast(ubHalf, ubF1, RoundMode::CAST_NONE, kSubElems);
            PipeBarrier<PIPE_V>();
            Cast(ubHi4[off / 2].ReinterpretCast<int4b_t>(), ubHalf,
                 RoundMode::CAST_RINT, kSubElems);
            PipeBarrier<PIPE_V>();

            Muls(ubF1, ubF1, -16.0f, kSubElems);
            PipeBarrier<PIPE_V>();
            Add(ubF0, ubF0, ubF1, kSubElems);
            PipeBarrier<PIPE_V>();
            Adds(ubF0, ubF0, -8.0f, kSubElems);
            PipeBarrier<PIPE_V>();
            Cast(ubHalf, ubF0, RoundMode::CAST_NONE, kSubElems);
            PipeBarrier<PIPE_V>();
            Cast(ubLo4[off / 2].ReinterpretCast<int4b_t>(), ubHalf,
                 RoundMode::CAST_RINT, kSubElems);
            PipeBarrier<PIPE_V>();
        }

        // Already exactly bf16-representable, so this round is a no-op in value.
        Cast(ubScBf, ubScAll, RoundMode::CAST_ROUND, kBr);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(evVMte3);
        WaitFlag<HardEvent::V_MTE3>(evVMte3);

        // Destination offset uses the PADDED row pitch and the group's padded
        // start -- both from mg_kgeom.h, so the GEMM reads exactly what lands here.
        const uint64_t planeOff = (uint64_t)row0 * KbPad
                                + mgk::GroupOffsetElems(g, GK, mgk::kCubeKElemsInt4) / 2;
        if (!shortG) {
            DataCopyExtParams op{static_cast<uint16_t>(rowsStore),
                                 static_cast<uint32_t>(GK / 2), 0u,
                                 static_cast<uint32_t>(KbPad - GK / 2), 0u};
            DataCopyPad(hiGm[planeOff], ubHi4, op);
            DataCopyPad(loGm[planeOff], ubLo4, op);
        } else {
            // Store gPad/2 bytes per row out of a GK/2-byte UB row pitch -- one
            // burst per row, for the same reason as the load above.
            DataCopyExtParams op1{1u, static_cast<uint32_t>(gPad / 2), 0u, 0u, 0u};
            for (uint32_t r = 0; r < rowsStore; ++r) {
                DataCopyPad(hiGm[planeOff + (uint64_t)r * KbPad], ubHi4[r * (GK / 2)], op1);
                DataCopyPad(loGm[planeOff + (uint64_t)r * KbPad], ubLo4[r * (GK / 2)], op1);
            }
        }

        // Group-major layout puts this job's `rowsStore` scales and ksums
        // contiguously -- one burst each, instead of the 2-byte and 4-byte
        // scattered writes per row the per-(row,group) version did.  The stride
        // is Mp, the padded row count, since that is the tensor's row extent.
        DataCopyExtParams os{1u, static_cast<uint32_t>(rowsStore * sizeof(bfloat16_t)), 0u, 0u, 0u};
        DataCopyExtParams ok{1u, static_cast<uint32_t>(rowsStore * sizeof(int32_t)), 0u, 0u, 0u};
        DataCopyPad(sGm[(uint64_t)g * Mp + row0], ubScBf, os);
        DataCopyPad(ksGm[(uint64_t)g * Mp + row0], ubKsI, ok);

        SetFlag<HardEvent::MTE3_V>(evMte3V);
        WaitFlag<HardEvent::MTE3_V>(evMte3V);
    }

    pipe.ReleaseEventID<HardEvent::MTE2_V>(evM2V);
    pipe.ReleaseEventID<HardEvent::V_MTE2>(evVMte2);
    pipe.ReleaseEventID<HardEvent::V_MTE3>(evVMte3);
    pipe.ReleaseEventID<HardEvent::MTE3_V>(evMte3V);
    pipe.Destroy();
}
