// gemm_raw.h — pure raw-GEMM template kernel for cube instruction-rate
// comparison on 910B4. No quantization / dequantization, no scale math:
//   C = A x B   (A: [M,K] ND, B: [K,N] ND, C: [M,N] CT)
// Types:
//   gemm_s4s4 : T=int4b_t     CT=int32_t  (hardware mad_s4, s32 accumulate)
//   gemm_s8s8 : T=int8_t      CT=int32_t  (hardware s8s8,   s32 accumulate)
//   gemm_bf16 : T=bfloat16_t  CT=float    (hardware bf16,   f32 accumulate)
//
// Tile structure (per AIC core, N-split):
//   TILE_M = M (16..128, multiple of 16; single m-tile per kernel call)
//   TILE_N = 128 columns per Mmad
//   INNER_K = 256 (L0A/L0B budget fits all three dtypes at M<=128)
// GM -> L1 via Nd2Nz (MTE3), L1 -> L0 via LoadData2D (MTE2), Mmad, Fixpipe out.
// Loop structure copied from CANN's production matmul_act block_mmad
// ping-pong template, simplified to single-buffer (no L1/L0C ping-pong).

#include "kernel_operator.h"

using namespace AscendC;

namespace gemm_raw {

constexpr uint32_t TILE_N = 128;   // columns per Mmad / L0B tile
constexpr uint32_t INNER_K = 256;  // K per L1/L0 tile

// 32B / sizeof(T): C0 element count per 32B line.
template <typename T> struct C0 { static constexpr uint32_t Value = 32 / sizeof(T); };
template <> struct C0<int4b_t> { static constexpr uint32_t Value = 64; };

template <typename T, typename CT>
__aicore__ inline void GemmRawKernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t N, uint32_t K)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (GetSubBlockIdx() != 0) { return; }

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockDim = GetBlockNum();
    const uint32_t nTiles = N / TILE_N;
    const uint32_t tilesPerCore = (nTiles + blockDim - 1) / blockDim;
    const uint32_t myStart = blockIdx * tilesPerCore;
    if (myStart >= nTiles) { return; }
    const uint32_t myTiles = (myStart + tilesPerCore <= nTiles) ? tilesPerCore : (nTiles - myStart);
    if (blockIdx == 0) { printf("[gemm] enter M=%u N=%u K=%u tiles=%u\n", M, N, K, myTiles); }

    GlobalTensor<T> aGm, bGm;
    GlobalTensor<CT> cGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a), M * K);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b), K * N);
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(c), M * N);

    TPipe pipe;
    TBuf<TPosition::A1> l1aBuf;
    TBuf<TPosition::B1> l1bBuf;
    TBuf<TPosition::A2> l0aBuf;
    TBuf<TPosition::B2> l0bBuf;
    TBuf<TPosition::CO1> l0cBuf;
    pipe.InitBuffer(l1aBuf, M * INNER_K * sizeof(T));
    pipe.InitBuffer(l1bBuf, INNER_K * TILE_N * sizeof(T));
    pipe.InitBuffer(l0aBuf, M * INNER_K * sizeof(T));
    pipe.InitBuffer(l0bBuf, INNER_K * TILE_N * sizeof(T));
    pipe.InitBuffer(l0cBuf, M * TILE_N * sizeof(CT));

    LocalTensor<T> aL1 = l1aBuf.Get<T>();
    LocalTensor<T> bL1 = l1bBuf.Get<T>();
    LocalTensor<T> aL0 = l0aBuf.Get<T>();
    LocalTensor<T> bL0 = l0bBuf.Get<T>();
    LocalTensor<CT> cL0 = l0cBuf.Get<CT>();

    constexpr uint32_t aC0 = C0<T>::Value;
    // int4 is carried in int8 form for Nd2Nz (2 int4 per byte)
    constexpr uint32_t ndElem = (IsSameType<T, int4b_t>::value) ? 2 : 1;

    for (uint32_t t = 0; t < myTiles; t++) {
        const uint32_t n0 = (myStart + t) * TILE_N;
        // ---------- K loop: GM -> L1 (Nd2Nz) -> L0 (LoadData2D) -> Mmad ----------
        for (uint32_t k0 = 0; k0 < K; k0 += INNER_K) {
            // A [M, INNER_K] ND slice -> L1 NZ
            {
                Nd2NzParams nd;
                nd.ndNum = 1;
                nd.nValue = M;
                nd.dValue = INNER_K / ndElem;
                nd.srcNdMatrixStride = 1;
                nd.srcDValue = K / ndElem;
                nd.dstNzC0Stride = (M + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
                nd.dstNzNStride = 1;
                nd.dstNzMatrixStride = 1;
                DataCopy(aL1, aGm[k0 / ndElem], nd);
            }
            // B [INNER_K, TILE_N] ND slice -> L1 NZ
            {
                Nd2NzParams nd;
                nd.ndNum = 1;
                nd.nValue = INNER_K;
                nd.dValue = TILE_N / ndElem;
                nd.srcNdMatrixStride = 1;
                nd.srcDValue = N / ndElem;
                nd.dstNzC0Stride = (INNER_K + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
                nd.dstNzNStride = 1;
                nd.dstNzMatrixStride = 1;
                DataCopy(bL1, bGm[(k0 * N + n0) / ndElem], nd);
            }
            if (blockIdx == 0 && k0 == 0) { printf("[gemm] after nd2nz\n"); }
            PipeBarrier<PIPE_ALL>();

            // L1 -> L0A: (M, K) non-transposed
            {
                LoadData2DParamsV2 ld;
                ld.mStartPosition = 0;
                ld.kStartPosition = 0;
                ld.mStep = M / BLOCK_CUBE;
                if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                    ld.kStep = INNER_K / BLOCK_CUBE;
                } else {
                    ld.kStep = INNER_K / aC0;
                }
                ld.srcStride = M / BLOCK_CUBE;
                ld.dstStride = ld.mStep;
                ld.ifTranspose = false;
                LoadData<T>(aL0, aL1, ld);
            }
            PipeBarrier<PIPE_ALL>();
            // L1 -> L0B: (K, N) non-transposed
            {
                LoadData2DParamsV2 ld;
                ld.mStartPosition = 0;
                ld.kStartPosition = 0;
                ld.mStep = INNER_K / BLOCK_CUBE;
                if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                    ld.kStep = TILE_N / BLOCK_CUBE;
                    ld.dstStride = ld.kStep;
                } else {
                    ld.kStep = (TILE_N / BLOCK_CUBE) * 2;
                    ld.dstStride = ld.kStep >> 1;
                }
                ld.srcStride = INNER_K / BLOCK_CUBE;
                ld.ifTranspose = true;
                LoadData<T>(bL0, bL1, ld);
            }
            if (blockIdx == 0 && k0 == 0) { printf("[gemm] after loaddata\n"); }

            // Big Mmad: M x 128 x 256
            MmadParams mm;
            mm.m = M;
            mm.n = TILE_N;
            mm.k = INNER_K;
            mm.cmatrixInitVal = (k0 == 0);
            mm.cmatrixSource = false;
            Mmad(cL0, aL0, bL0, mm);
            if (blockIdx == 0 && k0 == 0) { printf("[gemm] after mmad\n"); }
            PipeBarrier<PIPE_ALL>();
        }

        // ---------- L0C -> GM (row-major nz2nd) at column n0 ----------
        {
            FixpipeParamsV220 fp;
            uint64_t c0 = 32 / sizeof(CT);  // 32B per C0 line
            fp.nSize = (TILE_N + c0 - 1) / c0 * c0;
            fp.mSize = M;
            fp.dstStride = N * sizeof(CT) / 32;  // GM row stride in 32B units
            fp.srcStride = (M + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
            fp.unitFlag = 3;  // FINAL_ACCUMULATION
            fp.ndNum = 1;
            fp.srcNdStride = 1;
            fp.dstNdStride = 1;
            Fixpipe<CT, CT>(cGm[n0], cL0, fp);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

}  // namespace gemm_raw
