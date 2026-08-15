// probe_tiling.cpp — print TCubeTiling for a shape without touching the device.
#include <cstdio>
#include "adv_api/matmul/bmm_tiling.h"

enum class GemmDType { BF16 = 0, INT8 = 1, INT4 = 2 };

static AscendC::tiling::TCubeTiling ComputeTiling(GemmDType dt, int M, int N, int K, int coreNum)
{
    using namespace matmul_tiling;
    DataType aDt = DataType::DT_BF16;
    if (dt == GemmDType::INT8) aDt = DataType::DT_INT8;
    if (dt == GemmDType::INT4) aDt = DataType::DT_INT4;
    MultiCoreMatmulTiling mm;
    mm.SetDim(coreNum);
    mm.SetShape(M, N, K);
    AscendC::tiling::TCubeTiling t;
    auto ret = mm.GetTiling(t);
    std::fprintf(stderr, "[tiling] GetTiling ret=%ld\n", (long)ret);
    return t;
}

static void Dump(const AscendC::tiling::TCubeTiling& t)
{
#define P(f) std::fprintf(stderr, "  %-16s = %d\n", #f, (int)t.f)
    P(usedCoreNum); P(M); P(N); P(Ka); P(Kb);
    P(singleCoreM); P(singleCoreN); P(singleCoreK);
    P(baseM); P(baseN); P(baseK);
    P(depthA1); P(depthB1); P(stepM); P(stepN);
    P(batchM); P(batchN); P(singleBatchM); P(singleBatchN);
    P(stepKa); P(stepKb); P(dbL0A); P(dbL0B); P(dbL0C);
#undef P
}

int main(int argc, char** argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 16;
    int N = argc > 2 ? atoi(argv[2]) : 6912;
    int K = argc > 3 ? atoi(argv[3]) : 5120;
    int coreNum = argc > 4 ? atoi(argv[4]) : 2;
    std::fprintf(stderr, "shape M=%d N=%d K=%d core=%d\n", M, N, K, coreNum);
    for (int dt = 0; dt < 3; ++dt) {
        const char* name = dt == 0 ? "BF16" : (dt == 1 ? "INT8" : "INT4");
        std::fprintf(stderr, "== %s ==\n", name);
        Dump(ComputeTiling(static_cast<GemmDType>(dt), M, N, K, coreNum));
    }
    return 0;
}
