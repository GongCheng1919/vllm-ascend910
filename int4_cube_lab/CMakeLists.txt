cmake_minimum_required(VERSION 3.18)
project(int4_cube_lab LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(SOC_VERSION "ascend910b4" CACHE STRING "Ascend SoC version")
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-8.5.0")
endif()
message(STATUS "SoC: ${SOC_VERSION}")
message(STATUS "CANN root: ${ASCEND_CANN_PACKAGE_PATH}")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under ${ASCEND_CANN_PACKAGE_PATH}")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

set(_CANN_INC ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
set(_CANN_LIB ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64)

find_package(OpenMP)

function(add_op_test op_name)
    ascendc_library(${op_name}_kernel STATIC kernels/${op_name}.cpp)
    add_executable(${op_name}_test
        host/main_${op_name}.cpp
        host/gemm_ref.cpp)
    target_include_directories(${op_name}_test PRIVATE ${_CANN_INC} host)
    target_link_directories(${op_name}_test PRIVATE ${_CANN_LIB})
    target_link_libraries(${op_name}_test PRIVATE ${op_name}_kernel ascendcl runtime)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${op_name}_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# TILE_M=128 (compute-bound shapes) plus short-tile builds for decode-shaped M.
add_op_test(perchannel_int4_gemm)
add_op_test(perchannel_int8_gemm)
add_op_test(perchannel_int4_gemm_m64)
add_op_test(perchannel_int8_gemm_m64)
add_op_test(perchannel_int4_gemm_m16)
add_op_test(perchannel_int8_gemm_m16)

# W4A8: int8 activation via MSD split into two int4 planes, int4 weight.
add_op_test(perchannel_w4a8_gemm)
add_op_test(perchannel_w4a8_gemm_m64)
add_op_test(perchannel_w4a8_gemm_m32)
add_op_test(perchannel_w4a8_gemm_m16)

# P0 probe: mid-group W4A8, TILE_M x GK grid.
add_op_test(midgroup_w4a8_gemm_m16_g256)
add_op_test(midgroup_w4a8_gemm_m16_g512)
add_op_test(midgroup_w4a8_gemm_m16_g1024)
add_op_test(midgroup_w4a8_gemm_m64_g256)
add_op_test(midgroup_w4a8_gemm_m64_g512)
add_op_test(midgroup_w4a8_gemm_m64_g1024)
add_op_test(midgroup_w4a8_gemm_m128_g256)
add_op_test(midgroup_w4a8_gemm_m128_g512)
add_op_test(midgroup_w4a8_gemm_m128_g1024)

# P3 baseline: GK=1024 with an asymmetric weight (per-group zero point).
# Same source, ASYM_CFG=1 -- the sym builds above are the A/B control.
add_op_test(midgroup_w4a8_gemm_m16_g1024_asym)
add_op_test(midgroup_w4a8_gemm_m64_g1024_asym)
add_op_test(midgroup_w4a8_gemm_m128_g1024_asym)

# P5 ablation: same AIC, AIV reduced to the flag handshake.  Timing only --
# the output is garbage by construction (MGCKPT/P5_KERNEL_OPT.md §4).
add_op_test(midgroup_w4a8_gemm_m16_g1024_asym_noaiv)
add_op_test(midgroup_w4a8_gemm_m16_g1024_asym_nofix)
add_op_test(midgroup_w4a8_gemm_m16_g1024_asym_dmaonly)

# P5 (2026-08-14 reopen, D4): the same ladder at TILE_M=128, for the LARGE-M
# arm (M>=512) where the group-flush tax is 1.34-1.47x instead of 1.05-1.12x.
add_op_test(midgroup_w4a8_gemm_m128_g1024_asym_noaiv)
add_op_test(midgroup_w4a8_gemm_m128_g1024_asym_nofix)
add_op_test(midgroup_w4a8_gemm_m128_g1024_asym_dmaonly)

# P5 D6: the fix candidate -- split the stacked MSD flush into two 64 KB L0C
# tiles so kL0CDepth is 2 again at TILE_M=128.  Same tiling, same DMA traffic,
# same workspace bytes: this one MUST pass --check bit-exact.
add_op_test(midgroup_w4a8_gemm_m128_g1024_asym_split)
# D8: two-pass ordering of the same split -- the one that actually restores
# CROSS-GROUP L0C double buffering.
add_op_test(midgroup_w4a8_gemm_m128_g1024_asym_split2)
# D9: L2-banded tile traversal, alone and combined with the two-pass flush.
add_op_test(midgroup_w4a8_gemm_m128_g1024_asym_band)
add_op_test(midgroup_w4a8_gemm_m128_g1024_asym_band2)

# D10 SPARE: bf16 workspace flush.  NOT bit-exact -- built and measured so it
# can be enabled once its accuracy budget is paid.  Never make this default.
add_op_test(midgroup_w4a8_gemm_m128_g1024_asym_bf16ws)

# P5 D8: TILE_M=64 already has kL0CDepth=2 (cross-group L0C double buffering,
# the structure nitro's mid_group_gemm_fwd.cpp relies on).  Ladder to test
# whether that structure alone hides the flush.
add_op_test(midgroup_w4a8_gemm_m64_g1024_asym_nofix)
add_op_test(midgroup_w4a8_gemm_m64_g1024_asym_dmaonly)

# Activation-side quantiser: bf16 -> two int4 planes + per-group scale/ksum.
add_op_test(midgroup_quant_a_g1024)
add_op_test(midgroup_w4a8_gemm_m16_g1024_asym_dmaonly_gkb512)
add_op_test(midgroup_w4a8_gemm_m16_g1024_asym_gkb512)
add_op_test(midgroup_w4a8_gemm_m16_g1024_asym_dmaonly_gkb1536)
add_op_test(midgroup_w4a8_gemm_m16_g1024_asym_gkb1536)
