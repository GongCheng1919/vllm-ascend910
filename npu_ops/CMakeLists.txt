cmake_minimum_required(VERSION 3.16.0)
project(vllm_w4a8_npu_ops)
set(LINUX TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(_GLIBCXX_USE_CXX11_ABI ${GLIBCXX_USE_CXX11_ABI})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")

set(SOC_VERSION "ascend910b4" CACHE STRING "system on chip type")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN root" FORCE)
else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-8.5.0" CACHE PATH "CANN root" FORCE)
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under ${ASCEND_CANN_PACKAGE_PATH}")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

set(_LIB_NAME vllm_w4a8_npu_ops)

# The kernels are NOT copied here: int4_cube_lab is the single source of truth,
# and its C++ harness is what verifies them (CPU reference, race stress, and the
# fakequant_lab seam).  A copy would let the verified source and the deployed one
# drift apart silently.
# Kernels are COPIED here rather than referenced from ../int4_cube_lab: the two
# trees must be kept in sync by hand, but a cross-tree relative path made the
# build fragile.  int4_cube_lab remains the place they are verified.
set(_KDIR kernel)

set(_KERNEL_LIBS)

# GEMM: GK=1024 asymmetric, one build per TILE_M (the host dispatches on M).
foreach(TM 16 64 128)
    ascendc_library(midgroup_w4a8_gemm_m${TM}_g1024_asym_kernel STATIC
                    ${_KDIR}/midgroup_w4a8_gemm_m${TM}_g1024_asym.cpp)
    list(APPEND _KERNEL_LIBS midgroup_w4a8_gemm_m${TM}_g1024_asym_kernel)
endforeach()

# Activation quantiser + MSD split (AIV-only).
ascendc_library(midgroup_quant_a_g1024_kernel STATIC ${_KDIR}/midgroup_quant_a_g1024.cpp)
list(APPEND _KERNEL_LIBS midgroup_quant_a_g1024_kernel)

file(GLOB _HOST_SRCS ${CMAKE_CURRENT_LIST_DIR}/host/*.cpp)
add_library(${_LIB_NAME} SHARED ${_HOST_SRCS})

target_link_libraries(${_LIB_NAME} PRIVATE
  ${_KERNEL_LIBS}
  torch_npu
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
)
target_link_directories(${_LIB_NAME} PRIVATE
  ${TORCH_PATH}/lib
  ${TORCH_NPU_PATH}/lib
)
target_include_directories(${_LIB_NAME} PRIVATE
  ${TORCH_NPU_PATH}/include
  ${TORCH_PATH}/include
  ${TORCH_PATH}/include/torch/csrc/api/include
)
install(TARGETS ${_LIB_NAME} LIBRARY DESTINATION lib)
