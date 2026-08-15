cmake_minimum_required(VERSION 3.18)
project(cann_kernel_lab LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(SOC_VERSION "ascend910b4" CACHE STRING "Ascend SoC version")
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-8.5.0")
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

set(_CANN_INC ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include)
set(_CANN_LIB ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64)

# One ascendc_library per dtype; all share kernels/gemm_raw.h
ascendc_library(gemm_s4s4_kernel STATIC kernels/gemm_s4s4.cpp)
ascendc_library(gemm_s8s8_kernel STATIC kernels/gemm_s8s8.cpp)
ascendc_library(gemm_bf16_kernel STATIC kernels/gemm_bf16.cpp)

add_executable(gemm_test
    host/main_gemm.cpp
    host/ref_gemm.cpp
)
target_include_directories(gemm_test PRIVATE ${_CANN_INC} host)
target_link_directories(gemm_test PRIVATE ${_CANN_LIB})
target_link_libraries(gemm_test PRIVATE
    gemm_s4s4_kernel
    gemm_s8s8_kernel
    gemm_bf16_kernel
    ascendcl
    runtime
)
