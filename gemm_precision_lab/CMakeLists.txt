cmake_minimum_required(VERSION 3.18)
project(gemm_precision_lab LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# ---- SoC + CANN paths ----
set(SOC_VERSION "ascend910b4" CACHE STRING "Ascend SoC version")
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-8.5.0")
endif()
message(STATUS "SoC: ${SOC_VERSION}")
message(STATUS "CANN root: ${ASCEND_CANN_PACKAGE_PATH}")

# ---- ascendc.cmake (provides ascendc_library) ----
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

# ---- One GEMM op: kernel + host test ----
# Expects kernels/<op>.cpp (MatmulImpl GEMM) + host/main_<op>.cpp
function(add_gemm_op op_name)
    ascendc_library(${op_name}_kernel STATIC kernels/${op_name}.cpp)
    add_executable(${op_name}_test host/main_${op_name}.cpp)
    target_include_directories(${op_name}_test PRIVATE ${_CANN_INC} host)
    target_link_directories(${op_name}_test PRIVATE ${_CANN_LIB})
    target_link_libraries(${op_name}_test PRIVATE
        ${op_name}_kernel
        ascendcl
        runtime
        ${_CANN_LIB}/libtiling_api.a
    )
endfunction()

# ---- Register ops ----
add_gemm_op(gemm_bf16)
add_gemm_op(gemm_int8)
add_gemm_op(gemm_int4)
