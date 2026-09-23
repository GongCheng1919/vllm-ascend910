cmake_minimum_required(VERSION 3.16)
project(rans_kernels LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-8.5.0")
include(${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
# NOTE: always configure with -DCMAKE_BUILD_TYPE=<type>. With it empty, CANN's
# merge_mix_obj.sh runs `shift 2` with one argument left and dies under set -e
# printing nothing, surfacing only as "Error 1" on the *_merge_obj target.
ascendc_library(ans_fixed_decode_gemm_kernel SHARED kernels/ans_fixed_decode_gemm.cpp)
