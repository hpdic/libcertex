# SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# CertEx for cuVS: the CUDA C++ implementation, laid out as a cuVS module (include/cuvs/neighbors,
# src/neighbors/certex) so that it can move into cuVS unchanged. It builds against an installed
# cuVS, e.g. the libcuvs pip wheel:
#
#   SP=$(python -c "import site; print(site.getsitepackages()[0])")
#   cmake -S cpp -B build -DCMAKE_PREFIX_PATH="$SP/libcuvs/lib64/cmake;$SP/libraft/lib64/cmake;..."
#         -DCERTEX_CUTLASS_DIR=/path/to/cutlass
#   cmake --build build -j

cmake_minimum_required(VERSION 3.26)
# sm_80 (A100 and later: cp.async and mma.sync kernels) and sm_90a (Hopper: TMA and WGMMA kernels);
# the kernel is chosen at run time by the compute capability of the device
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;90a")
endif()
project(certex_cuvs VERSION 0.1.0 LANGUAGES C CXX CUDA)

option(CERTEX_BUILD_TESTS "Build the tests and benchmarks" ON)
option(CERTEX_NVTX "Enable NVTX markers (as CUVS_NVTX in cuVS)" OFF)
option(CERTEX_WERROR "Treat warnings as errors with the flags of the cuVS build" OFF)

# CUTLASS / CuTe headers: a local source tree, or the pinned release fetched at configure time
set(CERTEX_CUTLASS_DIR "" CACHE PATH "CUTLASS source tree (header-only include/ is used)")
if(NOT EXISTS "${CERTEX_CUTLASS_DIR}/include/cute/tensor.hpp")
  include(FetchContent)
  # SOURCE_SUBDIR without a CMakeLists.txt: download only, no CUTLASS build targets
  FetchContent_Declare(cutlass GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
                               GIT_TAG v4.1.0 GIT_SHALLOW TRUE SOURCE_SUBDIR headers-only)
  FetchContent_MakeAvailable(cutlass)
  set(CERTEX_CUTLASS_DIR "${cutlass_SOURCE_DIR}")
endif()

find_package(cuvs REQUIRED)
find_package(CUDAToolkit REQUIRED)

# A pip-installed cuVS may need a newer nvJitLink than the local toolkit provides; point
# CERTEX_NVJITLINK at that library (e.g. site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12).
set(CERTEX_NVJITLINK "" CACHE FILEPATH "nvJitLink library required by the installed cuVS")
set(CERTEX_RPATH "" CACHE STRING "extra run-time search paths (; separated)")
set(CMAKE_BUILD_RPATH "${CERTEX_RPATH}")
if(CERTEX_RPATH)
  # DT_RPATH (not RUNPATH), so that the libraries of a pip cuVS also find their own dependencies
  add_link_options("LINKER:--disable-new-dtags")
endif()

# tensor-core kernels (CUTLASS / CuTe: Hopper TMA and WGMMA, Ampere cp.async and mma.sync); plain
# C++ interface in kernels.hpp
add_library(certex_kernels OBJECT src/neighbors/certex/detail/screen_sm90.cu
                                  src/neighbors/certex/detail/screen_pingpong_sm90.cu
                                  src/neighbors/certex/detail/screen_sm80.cu
                                  src/neighbors/certex/detail/screen_int8.cu)
set_target_properties(certex_kernels PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
                                                POSITION_INDEPENDENT_CODE ON)
target_include_directories(certex_kernels PRIVATE "${CERTEX_CUTLASS_DIR}/include"
                           $<TARGET_PROPERTY:cuvs::cuvs,INTERFACE_INCLUDE_DIRECTORIES>)  # export.hpp
target_compile_options(certex_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

add_library(certex_cuvs SHARED src/neighbors/certex/certex.cu src/neighbors/certex/certex_serialize.cu
                               src/neighbors/certex/certex_shards.cu
                               src/neighbors/certex/certex_tune.cu
                               $<TARGET_OBJECTS:certex_kernels>)
set_target_properties(certex_cuvs PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
                                             CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON)
target_include_directories(certex_cuvs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(certex_cuvs PRIVATE
                       $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
# the nvJitLink of the installed cuVS precedes cuvs::cuvs, which also names the toolkit's copy
target_link_libraries(certex_cuvs PUBLIC ${CERTEX_NVJITLINK} cuvs::cuvs PRIVATE CUDA::cublas
                                         CUDA::cusolver)
if(CERTEX_WERROR)  # the warning flags of cuvs_objs in cuVS
  set(certex_werror $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations,-Wno-reorder
                    -Werror=all-warnings> $<$<COMPILE_LANGUAGE:CXX>:-Wall -Werror -Wno-error=deprecated-declarations
                    -Wno-reorder>)
  target_compile_options(certex_kernels PRIVATE ${certex_werror})
  target_compile_options(certex_cuvs PRIVATE ${certex_werror})
endif()
if(CERTEX_NVTX)
  target_compile_definitions(certex_cuvs PRIVATE NVTX_ENABLED)
  target_link_libraries(certex_cuvs PRIVATE CUDA::nvtx3)
endif()

# C API (c/, in the layout of the cuVS C API): libcertex_c on cuvs::c_api for the resource handle
# and the error text; the DLPack header from the system or the pinned release
option(CERTEX_BUILD_C_API "Build the C API (c/)" ON)
if(CERTEX_BUILD_C_API)
  find_package(cuvs REQUIRED COMPONENTS c_api)
  find_path(CERTEX_DLPACK_INCLUDE dlpack/dlpack.h)
  if(NOT CERTEX_DLPACK_INCLUDE)
    include(FetchContent)
    # SOURCE_SUBDIR without a CMakeLists.txt: download only
    FetchContent_Declare(dlpack GIT_REPOSITORY https://github.com/dmlc/dlpack.git
                                GIT_TAG v1.0 GIT_SHALLOW TRUE SOURCE_SUBDIR headers-only)
    FetchContent_MakeAvailable(dlpack)
    set(CERTEX_DLPACK_INCLUDE "${dlpack_SOURCE_DIR}/include" CACHE PATH "" FORCE)
  endif()
  add_library(certex_c SHARED ../c/src/neighbors/certex.cpp)
  set_target_properties(certex_c PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
                                            POSITION_INDEPENDENT_CODE ON)
  target_include_directories(certex_c PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../c/include"
                                             "${CERTEX_DLPACK_INCLUDE}")
  target_link_libraries(certex_c PUBLIC cuvs::c_api PRIVATE certex_cuvs)
endif()

# Examples (examples/cpp and examples/c): the smallest complete programs on the C++ module and the C API
option(CERTEX_BUILD_EXAMPLES "Build the C++ and C examples" ON)
if(CERTEX_BUILD_EXAMPLES)
  add_executable(certex_example_cpp ../examples/cpp/exact_search.cu)
  target_link_libraries(certex_example_cpp PRIVATE certex_cuvs)
  set_target_properties(certex_example_cpp PROPERTIES CUDA_STANDARD 20)
  if(CERTEX_BUILD_C_API)
    add_executable(certex_example_c ../examples/c/exact_search.c)
    target_link_libraries(certex_example_c PRIVATE certex_c CUDA::cudart)
  endif()
endif()

if(NOT CERTEX_BUILD_TESTS)
  return()
endif()

# tests (GoogleTest, as in cuVS): ctest, or the executables directly
enable_testing()
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git
                                  GIT_TAG v1.15.2 GIT_SHALLOW TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

function(certex_test NAME)
  add_executable(${NAME} ${ARGN})
  target_link_libraries(${NAME} PRIVATE certex_cuvs GTest::gtest GTest::gtest_main)
  set_target_properties(${NAME} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(${NAME} PRIVATE
                         $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
  add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

certex_test(NEIGHBORS_ANN_CERTEX_TEST tests/neighbors/ann_certex/test_float.cu
            tests/neighbors/ann_certex/test_half.cu tests/neighbors/ann_certex/test_int8_t.cu
            tests/neighbors/ann_certex/test_uint8_t.cu tests/neighbors/ann_certex/test_updates.cu
            tests/neighbors/ann_certex/test_sm80.cu tests/neighbors/ann_certex/test_shards.cu
            tests/neighbors/ann_certex/test_tune.cu)
certex_test(NEIGHBORS_CERTEX_KERNELS_TEST tests/neighbors/certex_kernels.cu)
if(CERTEX_BUILD_C_API)
  certex_test(NEIGHBORS_CERTEX_C_TEST ../c/tests/neighbors/certex_c.cu
              ../c/tests/neighbors/run_certex_c.c)
  target_link_libraries(NEIGHBORS_CERTEX_C_TEST PRIVATE certex_c)
endif()

add_executable(certex_kernel_bench bench/kernel_bench.cu)
target_link_libraries(certex_kernel_bench PRIVATE certex_cuvs CUDA::curand)
set_target_properties(certex_kernel_bench PROPERTIES CUDA_STANDARD 20)

add_executable(certex_bigann bench/certex_bigann.cu)
target_link_libraries(certex_bigann PRIVATE certex_cuvs)
set_target_properties(certex_bigann PROPERTIES CUDA_STANDARD 20)
target_compile_options(certex_bigann PRIVATE
                       $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
