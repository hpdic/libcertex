# SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# The Python extension certex._core and the CUDA library it wraps, built by scikit-build-core
# (pip install .). cuVS, RAFT, and RMM come from the RAPIDS pip wheels of the build environment;
# at run time the extension finds them next to it in site-packages.

cmake_minimum_required(VERSION 3.26)
project(certex_python LANGUAGES CXX CUDA)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

# prefix paths and the nvJitLink of the RAPIDS wheels installed in the build environment
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import os, libcuvs, libraft, librmm, rapids_logger, libkvikio; \
print(';'.join(os.path.dirname(m.__file__) for m in (libcuvs, libraft, librmm, rapids_logger, libkvikio)))"
  OUTPUT_VARIABLE RAPIDS_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
foreach(d IN LISTS RAPIDS_DIRS)
  list(APPEND CMAKE_PREFIX_PATH "${d}/lib64/cmake" "${d}/lib64/rapids/cmake")
endforeach()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import os, nvidia.nvjitlink as m; print(os.path.join(list(m.__path__)[0], 'lib', 'libnvJitLink.so.12'))"
  OUTPUT_VARIABLE NVJITLINK OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE NO_NVJITLINK)
if(NOT NO_NVJITLINK AND EXISTS "${NVJITLINK}")
  set(CERTEX_NVJITLINK "${NVJITLINK}" CACHE FILEPATH "" FORCE)
endif()

# the wheels sit next to the certex package in site-packages
set(CERTEX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CERTEX_BUILD_C_API OFF CACHE BOOL "" FORCE)  # the wheel ships the Python binding only
set(CERTEX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CERTEX_RPATH "" CACHE STRING "" FORCE)
add_link_options("LINKER:--disable-new-dtags")
add_subdirectory(../cpp cpp)
set(SITE_RPATH "$ORIGIN;$ORIGIN/../libcuvs/lib64;$ORIGIN/../libraft/lib64;$ORIGIN/../librmm/lib64"
               "$ORIGIN/../rapids_logger/lib64;$ORIGIN/../libkvikio/lib64;$ORIGIN/../nvidia/nvjitlink/lib"
               "$ORIGIN/../nvidia/cublas/lib;$ORIGIN/../nvidia/cusolver/lib;$ORIGIN/../nvidia/cusparse/lib")
set_target_properties(certex_cuvs PROPERTIES INSTALL_RPATH "${SITE_RPATH}")

execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_core NB_STATIC STABLE_ABI src/core.cpp)
set_target_properties(_core PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
                                       INSTALL_RPATH "${SITE_RPATH}")
target_link_libraries(_core PRIVATE certex_cuvs)

install(TARGETS _core certex_cuvs LIBRARY DESTINATION certex)
