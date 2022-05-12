# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: *
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. * Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. * Neither the name of NVIDIA CORPORATION nor the names
# of its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
# EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tritonlegobackend LANGUAGES C CXX)

#
# Options
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend." ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend." ON)
option(TRITON_ENABLE_NVTX "Include nvtx markers collection in backend." OFF)
set(TRITON_LEGO_LIB_PATHS
    ""
    CACHE
      PATH
      "Paths to Lego libraries. Multiple paths may be specified by separating them with a semicolon."
)
set(TRITON_LEGO_INCLUDE_PATHS
    ""
    CACHE
      PATH
      "Paths to Lego includes. Multiple paths may be specified by separating them with a semicolon."
)

set(TRITON_BACKEND_REPO_TAG
    "lego"
    CACHE STRING "Tag for triton-inference-server/backend repo.")
set(TRITON_CORE_REPO_TAG
    "lego"
    CACHE STRING "Tag for triton-inference-server/core repo.")
set(TRITON_COMMON_REPO_TAG
    "lego"
    CACHE STRING "Tag for triton-inference-server/common repo.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TRITON_LEGO_BACKEND_LIBNAME triton_lego)
set(TRITON_LEGO_BACKEND_INSTALLDIR ${CMAKE_INSTALL_PREFIX}/backends/lego)

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the transitive
# closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/Raphael-Hao/lego_common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG})
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/Raphael-Hao/lego_core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG})
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/Raphael-Hao/lego_backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG})
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
  set(CUDA_NVCC_FLAGS -std=c++11)
else()
  message(
    FATAL_ERROR "Lego backend requires TRITON_ENABLE_GPU=1, CMake will exit.")
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_NVTX})
  add_definitions(-DTRITON_ENABLE_NVTX=1)
endif() # TRITON_ENABLE_NVTX

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_lego.ldscript libtriton_lego.ldscript COPYONLY)

add_library(
  triton-lego-backend SHARED
  src/lego.cc
  src/lego_model.cc
  src/lego_model.h
  src/lego_model_instance.cc
  src/lego_model_instance.h
  src/lego_utils.cc
  src/lego_utils.h
  src/loader.cc
  src/loader.h
  src/logging.cc
  src/logging.h)

add_library(TritonLegoBackend::triton-lego-backend ALIAS triton-lego-backend)

target_include_directories(triton-lego-backend
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(triton-lego-backend
                           PRIVATE ${TRITON_LEGO_INCLUDE_PATHS})

target_compile_features(triton-lego-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-lego-backend
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-type-limits
    -Werror
    -Wno-deprecated-declarations>
    $<$<CXX_COMPILER_ID:MSVC>:/Wall
    /D_WIN32_WINNT=0x0A00
    /EHsc>)

# C/C++ defines that are used directly by this backend.
target_compile_definitions(triton-lego-backend PRIVATE TRITON_ENABLE_GPU=1)

set_target_properties(
  triton-lego-backend
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             OUTPUT_NAME ${TRITON_LEGO_BACKEND_LIBNAME}
             SKIP_BUILD_RPATH TRUE
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH_USE_LINK_PATH FALSE
             INSTALL_RPATH "$\{ORIGIN\}"
             LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_lego.ldscript
             LINK_FLAGS "-Wl,--version-script libtriton_lego.ldscript")

foreach(p ${TRITON_LEGO_LIB_PATHS})
  set(TRITON_LEGO_LDFLAGS ${TRITON_LEGO_LDFLAGS} "-L${p}")
endforeach(p)

find_library(NVINFER_LIBRARY NAMES nvinfer)
find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin)
target_link_libraries(
  triton-lego-backend
  PRIVATE triton-core-serverapi # from repo-core
          triton-core-serverstub # from repo-core
          triton-backend-utils # from repo-backend
          -lpthread
          ${NVINFER_LIBRARY}
          ${NVINFER_PLUGIN_LIBRARY})

target_link_libraries(triton-lego-backend PRIVATE ${TRITON_LEGO_LDFLAGS})

target_link_libraries(triton-lego-backend PRIVATE CUDA::cudart)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonLegoBackend)

install(
  TARGETS triton-lego-backend
  EXPORT triton-lego-backend-targets
  LIBRARY DESTINATION ${TRITON_LEGO_BACKEND_INSTALLDIR}
  ARCHIVE DESTINATION ${TRITON_LEGO_BACKEND_INSTALLDIR})

install(
  EXPORT triton-lego-backend-targets
  FILE TritonLegoBackendTargets.cmake
  NAMESPACE TritonLegoBackend::
  DESTINATION ${INSTALL_CONFIGDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonLegoBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonLegoBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonLegoBackendConfig.cmake
        DESTINATION ${INSTALL_CONFIGDIR})

#
# Export from build tree
#
export(
  EXPORT triton-lego-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonLegoBackendTargets.cmake
  NAMESPACE TritonLegoBackend::)

export(PACKAGE TritonLegoBackend)
