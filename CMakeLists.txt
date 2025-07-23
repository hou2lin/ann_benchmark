# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

# Set CMake policies to suppress warnings
cmake_policy(SET CMP0167 NEW)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(raft_demo)

project(raft_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
else()
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O0 -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g")
endif()

# ------------- configure cuvs -----------------#

# Set MKL paths for diskann (Ubuntu package)
set(MKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Alternative MKL paths
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/mkl")
list(APPEND CMAKE_PREFIX_PATH "/usr/share/cmake/mkl")
list(APPEND CMAKE_PREFIX_PATH "/opt/intel/oneapi/mkl/latest")
list(APPEND CMAKE_PREFIX_PATH "/opt/intel/mkl")

# Set MKL environment variables
set(ENV{MKL_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(ENV{CMAKE_PREFIX_PATH} "${CMAKE_PREFIX_PATH}")

# Set MKL variables for Ubuntu package
set(MKL_INCLUDE_DIR "/usr/include/mkl")
set(MKL_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu")

# Add custom FindMKL module to module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Add diskann include directory
include_directories(build/_deps/diskann-src/include)

rapids_cpm_init()
include(cmake/thirdparty/get_cuvs.cmake)
include(cmake/thirdparty/get_diskann.cmake)
find_package(cuvs REQUIRED)
find_package(OpenMP REQUIRED)

# -------------- compile tasks ----------------- #
#add_executable(test_cagra src/test_cagra.cu)
#target_link_libraries(test_cagra PRIVATE raft::raft raft::compiled)

#add_executable(test_ivf_flat src/test_ivf_flat.cu)
#target_link_libraries(test_ivf_flat PRIVATE raft::raft raft::compiled)

#add_executable(test_ivf_pq src/test_ivf_pq.cu)
#target_link_libraries(test_ivf_pq PRIVATE raft::raft raft::compiled)

add_executable(test_ivf_flat_cuvs src/test_ivf_flat_cuvs.cu)
target_link_libraries(test_ivf_flat_cuvs PRIVATE cuvs::cuvs)

add_executable(test_ivf_pq_cuvs src/test_ivf_pq_cuvs.cu)
target_link_libraries(test_ivf_pq_cuvs PRIVATE cuvs::cuvs)

add_executable(test_cagra_cuvs src/test_cagra_cuvs.cu)
target_link_libraries(test_cagra_cuvs PRIVATE cuvs::cuvs)

add_executable(test_vamana_cuvs src/test_vamana_cuvs.cu)
target_link_libraries(test_vamana_cuvs cuvs diskann::diskann OpenMP::OpenMP_CXX)
target_include_directories(test_vamana_cuvs PRIVATE build/_deps/diskann-src/include)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
endif()


