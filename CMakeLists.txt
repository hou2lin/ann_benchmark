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

rapids_cpm_init()
include(cmake/thirdparty/get_cuvs.cmake)

# -------------- compile tasks ----------------- #
#add_executable(test_cagra src/test_cagra.cu)
#target_link_libraries(test_cagra PRIVATE raft::raft raft::compiled)

#add_executable(test_ivf_flat src/test_ivf_flat.cu)
#target_link_libraries(test_ivf_flat PRIVATE raft::raft raft::compiled)

#add_executable(test_ivf_pq src/test_ivf_pq.cu)
#target_link_libraries(test_ivf_pq PRIVATE raft::raft raft::compiled)

add_executable(test_ivf_flat_netease src/test_ivf_flat_netease.cu)
target_link_libraries(test_ivf_flat_netease PRIVATE cuvs::cuvs)

add_executable(test_ivf_pq_netease src/test_ivf_pq_netease.cu)
target_link_libraries(test_ivf_pq_netease PRIVATE cuvs::cuvs)

add_executable(test_cagra_netease src/test_cagra_netease.cu)
target_link_libraries(test_cagra_netease PRIVATE cuvs::cuvs)

add_executable(test_vamana_netease src/test_vamana_netease.cu)
target_link_libraries(test_vamana_netease PRIVATE cuvs::cuvs)


