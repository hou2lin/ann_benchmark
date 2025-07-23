/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "index.h"
#include <omp.h>
#include <raft/distance/distance_types.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace DISKANN {

template <typename T>
class DiskANNMemory {
 public:
  struct SearchParam {
    uint32_t L_search;
    uint32_t num_threads = omp_get_max_threads() / 2;
  };

  DiskANNMemory(cuvs::distance::DistanceType metric, int dim) : metric_(metric), dim_(dim) {}

  void Load(const std::string& index_file, uint32_t num_threads, uint32_t search_l)
  {
    diskann::Metric diskann_metric;
    if (metric_ == cuvs::distance::DistanceType::L2Expanded) {
      diskann_metric = diskann::Metric::L2;
    } else if (metric_ == cuvs::distance::DistanceType::InnerProduct) {
      diskann_metric = diskann::Metric::INNER_PRODUCT;
    } else {
      throw std::runtime_error("Unsupported metric for DiskANN");
    }

    // Create parameters for the Index constructor with reasonable defaults
    auto index_params = std::make_shared<diskann::IndexWriteParameters>(
        100,    // search_list_size
        64,     // max_degree
        false,  // saturate_graph
        100,    // max_occlusion_size
        1.2f,   // alpha
        num_threads,  // num_threads
        100     // filter_list_size
    );
    auto search_params = std::make_shared<diskann::IndexSearchParams>(
        100,    // initial_search_list_size
        num_threads  // num_search_threads
    );
    
    mem_index_ = std::make_shared<diskann::Index<T>>(diskann_metric, dim_, 0, 
                                                    index_params, search_params);
    mem_index_->load(index_file.c_str(), num_threads, search_l);
  }

  void Search(const T* queries,
              int batch_size,
              int k,
              const SearchParam& params,
              int64_t* indices,
              float* distances) const
  {
#pragma omp parallel for schedule(dynamic, 1) num_threads(params.num_threads)
    for (int i = 0; i < batch_size; i++) {
      mem_index_->search(queries + i * dim_,
                         static_cast<size_t>(k),
                         params.L_search,
                         reinterpret_cast<uint64_t*>(indices + i * k),
                         distances + i * k);
    }
  }

 private:
  cuvs::distance::DistanceType metric_;
  int dim_;
  std::shared_ptr<diskann::Index<T>> mem_index_{nullptr};
};

}  // namespace DISKANN 