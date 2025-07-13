#pragma once
#include <algorithm>
#include <iostream>
#include <vector>
#include <raft/core/resource/cuda_stream.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/brute_force.cuh>
#include <cstddef>
#include <iostream>
#include <vector>
#include <rmm/device_buffer.hpp>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <rmm/device_uvector.hpp>
#include "cuda_runtime.h"


namespace GT {

template<typename T>
class GroudTruth {
public:
    GroudTruth(int topk, int query_nums): topk(topk), query_nums(query_nums){};

   float compute_recall_ratio(const int64_t* gt_results, const int64_t * output_results) {
    float set_cnt = 0.0;
    for (int i = 0; i < this->query_nums; i ++) {
        std::vector<int> gt_vec(gt_results + i * this->topk, gt_results + (i + 1) * this->topk); 
        for (int j = 0; j < this->topk; j ++) {
            if (find(gt_vec.begin(), gt_vec.end(), output_results[i * this->topk + j]) != gt_vec.end()) {
                set_cnt ++;
            }
        }
    }
    return (set_cnt) / float(this->query_nums * this->topk);
}

private:
    int topk;
    int query_nums;
};

} //end GT