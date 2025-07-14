/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cstdio>

#include <raft/random/rng.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include "utils.h"
#include "groudtruth.h"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/linalg/add.cuh>
#include "bindata.h"

#define _PRINT_ORIGINAL_DATA 0

void printMem(size_t freeMem, size_t totalMem, std::string msg) {
   std::cout<< "[LOG_INFO] " << msg << "  freeMem is : " <<
    freeMem /  (1024 * 1024 * 1024) << " GB "<< "  totalMem is : " 
    << totalMem / (1024 * 1024 * 1024) 
    << " GB " << " using Mem is : " 
    << (totalMem - freeMem) / (1024 * 1024 * 1024) << " GB " 
    << " freeMem all bytes is : " << freeMem  << 
    " totalMem all bytes is : " << totalMem <<  std::endl;
}

// Edge operation to replace invalid edges with edge to node 0
struct edge_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in, UnusedArgs...) const
  {
    return in == raft::upper_bound<Type>() ? Type(0) : in;
  }
};

template<typename T>
void dispatcher_raft_vamana(int64_t n_samples, int64_t n_dim, int64_t topk, int64_t n_queries, int64_t batch_size, 
                            int64_t graph_degree, int64_t visited_size, float max_fraction, int64_t vamana_iters, 
                            int64_t search_width, int64_t itopk_size, int64_t max_iterations, int if_read_exited_data, int split_bast_set_rows) {
    float mean = 0.1;
    float sigma = 1.0;
    uint64_t seed = 233376;
    size_t freeMem, totalMem;
    
    raft::resources dev_resources;
    cudaMemGetInfo(&freeMem, &totalMem); 
    printMem(freeMem, totalMem, "初始化开始: ");
    
    raft::resource::set_workspace_to_pool_resource(
       dev_resources, std::make_optional<std::size_t>(8 * 1024 * 1024 * 1024ull));
    
    DATA::BinFile<T> bindata(split_bast_set_rows);
    if (if_read_exited_data) {
        std::string  base_file = "./base.fbin";
        std::string  query_file = "./query.fbin";
        bindata.open_file(base_file, true);
        bindata.open_file(query_file, false);
        bindata.read_base_set();
        bindata.read_query_set();
        n_samples = bindata.base_set_nrows;
        n_dim = bindata.base_set_ndims;
        n_queries = bindata.query_set_nrows;
    }
    
    auto dataset = raft::make_device_matrix<T>(dev_resources, n_samples, n_dim);
    auto queries = raft::make_device_matrix<T>(dev_resources, n_queries, n_dim);
    auto result_ids = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);
    auto result_distance = raft::make_device_matrix<float>(dev_resources, n_queries, topk);
    
    int64_t *h_result_ids = nullptr;
    RAFT_CUDA_TRY(cudaHostAlloc(&h_result_ids, sizeof(int64_t) * n_queries * topk, cudaHostAllocDefault));
    float *h_result_distance = nullptr;
    RAFT_CUDA_TRY(cudaHostAlloc(&h_result_distance, sizeof(float) * n_queries * topk, cudaHostAllocDefault));
    
    auto metric = cuvs::distance::DistanceType::L2Expanded;
    
    // Generate or load dataset
    if (!if_read_exited_data) {
        {
            raft::random::RngState rng_state(seed);
            raft::random::normal(dev_resources, rng_state, 
                                dataset.data_handle(), dataset.size(), mean, sigma);
            RAFT_CUDA_TRY(cudaDeviceSynchronize());
        }
        
        {
            raft::random::RngState rng_state(seed + 19987);
            raft::random::normal(dev_resources, rng_state,
                                queries.data_handle(), queries.size(), mean, sigma);
            RAFT_CUDA_TRY(cudaDeviceSynchronize());
        }
    } else {
        RAFT_CUDA_TRY(cudaMemcpy(dataset.data_handle(), bindata.base_set, bindata.base_set_nrows * bindata.base_set_ndims * sizeof(T), cudaMemcpyHostToDevice));
        RAFT_CUDA_TRY(cudaMemcpy(queries.data_handle(), bindata.query_set, bindata.query_set_nrows * bindata.query_set_ndims * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Ground truth with brute force
    GT::GroudTruth<T> bf_gt(topk, n_queries);
    
    cuvs::neighbors::brute_force::index_params bf_idx_params;
    auto bf_index = cuvs::neighbors::brute_force::build(dev_resources, bf_idx_params, raft::make_const_mdspan(dataset.view()));
    
    cuvs::neighbors::brute_force::search_params bf_srch_params;
    cuvs::neighbors::brute_force::search(dev_resources, bf_srch_params, bf_index, 
                                        raft::make_const_mdspan(queries.view()), result_ids.view(), result_distance.view());
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    
    std::vector<int64_t> bf_results_h(n_queries * topk);
    RAFT_CUDA_TRY(cudaMemcpy(bf_results_h.data(), result_ids.data_handle(), 
                               sizeof(int64_t) * n_queries * topk, cudaMemcpyDefault));

    // Build Vamana index
    using namespace cuvs::neighbors;
    vamana::index_params index_params;
    index_params.metric = metric;
    index_params.graph_degree = graph_degree;
    index_params.visited_size = visited_size;
    index_params.max_fraction = max_fraction;
    index_params.vamana_iters = vamana_iters;

    cudaMemGetInfo(&freeMem, &totalMem); 
    printMem(freeMem, totalMem, "构建索引之前: ");
    
    UTIL::Timer ttt(true);
    ttt.update_time();
    auto vamana_index = vamana::build(dev_resources, index_params, 
                                    raft::make_const_mdspan(dataset.view()));
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    float build_time = ttt.elapsed_ms();
    
    std::cout << std::endl;
    std::cout << "[LOG_INFO] 构建Vamana索引时间 : " << (build_time / 1000.0f) << std::endl;
    std::cout << "[LOG_INFO] Vamana索引包含 " << vamana_index.size() << " 个向量" << std::endl;
    std::cout << "[LOG_INFO] Vamana图度数 " << vamana_index.graph_degree() << ", 图大小 ["
              << vamana_index.graph().extent(0) << ", " << vamana_index.graph().extent(1) << "]" << std::endl;
    std::cout << std::endl;
    
    cudaMemGetInfo(&freeMem, &totalMem); 
    printMem(freeMem, totalMem, "构建索引之后: ");
    
    // Convert Vamana graph to CAGRA index for search
    auto graph_valid = raft::make_device_matrix<uint32_t, int64_t>(
        dev_resources, vamana_index.graph().extent(0), vamana_index.graph().extent(1));
    raft::linalg::map(dev_resources, graph_valid.view(), edge_op{}, vamana_index.graph());
    
    auto cagra_index = cagra::index<T, uint32_t>(dev_resources,
                                                 metric,
                                                 raft::make_const_mdspan(dataset.view()),
                                                 raft::make_const_mdspan(graph_valid.view()));
    
    // Search using CAGRA on Vamana graph
    cagra::search_params search_params;
    search_params.itopk_size = itopk_size;
    search_params.max_iterations = max_iterations;
    search_params.search_width = search_width;
    search_params.algo = cagra::search_algo::AUTO;
    
    const std::size_t num_batches = (n_queries - 1) / batch_size + 1;
    UTIL::Perf perf(n_queries, batch_size);
    
    cudaMemGetInfo(&freeMem, &totalMem); 
    printMem(freeMem, totalMem, "search之前:   ");

    for (int batch_id = 0; batch_id < num_batches; ++ batch_id) {
       const std::size_t row = batch_id * batch_size;
       const std::size_t actual_batch_size = (batch_id == num_batches - 1) ? n_queries - row : batch_size;
       
       perf.timer.update_time();
       
       auto queries_v = raft::make_device_matrix_view<const T, int64_t>((queries.data_handle() + row * n_dim), actual_batch_size, n_dim);
       auto neighbors_v = raft::make_device_matrix_view<uint32_t, int64_t>((reinterpret_cast<uint32_t*>(result_ids.data_handle()) + row * topk), actual_batch_size, topk);
       auto distances_v = raft::make_device_matrix_view<float, int64_t>((result_distance.data_handle() + row * topk), actual_batch_size, topk);
       
       cuvs::neighbors::cagra::search(dev_resources, search_params, cagra_index, queries_v, neighbors_v, distances_v);

       RAFT_CUDA_TRY(cudaDeviceSynchronize());
       if (actual_batch_size != batch_size) continue;
       perf.setTime(perf.timer.elapsed_ms());
    }
    
    auto ret = perf.getPerfResult();
    cudaMemGetInfo(&freeMem, &totalMem); 
    printMem(freeMem, totalMem, "search之后:  ");
    
    // Copy results back to host
    auto stream = raft::resource::get_cuda_stream(dev_resources);
    RAFT_CUDA_TRY(cudaMemcpyAsync(h_result_ids, result_ids.data_handle(), 
                            sizeof(int64_t) * n_queries * topk, cudaMemcpyDefault, stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(h_result_distance, result_distance.data_handle(),
                            sizeof(float) * n_queries * topk, cudaMemcpyDefault, stream));
    raft::resource::sync_stream(dev_resources);
   
    // Compute recall
    float recall_ratio_ret = bf_gt.compute_recall_ratio(bf_results_h.data(), h_result_ids);
    
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "[LOG_INFO] qps is : " << std::get<0>(ret) << ", all_run_search_time is :  " 
       << std::get<1>(ret) << ", avg_search_time is :  " << std::get<2>(ret) << ", best_search_time_p99 is :  " 
       << std::get<3>(ret) << ", recall_ratio_ret is : " << recall_ratio_ret << std::endl;
    
    RAFT_CUDA_TRY(cudaFreeHost(h_result_ids));
    RAFT_CUDA_TRY(cudaFreeHost(h_result_distance));
}

int main(int argc, char* argv[]) {
    UTIL::CommandLineParser parser(argc, argv);
    
    // Default parameters
    int64_t n_samples = 10000000;
    int64_t n_dim = 768;
    int64_t topk = 10;
    int64_t n_queries = 10;
    int batch_size = 10;
    
    // Vamana parameters
    int64_t graph_degree = 64;
    int64_t visited_size = 128;
    float max_fraction = 0.06;
    int64_t vamana_iters = 1;
    
    // CAGRA search parameters
    int64_t search_width = 1;
    int64_t itopk_size = 64;
    int64_t max_iterations = 0;
    
    int split_bast_set_rows = 0;
    int if_read_exited_data = 0;
    
    // Parse command line arguments
    if (parser.HasOption("--num")) {
       n_samples = std::stoi(parser.GetOptionValue("--num"));
    }
    if (parser.HasOption("--dim")) {
       n_dim = std::stoi(parser.GetOptionValue("--dim"));
    }
    if (parser.HasOption("--topk")) {
       topk = std::stoi(parser.GetOptionValue("--topk"));
    }
    if (parser.HasOption("--queries")) {
       n_queries = std::stoi(parser.GetOptionValue("--queries"));
    }
    if (parser.HasOption("--graph_degree")) {
       graph_degree = std::stoi(parser.GetOptionValue("--graph_degree"));
    }
    if (parser.HasOption("--visited_size")) {
       visited_size = std::stoi(parser.GetOptionValue("--visited_size"));
    }
    if (parser.HasOption("--max_fraction")) {
       max_fraction = std::stof(parser.GetOptionValue("--max_fraction"));
    }
    if (parser.HasOption("--vamana_iters")) {
       vamana_iters = std::stoi(parser.GetOptionValue("--vamana_iters"));
    }
    if (parser.HasOption("--search_width")) {
       search_width = std::stoi(parser.GetOptionValue("--search_width"));
    }
    if (parser.HasOption("--itopk_size")) {
       itopk_size = std::stoi(parser.GetOptionValue("--itopk_size"));
    }
    if (parser.HasOption("--max_iterations")) {
       max_iterations = std::stoi(parser.GetOptionValue("--max_iterations"));
    }
    if (parser.HasOption("--batch_size")) {
       batch_size = std::stoi(parser.GetOptionValue("--batch_size"));
    }
    if (parser.HasOption("--if_read_exited_data")) {
       if_read_exited_data = std::stoi(parser.GetOptionValue("--if_read_exited_data"));
    }
    if (parser.HasOption("--split_bast_set_rows")) {
       split_bast_set_rows = std::stoi(parser.GetOptionValue("--split_bast_set_rows"));
    }
    
    std::cout << "[LOG_INFO] num is : " << n_samples << ", dim is : " << n_dim << ", topk is : " << topk << ", queries is : " << n_queries  << ", batch_size is : " << batch_size << std::endl;
    std::cout << "[LOG_INFO] graph_degree is : " << graph_degree << ", visited_size is : " << visited_size << ", max_fraction is : " << max_fraction << ", vamana_iters is : " << vamana_iters << std::endl;
    std::cout << "[LOG_INFO] search_width is : " << search_width << ", itopk_size is : " << itopk_size << ", max_iterations is : " << max_iterations << std::endl;
    
    dispatcher_raft_vamana<float>(n_samples, n_dim, topk, n_queries, batch_size, graph_degree, visited_size, max_fraction, vamana_iters, search_width, itopk_size, max_iterations, if_read_exited_data, split_bast_set_rows); 

    return 0;
} 