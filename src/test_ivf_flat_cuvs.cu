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
#include <filesystem>
#include <memory>
 
 #include <cuvs/neighbors/ivf_flat.hpp>
 #include <cuvs/neighbors/brute_force.hpp>
 #include <raft/distance/distance_types.hpp>
 #include <raft/random/rng.cuh>
 #include <raft/util/cuda_rt_essentials.hpp>
 #include <raft/core/device_mdarray.hpp>
 #include <raft/core/resources.hpp>
 #include "utils.h"
 #include "groudtruth.h"
 #include "bindata.h"


 #define _PRINT_ORIGINAL_DATA 0
 #define USE_FLOAT 1
 #define USE_INT8_T 0
 #define USE_UINT8_T 0
 
 void printMem(size_t freeMem, size_t totalMem, std::string msg) {
    std::cout<< "[LOG_INFO] " << msg << "  freeMem is : " <<
     freeMem /  (1024 * 1024 * 1024) << " GB "<< "  totalMem is : " 
     << totalMem / (1024 * 1024 * 1024) 
     << " GB " << " using Mem is : " 
     << (totalMem - freeMem) / (1024 * 1024 * 1024) << " GB " 
     << " freeMem all bytes is : " << freeMem  << 
     " totalMem all bytes is : " << totalMem <<  std::endl;
}

//dispatcher_raft_ivf<float>(n_samples, n_dim, topk, n_queries, n_lists, n_probes, run_count);
template<typename T>
void dispatcher_raft_ivf(int64_t n_samples, int64_t n_dim, int64_t topk, int64_t n_queries, int64_t n_lists, int64_t n_probes, int64_t batch_size, int if_read_exited_data, int split_bast_set_rows) {
     float mean = 0.1;
     float sigma = 1.0;
     uint64_t seed = 233376;
     size_t freeMem, totalMem; //used to cache freemem and totalmem
    /* initialization */
     raft::resources dev_resources;
     cudaMemGetInfo(&freeMem, &totalMem); 
     printMem(freeMem, totalMem, "初始化开始: ");
     // Initialize resources - no need to explicitly get cublas handle
     /* create input and output arrays on GPU */
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
     auto dataset = raft::make_device_matrix<T>(dev_resources, n_samples, n_dim);        // [N, D] * T
     auto queries = raft::make_device_matrix<T>(dev_resources, n_queries, n_dim);        // [Nq, D] * T
     auto result_ids = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);    // [Nq, topk] * int64
     auto result_distance = raft::make_device_matrix<float>(dev_resources, n_queries, topk); // [Nq, topk] * float
     int64_t *h_result_ids = nullptr;
     RAFT_CUDA_TRY(cudaHostAlloc(&h_result_ids, sizeof(int64_t) * n_queries * topk, cudaHostAllocDefault));
     float *h_result_distance = nullptr;
     RAFT_CUDA_TRY(cudaHostAlloc(&h_result_distance, sizeof(float) * n_queries * topk, cudaHostAllocDefault));
     auto metric = cuvs::distance::DistanceType::L2Unexpanded;
     if (!if_read_exited_data) {
      // set dataset
      {
            raft::random::RngState rng_state(seed);
            std::vector<T> means(n_dim, mean);
            auto d_means = raft::make_device_vector<T>(dev_resources, n_dim);
            RAFT_CUDA_TRY(cudaMemcpy(d_means.data_handle(), means.data(),
                                    sizeof(T) * means.size(), 
                                    cudaMemcpyDefault));
   
            raft::random::normal(dev_resources, rng_state, 
                                 dataset.data_handle(), dataset.size(), mean, sigma);
            RAFT_CUDA_TRY(cudaDeviceSynchronize());
   
            #if _PRINT_ORIGINAL_DATA == 1
            {
               std::vector<T> h_dataset(n_samples * n_dim, 0);
               RAFT_CUDA_TRY(cudaMemcpy(h_dataset.data(), dataset.data_handle(),
                                       sizeof(T) * h_dataset.size(),
                                       cudaMemcpyDefault));
               RAFT_CUDA_TRY(cudaDeviceSynchronize());
   
               for (auto i = 0; i < 3; i++) {
                  printf("sample: %d:\t", i);
                  for (auto j = 0; j < n_dim; j++) {
                        printf("%.3f\t", h_dataset[i * n_dim + j]);
                  }
                  printf("\n");
               }
            }
            #endif
      }
   
      // generate random queries
      {
            raft::random::RngState rng_state(seed + 19987);
            std::vector<T> means(n_dim, mean);
            auto d_means = raft::make_device_vector<T>(dev_resources, n_dim);
            RAFT_CUDA_TRY(cudaMemcpy(d_means.data_handle(), means.data(),
                                    sizeof(T) * means.size(),
                                    cudaMemcpyDefault));
   
            raft::random::normal(dev_resources, rng_state,
                                 queries.data_handle(), queries.size(), mean, sigma);
            RAFT_CUDA_TRY(cudaDeviceSynchronize());
   
            #if _PRINT_ORIGINAL_DATA == 1
            {
               std::vector<T> h_queries(n_queries * n_dim, 0);
               RAFT_CUDA_TRY(cudaMemcpy(h_queries.data(), queries.data_handle(),
                                       sizeof(T) * h_queries.size(),
                                       cudaMemcpyDefault));
               RAFT_CUDA_TRY(cudaDeviceSynchronize());
   
               for (auto i = 0; i < std::min(n_queries, 3l); i++) {
                  printf("query: %d:\t", i);
                  for (auto j = 0; j < n_dim; j++) {
                        printf("%.3f\t", h_queries[i * n_dim + j]);
                  }
                  printf("\n");
               }
            }
            #endif
      }
     } else {
         RAFT_CUDA_TRY(cudaMemcpy(dataset.data_handle(), bindata.base_set, bindata.base_set_nrows * bindata.base_set_ndims * sizeof(T), cudaMemcpyHostToDevice));
         RAFT_CUDA_TRY(cudaMemcpy(queries.data_handle(), bindata.query_set, bindata.query_set_nrows * bindata.query_set_ndims * sizeof(T), cudaMemcpyHostToDevice));
     }
     //init GroudTruth object & call brutefore search
     GT::GroudTruth<T> bf_gt(topk, n_queries);
     
     // Build brute force index
     cuvs::neighbors::brute_force::index_params bf_idx_params;
     auto bf_index = cuvs::neighbors::brute_force::build(dev_resources, bf_idx_params, raft::make_const_mdspan(dataset.view()));
     
     // Search using brute force
     cuvs::neighbors::brute_force::search_params bf_srch_params;
     cuvs::neighbors::brute_force::search(dev_resources, bf_srch_params, bf_index, 
                                         raft::make_const_mdspan(queries.view()), result_ids.view(), result_distance.view());   
     RAFT_CUDA_TRY(cudaDeviceSynchronize());
     std::vector<int64_t> bf_results_h(n_queries * topk);
     RAFT_CUDA_TRY(cudaMemcpy(bf_results_h.data(), result_ids.data_handle(), 
                                sizeof(int64_t) * n_queries * topk,
                                cudaMemcpyDefault));
     
 
     /* config index parameters */
     using namespace cuvs::neighbors;
     ivf_flat::index_params index_params;
     index_params.kmeans_trainset_fraction = 1.0;    // use 0.8 dataset for kmeans training
     index_params.n_lists = n_lists;                 // number of inverted clusters
     index_params.metric = cuvs::distance::DistanceType::L2Expanded;
 
     /* train the index from a [N, D] dataset */
     cudaMemGetInfo(&freeMem, &totalMem); 
     printMem(freeMem, totalMem, "构建索引之前: ");
     UTIL::Timer ttt(true);
     ttt.update_time();
     auto index = cuvs::neighbors::ivf_flat::build(dev_resources, index_params, 
                                     raft::make_const_mdspan(dataset.view()));
     RAFT_CUDA_TRY(cudaDeviceSynchronize());
     float build_time = ttt.elapsed_ms();
     std::cout << std::endl;
     std::cout << "[LOG_INFO] 构建索引时间 : " << (build_time / 1000.0f) << std::endl;
     std::cout << std::endl;
     cudaMemGetInfo(&freeMem, &totalMem); 
     printMem(freeMem, totalMem, "构建索引之后: ");
     const std::string ivf_flat_index_file = "ivf_flat_index.bin";
     cuvs::neighbors::ivf_flat::serialize(dev_resources, ivf_flat_index_file, index);
     /* config search parameters*/
     cuvs::neighbors::ivf_flat::search_params search_params;
     search_params.n_probes = n_probes;
     
     /* search */
     const std::size_t num_batches = (n_queries - 1) / batch_size + 1;
     UTIL::Perf perf(n_queries, batch_size);
     
     cudaMemGetInfo(&freeMem, &totalMem); 
     printMem(freeMem, totalMem, "search之前:   ");

     for (int batch_id = 0; batch_id < num_batches; ++ batch_id) {
        const std::size_t row = batch_id * batch_size;
        const std::size_t actual_batch_size = (batch_id == num_batches - 1) ? n_queries - row : batch_size;
        
        perf.timer.update_time();
        /*
        ivf_flat::search<T, int64_t>(dev_resources, search_params, index, cuvs::make_const_mdspan(queries.view()),
                                        result_ids.view(), result_distance.view());
        */

        /*
        raft::neighbors::ivf_flat::search(
                    handle_, search_params_, *index_, queries, batch_size, k, (IdxT*)neighbors, distances, mr_ptr);
        */

        /*
        ivf_flat::search<float, int64_t>(dev_resources, search_params, index, raft::make_const_mdspan(queries.view()),
                                     result_ids.view(), result_distance.view());
        */
        
        auto queries_batch = raft::make_device_matrix_view<const T, int64_t>((queries.data_handle() + row * n_dim), actual_batch_size, n_dim);
        auto neighbors_batch = raft::make_device_matrix_view<int64_t, int64_t>((result_ids.data_handle() + row * topk), actual_batch_size, topk);
        auto distances_batch = raft::make_device_matrix_view<float, int64_t>((result_distance.data_handle() + row * topk), actual_batch_size, topk);
        
        cuvs::neighbors::ivf_flat::search(dev_resources, search_params, index, queries_batch, neighbors_batch, distances_batch);
        /*
        auto queries_v = raft::make_device_matrix_view<const float, int64_t>((queries.data_handle() + row * n_dim), actual_batch_size, n_dim);
        auto neighbors_v = raft::make_device_matrix_view<int64_t, int64_t>((result_ids.data_handle() + row * topk), actual_batch_size, topk);
        auto distances_v = raft::make_device_matrix_view<float, int64_t>((result_distance.data_handle() + row * topk), actual_batch_size, topk);
        */
        //ivf_flat::search<float, int64_t>(dev_resources, search_params, index, queries_v, neighbors_v, distances_v);

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        if (actual_batch_size != batch_size) continue; //don't record time, if the last actual_batch_size != batch_size
        perf.setTime(perf.timer.elapsed_ms());
     }
     auto ret = perf.getPerfResult();
     cudaMemGetInfo(&freeMem, &totalMem); 
     printMem(freeMem, totalMem, "search之后:  ");
     

     
     /* D->H */
     auto stream = raft::resource::get_cuda_stream(dev_resources);
     RAFT_CUDA_TRY(cudaMemcpyAsync(h_result_ids, result_ids.data_handle(), 
                             sizeof(int64_t) * n_queries * topk,
                             cudaMemcpyDefault, stream));
     RAFT_CUDA_TRY(cudaMemcpyAsync(h_result_distance, result_distance.data_handle(),
                             sizeof(float) * n_queries * topk,
                             cudaMemcpyDefault, stream));
     raft::resource::sync_stream(dev_resources);
    

     //compute_recall_ratio
     float recall_ratio_ret = bf_gt.compute_recall_ratio(bf_results_h.data(), h_result_ids);
     std::cout << std::endl;
     std::cout << std::endl;
     std::cout << "[LOG_INFO] qps is : " << std::get<0>(ret) << ", all_run_search_time is :  " 
        << std::get<1>(ret) << ", avg_search_time is :  " << std::get<2>(ret) << ", best_search_time_p99 is :  " << std::get<3>(ret) << ", recall_ratio_ret is : " << recall_ratio_ret << std::endl;
     
    
     /* release memory */
     RAFT_CUDA_TRY(cudaFreeHost(h_result_ids));
     RAFT_CUDA_TRY(cudaFreeHost(h_result_distance));
}

 
 int main(int argc, char* argv[]) {

     UTIL::CommandLineParser parser(argc, argv);
     
     /* default hyper parameters */
     int64_t n_samples = 7500000;      // number of samples of whole dataset
     int64_t n_dim = 768;            // feature dimension
     int64_t topk = 10;      
     int64_t n_queries = 10;         // number of queries
     int64_t n_lists = 10000;          // number of inverted lists (clusters)
     int64_t n_probes = 2000;  
     int batch_size = 10;
     int if_read_exited_data = 0;
     //int run_count = 1;
     std::string data_type = "";
     int split_bast_set_rows = 0;
     //bool dump_to_file = true;       // whether to save result to disk

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
     /*
     if (parser.HasOption("--run_count")) {
        run_count = std::stoi(parser.GetOptionValue("--run_count"));
     }
     */
     if (parser.HasOption("--n_lists")) {
        n_lists = std::stoi(parser.GetOptionValue("--n_lists"));
     }
     if (parser.HasOption("--n_probes")) {
        n_probes = std::stoi(parser.GetOptionValue("--n_probes"));
     }
     if (parser.HasOption("--data_type")) {
        data_type = parser.GetOptionValue("--data_type");
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


     std::cout << "[LOG_INFO] num is : " << n_samples << ", dim is : " << n_dim << ", topk is : " << topk << ", queries is : " << n_queries << ", n_lists is : " << n_lists << ", n_probes is : " << n_probes << ", batch_size is : " << batch_size << std::endl;
     //TODO: it seems that KNN only support float, other types will get compiling error
     dispatcher_raft_ivf<float>(n_samples, n_dim, topk, n_queries, n_lists, n_probes, batch_size, if_read_exited_data, split_bast_set_rows); 
 
     return 0;
 }