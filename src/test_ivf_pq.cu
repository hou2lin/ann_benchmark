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

 #include <cuvs/neighbors/ivf_pq.cuh>
 #include <cuvs/random/rng.cuh>
 #include <cuvs/util/cuda_rt_essentials.hpp>

#define _PRINT_ORIGINAL_DATA 0

int main() {

    /* hyper parameters */
    int64_t n_samples = 50000;      // number of samples of whole dataset
    int64_t n_dim = 256;            // feature dimension
    int64_t topk = 10;      
    int64_t n_queries = 10;         // number of queries
    int64_t n_lists = 128;          // number of inverted lists (clusters)
    float mean = 0.1;
    float sigma = 1.0;
    uint64_t seed = 233376;
    bool dump_to_file = true;       // whether to save result to disk

    /* initialization */
         raft::resources dev_resources;
    dev_resources.get_cublas_handle();
    raft::resource::set_workspace_to_pool_resource(
        dev_resources, std::make_optional<std::size_t>(8 * 1024 * 1024 * 1024ull)); // Use 8 GB of pool memory

    /* create input and output arrays on GPU */
         auto dataset = cuvs::make_device_matrix<float>(dev_resources, n_samples, n_dim);        // [N, D] * float
     auto queries = cuvs::make_device_matrix<float>(dev_resources, n_queries, n_dim);        // [Nq, D] * float
     auto result_ids = cuvs::make_device_matrix<int64_t>(dev_resources, n_queries, topk);    // [Nq, topk] * int64
     auto result_distance = cuvs::make_device_matrix<float>(dev_resources, n_queries, topk); // [Nq, topk] * float
    int64_t *h_result_ids = nullptr;
    RAFT_CUDA_TRY(cudaHostAlloc(&h_result_ids, sizeof(int64_t) * n_queries * topk, cudaHostAllocDefault));
    float *h_result_distance = nullptr;
    RAFT_CUDA_TRY(cudaHostAlloc(&h_result_distance, sizeof(float) * n_queries * topk, cudaHostAllocDefault));

    // set dataset
    {
         cuvs::random::RngState rng_state(seed);
        std::vector<float> means(n_dim, mean);
         auto d_means = cuvs::make_device_vector<float>(dev_resources, n_dim);
        RAFT_CUDA_TRY(cudaMemcpy(d_means.data_handle(), means.data(),
                                 sizeof(float) * means.size(), 
                                 cudaMemcpyDefault));

                     raft::random::normal(dev_resources, rng_state,
                                 dataset.view(), mean, sigma);
        RAFT_CUDA_TRY(cudaDeviceSynchronize());

        #if _PRINT_ORIGINAL_DATA == 1
        {
            std::vector<float> h_dataset(n_samples * n_dim, 0);
            RAFT_CUDA_TRY(cudaMemcpy(h_dataset.data(), dataset.data_handle(),
                                     sizeof(float) * h_dataset.size(),
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
        cuvs::random::RngState rng_state(seed + 19987);
        std::vector<float> means(n_dim, mean);
        auto d_means = cuvs::make_device_vector<float>(dev_resources, n_dim);
        RAFT_CUDA_TRY(cudaMemcpy(d_means.data_handle(), means.data(),
                                 sizeof(float) * means.size(),
                                 cudaMemcpyDefault));

                    raft::random::normal(dev_resources, rng_state,
                                 queries.view(), mean, sigma);
        RAFT_CUDA_TRY(cudaDeviceSynchronize());

        #if _PRINT_ORIGINAL_DATA == 1
        {
            std::vector<float> h_queries(n_queries * n_dim, 0);
            RAFT_CUDA_TRY(cudaMemcpy(h_queries.data(), queries.data_handle(),
                                     sizeof(float) * h_queries.size(),
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


    /* config index parameters */
         using namespace cuvs::neighbors;
    ivf_pq::index_params index_params;
    index_params.kmeans_trainset_fraction = 0.8;    // use 0.8 dataset for kmeans training
    index_params.n_lists = n_lists;                 // number of inverted clusters
         index_params.metric = cuvs::distance::DistanceType::L2Expanded;

    /* train the index from a [N, D] dataset */
    auto index = ivf_pq::build<float, int64_t>(dev_resources, index_params, 
                                     cuvs::make_const_mdspan(dataset.view()));
    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    /* config search parameters*/
    ivf_pq::search_params search_params;
    search_params.n_probes = n_lists;

    /* search */
         ivf_pq::search<float, int64_t>(dev_resources, search_params, index, cuvs::make_const_mdspan(queries.view()),
                                     result_ids.view(), result_distance.view());

    
    /* D->H */
    auto stream = dev_resources.get_stream();
    RAFT_CUDA_TRY(cudaMemcpyAsync(h_result_ids, result_ids.data_handle(), 
                            sizeof(int64_t) * n_queries * topk,
                            cudaMemcpyDefault, stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(h_result_distance, result_distance.data_handle(),
                            sizeof(float) * n_queries * topk,
                            cudaMemcpyDefault, stream));
    dev_resources.sync_stream();

    /* check result */
    if (!dump_to_file) {
        for (auto i = 0; i < n_queries; i++) {
            printf("Query %d:\t", i);
            for (auto j = 0; j < topk; j++) {
                printf("{%ld,%.1f}\t", h_result_ids[i * topk + j], h_result_distance[i * topk + j]);
            }
            printf("\n");
        }
    } else {
        const char *filename = "ivf_pq_result.txt";
        std::FILE *pf = std::fopen(filename, "w");
        if (nullptr == pf) return -1;

        char *buff = new char[topk * 32]();
        for (auto i = 0; i < n_queries; i++) {
            snprintf(buff, topk * 32, "Query %d:\t", i);
            fprintf(pf, "%s", buff);
            for (auto j = 0; j < topk; j++) {
                snprintf(buff, topk * 32, "{%ld,%.1f}\t", 
                         h_result_ids[i * topk + j],
                         h_result_distance[i * topk + j]);
                fprintf(pf, "%s", buff);
            }
            fprintf(pf, "%s\n", "");
        }
        delete[] buff;
        std::fflush(pf);
        std::fclose(pf);

        std::cout << "Saved to " << filename << std::endl;
    }

    /* release memory */
    RAFT_CUDA_TRY(cudaFreeHost(h_result_ids));
    RAFT_CUDA_TRY(cudaFreeHost(h_result_distance));

    return 0;
}