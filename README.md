# Example RAFT Project Template

This project is copied from `raft/cpp/template/`.

Verified under docker image `nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04`.

## prerequisites ##
```text
1. cmake >= 3.26.4
2. ninja
```

## compile ##
It will automatically download `RAFT`, `CuVS` and all its dependencies. Then compile the files in [src](./src/).

```text
src/
├── bindata.h
├── groudtruth.h
├── test_cagra.cu
├── test_cagra_netease.cu
├── test_ivf_flat.cu
├── test_ivf_flat_netease.cu
├── test_ivf_pq.cu
├── test_ivf_pq_netease.cu
├── test_vamana_netease.cu
└── utils.h
```
For each type of ANN Indexing, can set dataset `n_sample`, `n_dim`, and index configs.

```shell
$ cd ann-benchmark/
$ ./build.sh  
```

## run ##
```shell
$ ./build/test_ivf_pq_netease   #IVF_PQ benchmark
$ ./build/test_ivf_flat_netease   #IVF benchmark
$ ./build/test_cagra_netease    #CAGRA benchmark
$ ./build/test_vamana_netease    #Vamana benchmark
```
