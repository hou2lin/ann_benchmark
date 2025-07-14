# Example RAFT Project Template

This project is copied from `raft/cpp/template/`.

Verified under docker image `nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu22.04`.

## prerequisites ##
```text
cmake >= 3.31.8
```

```shell
wget https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.sh
./cmake-3.31.8-linux-x86_64.sh --prefix=/usr/local --exclude-subdir
cmake --version 
```

## compile ##
It will automatically download `CuVS`, `RAFT` and all its dependencies. Then compile the files in [src](./src/).
```text
src/
├── bindata.h
├── groudtruth.h
├── test_cagra.cu
├── test_cagra_cuvs.cu
├── test_ivf_flat.cu
├── test_ivf_flat_cuvs.cu
├── test_ivf_pq.cu
├── test_ivf_pq_cuvs.cu
├── test_vamana_cuvs.cu
└── utils.h
```
For each type of ANN Indexing, can set dataset `n_sample`, `n_dim`, and index configs.

```shell
$ cd ann-benchmark/
$ ./build.sh  
```

## run ##
```shell
$ ./build/test_ivf_pq_cuvs   #IVF_PQ benchmark
$ ./build/test_ivf_flat_cuvs   #IVF benchmark
$ ./build/test_cagra_cuvs    #CAGRA benchmark
$ ./build/test_vamana_cuvs    #Vamana benchmark
```

