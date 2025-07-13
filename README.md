# Example RAFT Project Template

This project is copied from `raft/cpp/template/`.

Verified under docker image `nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04`.

## prerequisites ##
```text
1. cmake >= 3.26.4
2. ninja
```

## compile ##
It will automatically download `RAFT` and all its dependencies. Then compile the files in [src](./src/).

```shell
$ cd raft_demo/
# option 1
$ ./build.sh  
# option 2
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DRATF_NVTX=OFF -DCMAKE_CUDA_ARCHITECTURES="NATIVE" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
$ make -j8
```

## run ##
```shell
$ ./test_ivf_pq     # will get ivf_pq_result.txt
$ ./test_ivf_flat   # will get ivf_flat_result.txt
```