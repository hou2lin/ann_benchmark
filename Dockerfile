FROM nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git vim python3-pip make g++ libomp-dev \
        libaio-dev libgoogle-perftools-dev \
        clang-format libboost-all-dev libmkl-full-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.sh && \
    chmod +x cmake-3.31.8-linux-x86_64.sh && \
    ./cmake-3.31.8-linux-x86_64.sh --prefix=/usr/local --exclude-subdir && \
    rm cmake-3.31.8-linux-x86_64.sh

RUN pip3 install --no-cache-dir --upgrade pip
