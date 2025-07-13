#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.

# raft empty project template build script

# Abort script on first error
set -e

# Fix Git ownership issues for CMake FetchContent
git config --global --add safe.directory '*' 2>/dev/null || true
git config --global advice.detachedHead false 2>/dev/null || true

# Ensure Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Trying to use python..."
    if command -v python &> /dev/null; then
        # Create a symlink or alias for python3
        export PYTHON3_EXECUTABLE=$(which python)
    else
        echo "Error: Neither python3 nor python found. Please install Python."
        exit 1
    fi
else
    export PYTHON3_EXECUTABLE=$(which python3)
fi

PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

BUILD_TYPE=Release
BUILD_DIR=build/

CUVS_REPO_REL=""
EXTRA_CMAKE_ARGS=""
set -e


if [[ ${CUVS_REPO_REL} != "" ]]; then
  CUVS_REPO_PATH="`readlink -f \"${CUVS_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_cuvs_SOURCE=${CUVS_REPO_PATH}"
fi

if [ "$1" == "clean" ]; then
  rm -rf build
  exit 0
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DCUVS_NVTX=OFF \
 -DCMAKE_CUDA_ARCHITECTURES="NATIVE" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DFETCHCONTENT_QUIET=OFF \
 -DGIT_EXECUTABLE=$(which git) \
 -DPython3_EXECUTABLE=${PYTHON3_EXECUTABLE} \
 ${EXTRA_CMAKE_ARGS} \
 ../

cmake  --build . -j4
