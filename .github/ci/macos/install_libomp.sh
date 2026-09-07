#!/bin/bash
# Stage the conda-forge libomp at /opt/llvm-openmp so the wheels link an
# OpenMP runtime that supports macOS versions older than the build machine
# (the Homebrew build does not). The .conda payload is a zstd-compressed
# tar; libiomp5.dylib is dropped so the loader cannot pick the duplicate
# runtime, and the remaining libomp.dylib is re-signed with an ad-hoc
# identity after its install name is rewritten.
retry () {
    $* || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

set -ex

OMP_PREFIX=/opt/llvm-openmp
sudo mkdir -p ${OMP_PREFIX}
sudo chown -R $USER: ${OMP_PREFIX}
# zstd is needed to extract the payload
retry brew install zstd
pushd ${OMP_PREFIX}
  llvm_openmp_version="21.1.8-h4a912ad_0"
  retry curl -OLs https://conda.anaconda.org/conda-forge/osx-arm64/llvm-openmp-${llvm_openmp_version}.conda
  tar -xvf llvm-openmp-${llvm_openmp_version}.conda
  rm llvm-openmp-${llvm_openmp_version}.conda
  tar -xvf pkg-llvm-openmp-${llvm_openmp_version}.tar.zst
  rm pkg-llvm-openmp-${llvm_openmp_version}.tar.zst
  rm info-llvm-openmp-${llvm_openmp_version}.tar.zst
  rm lib/libiomp5.dylib
  install_name_tool -id ${OMP_PREFIX}/lib/libomp.dylib lib/libomp.dylib
  codesign -f -s - lib/libomp.dylib
popd
