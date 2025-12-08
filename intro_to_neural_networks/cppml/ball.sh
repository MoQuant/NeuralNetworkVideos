#!/bin/bash

set -e  # stop on any error

echo "🚀 Compiling Neural Network"

OUT=nnet

# Remove old binary
rm -f $OUT system_errors.txt

clang++ neural.cpp -o $OUT \
  -std=c++17 -O3 \
  -I./libtorch/include \
  -I./libtorch/include/torch/csrc/api/include \
  -L./libtorch/lib -ltorch -ltorch_cpu -lc10 \
  -lpthread \
  -Wl,-rpath,@loader_path/libtorch/lib \
  -fmax-errors=1 \
  2>&1 | tee system_errors.txt

export DYLD_LIBRARY_PATH=$PWD/libtorch/lib:$DYLD_LIBRARY_PATH

echo "✅ Compiled Neural Network"
