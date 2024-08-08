#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create clang-tidy conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate clang-tidy testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n clang_tidy
# Temporarily allow unbound variables for conda activation.
set +u
conda activate clang_tidy
set -u

rapids-print-env

env PATH=${PATH}:/usr/local/cuda/bin

# we do this in two separate parts, one for the library and tests
# and a second one for the bindings, where we can add torch bindings regardless
# of the ABI for library / tests. We'll ignore the generated database for the
# library in the second run.
CMAKE_EXTRA_ARGS="--cmake-args=\"-DBUILD_OPS_WITH_TORCH_C10_API=OFF\""
rapids-logger "Generate compilation databases for C++ library and tests"
./build.sh clean libwholegraph tests pylibwholegraph --allgpuarch --compile-cmd ${CMAKE_EXTRA_ARGS}

#  -git_modified_only -v
rapids-logger "Run clang-tidy"
python scripts/checks/run-clang-tidy.py \
  -ignore wholememory_binding \
  cpp/build/compile_commands.json \
  python/pylibwholegraph/_skbuild/build/compile_commands.json \
  -v
