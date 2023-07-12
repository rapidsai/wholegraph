#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

# setup.py updates
sed -i "s/^version = .*/version = \"${VERSION}\"/g" \
  python/pylibwholegraph/pyproject.toml

# pyproject.toml cuda suffixes
sed -i "s/name = \"pylibwholegraph\"/name = \"pylibwholegraph${CUDA_SUFFIX}\"/g" python/pylibwholegraph/pyproject.toml

