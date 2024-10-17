#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -e          # abort the script on error
set -o pipefail # piped commands propagate their error
set -E          # ERR traps are inherited by subcommands

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# determine PyTorch source
PKG_CUDA_VER="$(echo ${CUDA_VERSION} | cut -d '.' -f1,2 | tr -d '.')"
PKG_CUDA_VER_MAJOR=${PKG_CUDA_VER:0:2}
if [[ "${PKG_CUDA_VER_MAJOR}" == "12" ]]; then
  INDEX_URL="https://download.pytorch.org/whl/cu121"
else
  INDEX_URL="https://download.pytorch.org/whl/cu${PKG_CUDA_VER}"
fi

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
  -v \
  --extra-index-url "${INDEX_URL}" \
  "$(echo ./dist/pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]" \
  'torch>=2.0,<2.4.0a0'

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-logger "pytest pylibwholegraph"
cd python/pylibwholegraph/pylibwholegraph/tests
python -m pytest \
  --cache-clear \
  --forked \
  --import-mode=append \
  .
