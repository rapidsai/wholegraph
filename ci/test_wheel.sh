#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -e          # abort the script on error
set -o pipefail # piped commands propagate their error
set -E          # ERR traps are inherited by subcommands

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/pylibwholegraph*.whl)

PKG_CUDA_VER="$(echo ${CUDA_VERSION} | cut -d '.' -f1,2 | tr -d '.')"
PKG_CUDA_VER_MAJOR=${PKG_CUDA_VER:0:2}
if [[ "${PKG_CUDA_VER_MAJOR}" == "12" ]]; then
  INDEX_URL="https://download.pytorch.org/whl/nightly/cu121"
else
  INDEX_URL="https://download.pytorch.org/whl/cu${PKG_CUDA_VER}"
fi
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-logger "Installing PyTorch"
rapids-retry python -m pip install --pre torch --index-url ${INDEX_URL}
rapids-retry python -m pip install pytest pytest-forked numpy
rapids-logger "pytest pylibwholegraph"
PYLIBWHOLEGRAPH_INSTALL_PATH=`python -c 'import os; import pylibwholegraph; print(os.path.dirname(pylibwholegraph.__file__))'`
PYTEST_PATH=${PYLIBWHOLEGRAPH_INSTALL_PATH}/tests
python -m pytest \
  --cache-clear \
  --forked \
  ${PYTEST_PATH}
