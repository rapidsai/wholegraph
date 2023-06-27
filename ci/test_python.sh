#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -e          # abort the script on error
set -o pipefail # piped commands propagate their error
set -E          # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR
EXITCODE=0
. /opt/conda/etc/profile.d/conda.sh

ARCH=$(arch)

if [ "${ARCH}" = "aarch64" ]; then
  rapids-logger "Exiting aarch64 due to no pytorch-cuda"
  exit ${EXITCODE}
fi

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=${ARCH};py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

PACKAGES="pylibwholegraph"

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  "${PACKAGES}"

rapids-logger "Check GPU usage"
nvidia-smi

set +e

rapids-logger "pytest pylibwholegraph"
PYLIBWHOLEGRAPH_INSTALL_PATH=`python -c 'import os; import pylibwholegraph; print(os.path.dirname(pylibwholegraph.__file__))'`
PYTEST_PATH=${PYLIBWHOLEGRAPH_INSTALL_PATH}/tests
pytest \
  --cache-clear \
  --forked \
  ${PYTEST_PATH}

echo "test_python is exiting with value: ${EXITCODE}"
exit ${EXITCODE}
