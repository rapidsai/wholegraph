#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

ARCH=$(arch)
EXITCODE=0

if [ "${ARCH}" = "aarch64" ]; then
  rapids-logger "Exiting aarch64 due to no pytorch-cuda"
  exit ${EXITCODE}
fi

if [ "${RAPIDS_CUDA_VERSION:0:2}" == "12" ]; then
  rapids-logger "Exiting CUDA 12 due to no pytorch stable yet"
  exit ${EXITCODE}
fi

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=${ARCH};py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

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

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  'mkl<2024.1.0' \
  "pylibwholegraph=${RAPIDS_VERSION}"

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "pytest pylibwholegraph"
./ci/run_pytests.sh && EXITCODE=$? || EXITCODE=$?

echo "test_python is exiting with value: ${EXITCODE}"
exit ${EXITCODE}
