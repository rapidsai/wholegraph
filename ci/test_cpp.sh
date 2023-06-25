#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -e          # abort the script on error
set -o pipefail # piped commands propagate their error
set -E          # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR
EXITCODE=0
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

PACKAGES="libwholegraph libwholegraph-tests"

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  "${PACKAGES}"

rapids-logger "Check GPU usage"
nvidia-smi

set +e

# Run libwholegraph tests from libwholegraph-tests package
rapids-logger "Run tests"
INSTALLED_TEST_PATH=${CONDA_PREFIX}/bin/gtests/libwholegraph

for file in "${INSTALLED_TEST_PATH}"/*; do
  if [[ -x "$file" ]]; then
    rapids-logger "Running: $file"
    "$file"
    exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
      echo "Test $file returned a non-zero exit code: $exit_code"
      exit $exit_code
    fi
  fi
done

exit 0
