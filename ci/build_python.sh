#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

PACKAGES="libwholegraph"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)
export RAPIDS_PACKAGE_VERSION=${version}

rapids-logger "Begin py build"

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
rapids-logger "Begin pylibwholegraph build"
version_file_pylibwholegraph="python/pylibwholegraph/pylibwholegraph/_version.py"
sed -i "/^__version__/ s/= .*/= ${version}/g" ${version_file_pylibwholegraph}
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_pylibwholegraph}
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/pylibwholegraph

rapids-upload-conda-to-s3 python
