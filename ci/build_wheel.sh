#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="pylibwholegraph"
package_dir="python/pylibwholegraph"

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

cd "${package_dir}"

sccache --zero-stats

SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DBUILD_SHARED_LIBS=OFF;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE;-DCUDA_STATIC_RUNTIME=ON;-DWHOLEGRAPH_BUILD_WHEELS=ON" \
  python -m pip wheel . -w dist -v --no-deps --disable-pip-version-check

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair \
  --exclude libcuda.so.1 \
  --exclude libnvidia-ml.so.1 \
  -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python final_dist
