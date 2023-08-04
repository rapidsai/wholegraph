#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

source rapids-configure-sccache
source rapids-date-string

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
version_override="$(rapids-pip-wheel-version ${RAPIDS_DATE_STRING})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

ci/release/apply_wheel_modifications.sh ${version_override} "-${RAPIDS_PY_CUDA_SUFFIX}"
echo "The package name and/or version was modified in the package source. The git diff is:"
git diff

cd python/pylibwholegraph

# Hardcode the output dir
SKBUILD_CONFIGURE_OPTIONS="-DDETECT_CONDA_ENV=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE -DCUDA_STATIC_RUNTIME=ON -DWHOLEGRAPH_BUILD_WHEELS=ON" \
  python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair --exclude libcuda.so.1 -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
