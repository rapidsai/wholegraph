#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
# Temporarily allow unbound variables for conda activation.
set +u
conda activate docs
set -u

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  "libwholegraph=${RAPIDS_VERSION}"

rapids-logger "Build Doxygen docs"
pushd cpp
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libwholegraph/xml_tar"
tar -czf "${RAPIDS_DOCS_DIR}/libwholegraph/xml_tar"/xml.tar.gz -C xml .
popd

rapids-logger "Output temp dir: ${RAPIDS_DOCS_DIR}"

RAPIDS_VERSION_NUMBER="$(rapids-version-major-minor)" rapids-upload-docs
