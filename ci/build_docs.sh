#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n docs
# Temporarily allow unbound variables for conda activation.
set +u
conda activate docs
set -u

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)
export RAPIDS_VERSION_NUMBER="23.12"
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  libwholegraph pylibwholegraph

rapids-logger "Build Doxygen docs"
pushd cpp
doxygen Doxyfile
mkdir -p ../docs/wholegraph/_html/doxygen_docs/libwholegraph/html
mv html/* ../docs/wholegraph/_html/doxygen_docs/libwholegraph/html
mkdir -p ../docs/wholegraph/_xml
# _xml is used for sphinx breathe project
mv xml/* "../docs/wholegraph/_xml"
popd

rapids-logger "Build Sphinx docs"
pushd docs/wholegraph
sphinx-build -b dirhtml ./source _html
sphinx-build -b text ./source _text
mkdir -p "${RAPIDS_DOCS_DIR}/wholegraph/"{html,txt}
mv _html/* "${RAPIDS_DOCS_DIR}/wholegraph/html"
mv _text/* "${RAPIDS_DOCS_DIR}/wholegraph/txt"
popd

rapids-logger "Output temp dir: ${RAPIDS_DOCS_DIR}"

rapids-upload-docs
