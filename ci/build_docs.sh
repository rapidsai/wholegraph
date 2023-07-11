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
VERSION_NUMBER="23.08"
export RAPIDS_VERSION_NUMBER=${VERSION_NUMBER}
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
libwholegraph pylibwholegraph numpy

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
mkdir -p "${RAPIDS_DOCS_DIR}/wholegraph/"{html,txt,doxygen_docs}
mv _html/* "${RAPIDS_DOCS_DIR}/wholegraph/html"
mv _text/* "${RAPIDS_DOCS_DIR}/wholegraph/txt"
popd

rapids-logger "Output temp dir: ${RAPIDS_DOCS_DIR}"

#if [[ "${RAPIDS_BUILD_TYPE}" == "branch" ]]; then
#  rapids-logger "Upload Docs to S3"
#  aws s3 sync --no-progress --delete cpp/html "s3://rapidsai-docs/libwholegraph/${VERSION_NUMBER}/html"
#  aws s3 sync --no-progress --delete docs/wholegraph/html "s3://rapidsai-docs/wholegraph/${VERSION_NUMBER}/html"
#  aws s3 sync --no-progress --delete docs/wholegraph/text "s3://rapidsai-docs/wholegraph/${VERSION_NUMBER}/txt"
#fi
