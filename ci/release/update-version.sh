#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCX_PY_VERSION="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG})"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG and short tag => $CURRENT_SHORT_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# rapids-cmake version
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' fetch_rapids.cmake

# cpp CMakeLists update
sed_runner 's/'"RAPIDS_VERSION \".*\")"'/'"RAPIDS_VERSION \"${NEXT_SHORT_TAG}\")"'/g' cpp/CMakeLists.txt

# Python CMakeLists updates
sed_runner 's/'"RAPIDS_VERSION .*)"'/'"RAPIDS_VERSION ${NEXT_FULL_TAG})"'/g' python/pylibwholegraph/CMakeLists.txt

# Python setup.py updates
sed_runner 's/'version=".*"'/'version="${NEXT_FULL_TAG}"'/g' python/pylibwholegraph/setup.py


# RTD update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/wholegraph/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/wholegraph/source/conf.py

# Python __init__.py updates
sed_runner "s/__version__ = .*/__version__ = \"${NEXT_FULL_TAG}\"/g" python/pylibwholegraph/pylibwholegraph/__init__.py

# Python pyproject.toml updates
sed_runner "s/^version = .*/version = \"${NEXT_FULL_TAG}\"/g" python/pylibwholegraph/pyproject.toml

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")

DEPENDENCIES=(
  libraft
  libraft-headers
  librmm
  pylibcugraphops
  pylibraft
  pyraft
  raft-dask
  rmm
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" ${FILE}
  done
  for FILE in python/**/pyproject.toml; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*\"/g" ${FILE}
  done
done

# Doxyfile update
sed_runner "s|PROJECT_NUMBER[[:space:]]*=.*|PROJECT_NUMBER=${NEXT_SHORT_TAG}|" cpp/Doxyfile


# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-action-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"

done
sed_runner "s/RAPIDS_VERSION_NUMBER=\".*/RAPIDS_VERSION_NUMBER=\"${NEXT_SHORT_TAG}\"/g" ci/build_docs.sh
