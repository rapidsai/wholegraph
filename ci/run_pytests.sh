#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

pytest --cache-clear --forked "$@" "$(python -c 'import os; import pylibwholegraph; print(os.path.dirname(pylibwholegraph.__file__))')/tests"
