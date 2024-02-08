#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/pylibwholegraph/pylibwholegraph/

pytest --cache-clear --forked --import-mode=append "$@" tests
