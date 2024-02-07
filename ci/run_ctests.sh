#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

if [ -d "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libwholegraph/" ]; then
    # Support customizing the ctests' install location
    cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libwholegraph/"
    find . -type f -executable -print0 | xargs -0 -r -t -n1 -P1 sh -c 'exec "$0"';
fi
