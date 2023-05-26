#!/bin/bash

# Copyright (c) 2019-2023, NVIDIA CORPORATION.

# wholegraph build script for single components

# This script is used to build single components in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean libwholegraph tests "
VALIDARGS+="pylibwholegraph -v -g -n "
VALIDARGS+="--native --cmake-args --compile-cmd -h --help"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean                    - remove all existing build artifacts and configuration (start over).
   libwholegraph            - build the libwholegraph C++ library.
   tests                    - build the C++ (OPG) tests.
   benchmarks               - build benchmarks.
   pylibwholegraph          - build the pylibwholegraph Python package.
 and <flag> is:
   -v                          - verbose build mode
   -g                          - build for debug
   -n                          - no install step
   --native                    - build for the GPU architecture of the current system
   --cmake-args=\\\"<args>\\\" - add arbitrary CMake arguments to any cmake call
   --compile-cmd               - only output compile commands (invoke CMake without build)
   -h | --h[elp]               - print this text
"
LIBWHOLEGRAPH_BUILD_DIR=${REPODIR}/build
PYLIBWHOLEGRAPH_BUILD_DIRS="${REPODIR}/pylibwholegraph/build"
PYLIBWHOLEGRAPH_BUILD_DIRS+=" ${REPODIR}/pylibwholegraph/_skbuild"
PYLIBWHOLEGRAPH_BUILD_DIRS+=" ${REPODIR}/pylibwholegraph/pylibwholegraph/binding/include"
PYLIBWHOLEGRAPH_BUILD_DIRS+=" ${REPODIR}/pylibwholegraph/pylibwholegraph/binding/lib"
BUILD_DIRS="${LIBWHOLEGRAPH_BUILD_DIR} ${PYLIBWHOLEGRAPH_BUILD_DIRS}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
BUILD_ALL_GPU_ARCH=1
INSTALL_TARGET="--target install"

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}

export CMAKE_GENERATOR="${CMAKE_GENERATOR:=Ninja}"

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo $ARGS | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo $ARGS | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo $ARGS | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo $EXTRA_CMAKE_ARGS | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
}

if hasArg -h || hasArg --h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    # Check for cmake args
    cmakeArgs
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG=-v
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --native; then
    BUILD_ALL_GPU_ARCH=0
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
    if [ -d ${bd} ]; then
        find ${bd} -mindepth 1 -delete
        rmdir ${bd} || true
    fi
    done
    # remove any left-over cpython shared libraries
    find ${REPODIR}/pylibwholegraph -name "*.cpython*.so" -type f -delete
fi

# set values based on flags
if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
    WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="${WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES:=NATIVE}"
    echo "Building for the architecture of the GPU in the system..."
else
    WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="70-real;75-real;80-real;86-real;90"
    echo "Building for *ALL* supported GPU architectures..."
fi
if hasArg tests; then
    BUILD_TESTS=ON
else
    BUILD_TESTS=OFF
fi
if hasArg benchmarks; then
    BUILD_BENCHMARKS=ON
else
    BUILD_BENCHMARKS=OFF
fi

################################################################################
# libwholegraph
if hasArg libwholegraph; then
    cmake -S ${REPODIR}/cpp -B ${LIBWHOLEGRAPH_BUILD_DIR} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
          -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE \
          -DBUILD_TESTS=${BUILD_TESTS} \
          ${EXTRA_CMAKE_ARGS}

    cd ${LIBWHOLEGRAPH_BUILD_DIR}

    if ! hasArg --compile-cmd; then
        ## Build and (optionally) install library + tests
        cmake --build . -j${PARALLEL_LEVEL} ${INSTALL_TARGET} ${VERBOSE_FLAG}
    fi
fi

################################################################################
# pylibwholegraph
if hasArg pylibwholegraph; then
    # setup.py and cmake reference an env var LIBWHOLEGRAPH_DIR to find the
    # libwholegraph package (cmake).
    # If not set by the user, set it to LIBWHOLEGRAPH_BUILD_DIR
    LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR:=${LIBWHOLEGRAPH_BUILD_DIR}}
    if ! hasArg --compile-cmd; then
        cd ${REPODIR}/pylibwholegraph
        env LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR} \
        ${PYTHON} ${REPODIR}/pylibwholegraph/setup.py build_ext --inplace \
            --build-type=${BUILD_TYPE} \
            ${EXTRA_CMAKE_ARGS}
        if ! hasArg -n; then
            env LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR} \
            ${PYTHON} ${REPODIR}/pylibwholegraph/setup.py install \
                --build-type=${BUILD_TYPE} \
                ${EXTRA_CMAKE_ARGS}
        fi
    else
        # just invoke cmake without going through scikit-build
        env LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR} \
        cmake -S ${REPODIR}/pylibwholegraph -B ${REPODIR}/pylibwholegraph/_skbuild/build \
           -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
            ${EXTRA_CMAKE_ARGS}
    fi
fi
