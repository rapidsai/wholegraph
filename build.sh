#!/bin/bash

# Copyright (c) 2019-2023, NVIDIA CORPORATION.

# wholegraph build script

# This script is used to build component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="
    clean
    uninstall
    libwholegraph
    pylibwholegraph
    tests
    benchmarks
    docs
    -v
    -g
    -n
    --allgpuarch
    --native
    --cmake-args
    --compile-cmd
   --clean
    -h
    --help
"

HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean                    - remove all existing build artifacts and configuration (start over).
   uninstall                - uninstall libwholegraph and pylibwholegraph from a prior build/install (see also -n)
   libwholegraph            - build the libwholegraph C++ library.
   pylibwholegraph          - build the pylibwholegraph Python package.
   tests                    - build the C++ (OPG) tests.
   benchmarks               - build benchmarks.
   docs                     - build the docs
 and <flag> is:
   -v                          - verbose build mode
   -g                          - build for debug
   -n                          - no install step
   --allgpuarch               - build for all supported GPU architectures
   --cmake-args=\\\"<args>\\\" - add arbitrary CMake arguments to any cmake call
   --compile-cmd               - only output compile commands (invoke CMake without build)
   --clean                    - clean an individual target (note: to do a complete rebuild, use the clean target described above)
   -h | --h[elp]               - print this text

 default action (no args) is to build and install 'libwholegraph' and then 'pylibwholegraph'

 libwholegraph build dir is: ${LIBWHOLEGRAPH_BUILD_DIR}

 Set env var LIBWHOLEGRAPH_BUILD_DIR to override libwholegraph build dir.
"
LIBWHOLEGRAPH_BUILD_DIR=${LIBWHOLEGRAPH_BUILD_DIR:=${REPODIR}/cpp/build}

PYLIBWHOLEGRAPH_BUILD_DIRS="${REPODIR}/python/pylibwholegraph/build"
PYLIBWHOLEGRAPH_BUILD_DIRS+=" ${REPODIR}/python/pylibwholegraph/_skbuild"
PYLIBWHOLEGRAPH_BUILD_DIRS+=" ${REPODIR}/python/pylibwholegraph/pylibwholegraph/binding/include"
PYLIBWHOLEGRAPH_BUILD_DIRS+=" ${REPODIR}/python/pylibwholegraph/pylibwholegraph/binding/lib"

# All python build dirs using _skbuild are handled by cleanPythonDir, but
# adding them here for completeness
BUILD_DIRS="${LIBWHOLEGRAPH_BUILD_DIR}
            ${PYLIBWHOLEGRAPH_BUILD_DIRS}
"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
CMAKE_VERBOSE_OPTION=""
BUILD_TYPE=Release
BUILD_ALL_GPU_ARCH=0
INSTALL_TARGET="--target install"
PYTHON=${PYTHON:-python}

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:="`nproc`"}
BUILD_ABI=${BUILD_ABI:=ON}

export CMAKE_GENERATOR="${CMAKE_GENERATOR:=Ninja}"

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function buildAll {
    (( ${NUMARGS} == 0 )) || !(echo " ${ARGS} " | grep -q " [^-][a-zA-Z0-9\_\-]\+ ")
}

function cleanPythonDir {
    pushd $1 > /dev/null
    rm -rf dist *.egg-info
    find . -type d -name __pycache__ -print | xargs rm -rf
    find . -type d -name _skbuild -print | xargs rm -rf
    find . -type d -name _external_repositories -print | xargs rm -rf
    popd > /dev/null
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
    if ! (echo "${VALIDARGS}" | grep -q "^[[:blank:]]*${a}$"); then
        echo "Invalid option: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG=-v
    CMAKE_VERBOSE_OPTION="--log-level=VERBOSE"
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi

if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi

# If clean or uninstall targets given, run them prior to any other steps
if hasArg uninstall; then
    if [[ "$INSTALL_PREFIX" != "" ]]; then
        rm -rf ${INSTALL_PREFIX}/include/wholememory
        rm -f ${INSTALL_PREFIX}/lib/libwholegraph.so
        rm -rf ${INSTALL_PREFIX}/lib/cmake/wholegraph
    fi
    # This may be redundant given the above, but can also be used in case
    # there are other installed files outside of the locations above.
    if [ -e ${LIBWHOLEGRAPH_BUILD_DIR}/install_manifest.txt ]; then
        xargs rm -f < ${LIBWHOLEGRAPH_BUILD_DIR}/install_manifest.txt > /dev/null 2>&1
    fi
    # uninstall libwholegraph and pylibwholegraph installed from a prior "setup.py install"
    # FIXME: if multiple versions of these packages are installed, this only
    # removes the latest one and leaves the others installed. build.sh uninstall
    # can be run multiple times to remove all of them, but that is not obvious.
    pip uninstall -y libwholegraph pylibwholegraph
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    echo "- Cleaning"
    # Ignore errors for clean since missing files, etc. are not failures
    set +e
    # remove artifacts generated inplace
    # FIXME: ideally the "setup.py clean" command would be used for this, but
    # currently running any setup.py command has side effects (eg. cloning repos).
    # (cd ${REPODIR}/python && python setup.py clean)
    if [[ -d ${REPODIR}/python/pylibwholegraph ]]; then
        cleanPythonDir ${REPODIR}/python/pylibwholegraph
    fi

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
    find ${REPODIR}/python/pylibwholegraph -name "*.cpython*.so" -type f -delete
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
if buildAll || hasArg libwholegraph; then

    # set values based on flags
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="${WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES:=NATIVE}"
        echo "Building for the architecture of the GPU in the system..."
    else
        WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="70-real;75-real;80-real;86-real;90"
        echo "Building for *ALL* supported GPU architectures..."
    fi

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
if buildAll || hasArg pylibwholegraph; then
    if hasArg --clean; then
        cleanPythonDir ${REPODIR}/python/pylibwholegraph
    fi

    # setup.py and cmake reference an env var LIBWHOLEGRAPH_DIR to find the
    # libwholegraph package (cmake).
    # If not set by the user, set it to LIBWHOLEGRAPH_BUILD_DIR
    LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR:=${LIBWHOLEGRAPH_BUILD_DIR}}
    if ! hasArg --compile-cmd; then
        cd ${REPODIR}/python/pylibwholegraph
        env LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR} \
        ${PYTHON} setup.py build_ext --inplace \
            --build-type=${BUILD_TYPE} \
            ${EXTRA_CMAKE_ARGS}
        if ! hasArg -n; then
            env LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR} \
            ${PYTHON} setup.py install \
                --build-type=${BUILD_TYPE} \
                ${EXTRA_CMAKE_ARGS}
        fi
    else
        # just invoke cmake without going through scikit-build
        env LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR} \
        cmake -S ${REPODIR}/python/pylibwholegraph -B ${REPODIR}/python/pylibwholegraph/_skbuild/build \
           -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
            ${EXTRA_CMAKE_ARGS}
    fi
fi

################################################################################
# Build the docs
if hasArg docs; then
    if [ ! -d ${LIBWHOLEGRAPH_BUILD_DIR} ]; then
        mkdir -p ${LIBWHOLEGRAPH_BUILD_DIR}
        cd ${LIBWHOLEGRAPH_BUILD_DIR}
        cmake -B "${LIBWHOLEGRAPH_BUILD_DIR}" -S "${REPODIR}/cpp" \
              -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
              -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
              ${CMAKE_GENERATOR_OPTION} \
              ${CMAKE_VERBOSE_OPTION}
    fi
    cd ${LIBWHOLEGRAPH_BUILD_DIR}
    cmake --build "${LIBWHOLEGRAPH_BUILD_DIR}" -j${PARALLEL_LEVEL} --target docs_wholegraph ${VERBOSE_FLAG}
    cd ${REPODIR}/docs/wholegraph
    make html
fi
