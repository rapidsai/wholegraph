#=============================================================================
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
#
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
#=============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

set(RAPIDS_VERSION "24.12")
set(WHOLEGRAPH_VERSION "${RAPIDS_VERSION}.00")

include(FetchContent)

FetchContent_Declare(
        rapids-cmake
        GIT_REPOSITORY https://github.com/rapidsai/rapids-cmake.git
        GIT_TAG        origin/branch-${RAPIDS_VERSION}
)
FetchContent_MakeAvailable(rapids-cmake)
include(rapids-cmake)
include(rapids-cuda)
include(rapids-cpm)

rapids_cuda_init_architectures(PYLIBWHOLEGRAPH)

project(PYLIBWHOLEGRAPH VERSION ${WHOLEGRAPH_VERSION} LANGUAGES CXX CUDA)

##############################################################################
# - User Options  ------------------------------------------------------------

option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(WHOLEGRAPH_BUILD_WHEELS "Whether we're building a wheel for pypi" OFF)

##############################################################################
# - Base Rapids Options  -----------------------------------------------------

set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# pywholegraph Options
function(show_python_envs position)
    message(STATUS "[PYTHON INFO] POSITION: ${position} Python_ROOT_DIR=${Python_ROOT_DIR}")
    message(STATUS "                        ${position} Python_INCLUDE_DIR=${Python_INCLUDE_DIR} Python_LIBRARY=${Python_LIBRARY} Python_EXECUTABLE=${Python_EXECUTABLE}")
    message(STATUS "                        ${position} PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} PYTHON_LIBRARY=${PYTHON_LIBRARY} PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
    message(STATUS "                        ${position} Python_INCLUDE_DIRS=${Python_INCLUDE_DIRS} Python_LIBRARIES=${Python_LIBRARIES}")
endfunction(show_python_envs)

set(SKBUILD ON)
set(Python_USE_STATIC_LIBS TRUE)
if (NOT DEFINED BUILD_PYTHON_EXE)
    if (DEFINED Python_EXECUTABLE)
        set(BUILD_PYTHON_EXE ${Python_EXECUTABLE})
    elseif(DEFINED PYTHON_EXECUTABLE)
        set(BUILD_PYTHON_EXE ${PYTHON_EXECUTABLE})
    elseif(DEFINED Python_ROOT_DIR)
        set(BUILD_PYTHON_EXE "${Python_ROOT_DIR}/bin/python3")
    else()
        execute_process(COMMAND "which" "python3"
                OUTPUT_VARIABLE BUILD_PYTHON_EXE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
endif()
execute_process(COMMAND ${BUILD_PYTHON_EXE} -c "import sys; print(sys.exec_prefix)"
        OUTPUT_VARIABLE Python_ROOT_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND ${BUILD_PYTHON_EXE} -c "import sys; print(sys.executable)"
        OUTPUT_VARIABLE Python_EXECUTABLE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND ${BUILD_PYTHON_EXE} -c "import sysconfig; print(sysconfig.get_path('include'))"
        OUTPUT_VARIABLE Python_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND ${BUILD_PYTHON_EXE} -c "import sysconfig; print(sysconfig.get_config_var('LIBDEST'))"
        OUTPUT_VARIABLE Python_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python_INCLUDE_DIR})
set(PYTHON_LIBRARY ${Python_LIBRARY})
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)

show_python_envs("python env")

# default build type
rapids_cmake_build_type(Release)

message("CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")

if(DETECT_CONDA_ENV)
    rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
        message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
        set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
    endif()
endif()

rapids_cpm_init()

include(../../cpp/cmake/thirdparty/get_raft.cmake)

##############################################################################
# - Dependencies  ------------------------------------------------------------

#find_package(PythonInterp 3 REQUIRED)

#include(${CMAKE_CURRENT_LIST_DIR}/../cmake/thirdparty/nanobind.cmake)

# use <package>_ROOT here to take precedence over any other package
if (DEFINED ENV{LIBWHOLEGRAPH_DIR})
  set(wholegraph_ROOT "$ENV{LIBWHOLEGRAPH_DIR}")
endif()
find_package(wholegraph "${RAPIDS_VERSION}.0" EXACT)
message("WholeGraph")
if (WHOLEGRAPH_FOUND)
    message(STATUS "PYLIBWHOLEGRAPH: using pre-built wholegraph C++ package")
elseif(WHOLEGRAPH_BUILD_WHEELS)
    # statically link dependencies if building wheels
    message(STATUS "PYLIBWHOLEGRAPH: build wheels")
    add_subdirectory(../../cpp/ libwholegraph EXCLUDE_FROM_ALL)
else()
    message(FATAL_ERROR "PYLIBWHOLEGRAPH: could not find wholegraph package in "
            "cmake prefix ${CMAKE_PREFIX_PATH} or user dir $ENV{LIBWHOLEGRAPH_DIR}")
endif()

execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import os; import skbuild; print(os.path.join(os.path.dirname(skbuild.__file__), 'resources/cmake'))"
        OUTPUT_VARIABLE SKBUILD_CMAKE_MODULE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_MODULE_PATH "${SKBUILD_CMAKE_MODULE_PATH}")

include(rapids-cython-core)
rapids_cython_init()

##############################################################################
# - Display options ----------------------------------------------------------

message(VERBOSE "PYLIBWHOLEGRAPH: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")

##############################################################################
# - Compiler options ---------------------------------------------------------

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CXX_DEFINITIONS WHOLEGRAPH_VERSION=${WHOLEGRAPH_VERSION})

message(STATUS "PYLIBWHOLEGRAPH: DEFAULT_CXX_FLAGS='${DEFAULT_CXX_FLAGS}'")
message(STATUS "PYLIBWHOLEGRAPH: CXX_FLAGS='${CXX_FLAGS}'")
message(STATUS "PYLIBWHOLEGRAPH: CXX_DEFINITIONS='${CXX_DEFINITIONS}'")

##############################################################################
# - Variables ----------------------------------------------------------------

set(WHOLEGRAPH_CPP_TARGET "wholegraph::wholegraph" CACHE STRING "libwholegraph target name")

add_subdirectory(pylibwholegraph/binding)

# when used without setup.py, command is like:
# export LIBWHOLEGRAPH_DIR=`pwd`/../../cpp/build/install
# cmake ../ -DSKBUILD=ON
