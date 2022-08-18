# Copyright (c) 2022, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.20.1)

include(./fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 75-real 80-real 86)
endif ()
rapids_cuda_init_architectures(WHOLEGRAPH)
project(wholegraph CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

# PyTorch Option
set(_WHOLEGRAPH_BUILD_PYTORCH ON)
if (NOT DEFINED WHOLEGRAPH_BUILD_PYTORCH)
    set(WHOLEGRAPH_BUILD_PYTORCH ${_WHOLEGRAPH_BUILD_PYTORCH} CACHE BOOL "Build PyTorch support")
endif ()
set(WHOLEGRAPH_NEED_PYTHON ${WHOLEGRAPH_BUILD_PYTORCH})

# Test Option
set(_WHOLEGRAPH_ENABLE_TESTS ON)
if (NOT DEFINED WHOLEGRAPH_ENABLE_TESTS)
    set(WHOLEGRAPH_ENABLE_TESTS ${_WHOLEGRAPH_ENABLE_TESTS} CACHE BOOL "Enable Tests")
    message(STATUS "Setting WHOLEGRAPH_ENABLE_TESTS to  ${_WHOLEGRAPH_ENABLE_TESTS}")
endif ()

rapids_cmake_build_type(RelWithDebInfo)

# enable assert in RelWithDebInfo build type
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
message(STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

# Thread library
find_package(Threads)

# CUDA
find_package(CUDAToolkit REQUIRED 11.0)

# NCCL
include(${PROJECT_SOURCE_DIR}/cmake/thirdparty/get_nccl.cmake)

set(USE_CXX11_ABI TRUE)

if (WHOLEGRAPH_NEED_PYTHON)
    # PYTHON
    find_package(Python 3.6 COMPONENTS Interpreter REQUIRED)
    set(PY_EXE ${Python_EXECUTABLE})
    message(STATUS "Using python ${Python_EXECUTABLE}")

    find_package(PythonLibs 3.6 REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    message(STATUS "Using python include dirs ${PYTHON_INCLUDE_DIRS}")

    # pybind11
    find_package(pybind11 CONFIG REQUIRED)

    # Get Python suffix
    execute_process(COMMAND ${PY_EXE} -c "import sysconfig; print(next(x for x in [sysconfig.get_config_var('EXT_SUFFIX'), sysconfig.get_config_var('SO'), '.so'] if x))"
            OUTPUT_VARIABLE Python_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

    if (WHOLEGRAPH_BUILD_PYTORCH)
        # Find PyTorch
        # Get PyTorch cmake path
        execute_process(COMMAND ${PY_EXE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        set(Torch_ROOT "${TORCH_CMAKE_PREFIX}/Torch")
        set(TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0;8.6")
        find_package(Torch "1.9.0" "REQUIRED")
        execute_process(COMMAND ${PY_EXE} -c "from torch.utils.cpp_extension import CUDAExtension as ext; e = ext('', []); print(';'.join(e.library_dirs))"
                OUTPUT_VARIABLE Torch_LIBRARY_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "No CUDA runtime[^\n]*\n?" "" Torch_LIBRARY_DIRS "${Torch_LIBRARY_DIRS}")
        execute_process(COMMAND ${PY_EXE} -c "from torch.utils.cpp_extension import CUDAExtension as ext; e = ext('', []); print(';'.join(e.libraries))"
                OUTPUT_VARIABLE _Torch_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "No CUDA runtime[^\n]*\n?" "" _Torch_LIBRARIES "${_Torch_LIBRARIES}")
        foreach (_TLIB IN LISTS _Torch_LIBRARIES)
            find_library(FOUND_LIB_${_TLIB}
                    NAMES ${_TLIB}
                    HINTS ${Torch_LIBRARY_DIRS})
            list(APPEND TORCH_LIBRARIES ${FOUND_LIB_${_TLIB}})
        endforeach ()
        if (NOT TORCH_FOUND)
            message(FATAL_ERROR "Torch not found.")
            return()
        endif ()
        execute_process(COMMAND ${PY_EXE} -c "import torch; print(torch.torch.compiled_with_cxx11_abi())"
                OUTPUT_VARIABLE Torch_CXX11 OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(TOUPPER ${Torch_CXX11} Torch_CXX11)
        message(STATUS "Torch_CXX11: ${Torch_CXX11}")
        set(USE_CXX11_ABI ${Torch_CXX11})
     endif()

endif ()

add_library(whole_graph STATIC "")
target_sources(whole_graph PRIVATE
        ${PROJECT_SOURCE_DIR}/wholegraph/whole_memory.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/cuda_env_fns.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/file_utils.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/parallel_utils.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/whole_memory_embedding.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/whole_memory_memcpy.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/whole_memory_graph.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/graph_builder.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/whole_graph_sampler.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/whole_graph_weighted_sampler.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/whole_graph_subgraph_extractor.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/whole_graph_negative_sampler.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/gnn_ops.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/bootstrap_communicator.cc
        )

target_include_directories(whole_graph PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/wholegraph)
set_property(TARGET whole_graph PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(whole_graph PUBLIC -D_FILE_OFFSET_BITS=64)
if (${USE_CXX11_ABI})
    message(STATUS "Using CXX ABI = 1")
    target_compile_definitions(whole_graph PUBLIC -D_GLIBCXX_USE_CXX11_ABI=1)
else()
    message(STATUS "Using CXX ABI = 0")
    target_compile_definitions(whole_graph PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
endif()

target_link_libraries(whole_graph PRIVATE CUDA::cuda_driver NCCL::NCCL)
target_link_libraries(whole_graph PUBLIC CUDA::cudart)

if (WHOLEGRAPH_BUILD_PYTORCH)
    message(STATUS "Building PyTorch support library")
    add_subdirectory(wholegraph/torch)
endif ()

if (WHOLEGRAPH_ENABLE_TESTS)
    message(STATUS "Test enabled")
    add_subdirectory(test)
endif ()
