cmake_minimum_required(VERSION 3.14)

include(./fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

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

# Add default project CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -Wall -O3 -g -D_FILE_OFFSET_BITS=64")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/wholegraph)

# CUDA
find_package(CUDA REQUIRED 11.0)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_70,code=compute_70")
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${CUDA_LIBRARIES})
message(STATUS "CUDA_LIBRARIES ${CUDA_LIBRARIES}")
list(APPEND LINKER_LIBS cuda)

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
        if (${Torch_CXX11})
            message(STATUS "Setting Torch CXX ABI = 1")
            add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
        else()
            message(STATUS "Setting Torch CXX ABI = 0")
            add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        endif()
     endif()

endif (WHOLEGRAPH_NEED_PYTHON)

list(APPEND WHOLEGRAPH_SOURCES ${PROJECT_SOURCE_DIR}/wholegraph/whole_memory.cc
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
        ${PROJECT_SOURCE_DIR}/wholegraph/bootstrap_communicator.cc)

list(APPEND WHOLEGRAPH_LINK_LIBRARIES cuda)

set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA NCCL headers")
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder contains NVIDIA NCCL libraries")
list(APPEND NCCL_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})
find_path(NCCL_INCLUDE_DIRS NAMES nccl.h HINTS ${NCCL_INCLUDE_DIR})
find_library(NCCL_LIBRARIES NAMES nccl HINTS ${NCCL_LIB_DIR})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if (NCCL_FOUND)
    message(STATUS "NCCL_INCLUDE_DIRS: ${NCCL_INCLUDE_DIRS}")
    message(STATUS "NCCL_LIBRARIES: ${NCCL_LIBRARIES}")
    include_directories(${NCCL_INCLUDE_DIRS})
else()
    message(FATAL_ERROR, "NCCL not found.")
    return()
endif()

list(APPEND WHOLEGRAPH_LINK_LIBRARIES ${NCCL_LIBRARIES})

add_library(whole_graph STATIC
        ${WHOLEGRAPH_SOURCES})
set_property(TARGET whole_graph PROPERTY CUDA_ARCHITECTURES 70-real 80-real 86-real 70-virtual)

set_property(TARGET whole_graph PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(whole_graph ${LINKER_LIBS} ${WHOLEGRAPH_LINK_LIBRARIES})

if (WHOLEGRAPH_BUILD_PYTORCH)
    message(STATUS "Building PyTorch support library")
    add_subdirectory(wholegraph/torch)
endif ()

if (WHOLEGRAPH_ENABLE_TESTS)
    message(STATUS "Test enabled")
    add_subdirectory(test)
endif ()
