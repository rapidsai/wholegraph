cmake_minimum_required(VERSION 3.14)

project(wholememory CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

# PyTorch Option
set(_WHOLEMEMORY_BUILD_PYTORCH ON)
if (NOT DEFINED WHOLEMEMORY_BUILD_PYTORCH)
    set(WHOLEMEMORY_BUILD_PYTORCH ${_WHOLEMEMORY_BUILD_PYTORCH} CACHE BOOL "Build PyTorch support")
endif()

set(WHOLEMEMORY_NEED_PYTHON ${WHOLEMEMORY_BUILD_PYTORCH})

# Test Option
set(_WHOLEMEMORY_ENABLE_TESTS ON)
if (NOT DEFINED WHOLEMEMORY_ENABLE_TESTS)
    set(WHOLEMEMORY_ENABLE_TESTS ${_WHOLEMEMORY_ENABLE_TESTS} CACHE BOOL "Enable Tests")
    message(STATUS "Setting WHOLEMEMORY_ENABLE_TESTS to  ${_WHOLEMEMORY_ENABLE_TESTS}")
endif()

# Configure path to modules (for find_package)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

# Add default project CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -Wall -O3 -g -D_FILE_OFFSET_BITS=64")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/wholememory)

# CUDA
find_package(CUDA REQUIRED 11.0)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_70,code=compute_70")
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${CUDA_LIBRARIES})
message(STATUS "CUDA_LIBRARIES ${CUDA_LIBRARIES}")
list(APPEND LINKER_LIBS cuda)

if (WHOLEMEMORY_NEED_PYTHON)
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

endif(WHOLEMEMORY_NEED_PYTHON)

add_library(whole_memory STATIC
        ${PROJECT_SOURCE_DIR}/wholememory/whole_memory.cc
        ${PROJECT_SOURCE_DIR}/wholememory/cuda_env_fns.cc
        ${PROJECT_SOURCE_DIR}/wholememory/file_utils.cc
        ${PROJECT_SOURCE_DIR}/wholememory/parallel_utils.cc
        ${PROJECT_SOURCE_DIR}/wholememory/whole_memory_embedding.cu
        ${PROJECT_SOURCE_DIR}/wholememory/whole_memory_memcpy.cu
        ${PROJECT_SOURCE_DIR}/wholememory/whole_memory_graph.cu
        ${PROJECT_SOURCE_DIR}/wholememory/whole_memory_graph_sampler.cu
        ${PROJECT_SOURCE_DIR}/wholememory/gnn_ops.cu
        ${PROJECT_SOURCE_DIR}/wholememory/domain_socket_communicator.cc)
set_property(TARGET whole_memory PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(whole_memory ${LINKER_LIBS} cuda)

if (WHOLEMEMORY_BUILD_PYTORCH)
    message(STATUS "Building PyTorch support library")
    add_subdirectory(wholememory/torch)
endif()

if (WHOLEMEMORY_ENABLE_TESTS)
    message(STATUS "Test enabled")
    add_subdirectory(test)
endif ()
