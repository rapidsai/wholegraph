#=============================================================================
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

set(RAPIDS_VERSION "23.12")
set(WHOLEGRAPH_VERSION "${RAPIDS_VERSION}.00")

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  rapids-cmake
  GIT_REPOSITORY https://github.com/rapidsai/rapids-cmake.git
  GIT_TAG        origin/branch-${RAPIDS_VERSION}
)
FetchContent_MakeAvailable(rapids-cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86)
endif ()
rapids_cuda_init_architectures(WHOLEGRAPH)

project(WHOLEGRAPH VERSION ${WHOLEGRAPH_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_USE_RESPONSE_FILE_FOR_INCLUDES   FALSE)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES   FALSE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES   FALSE)

# Write the version header
rapids_cmake_write_version_file(include/wholegraph/version_config.hpp)

##############################################################################
# - User Options  ------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build libwholegraph shared libraries" ON)
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)
option(BUILD_WITH_NVSHMEM "Enable nvshmem integration" ON)
##############################################################################
# - Set options based on user defined one  -----------------------------------
set(_ctk_static_suffix "")
if(CUDA_STATIC_RUNTIME)
  set(_ctk_static_suffix "_static")
endif()

##############################################################################
# - Base rapids options ------------------------------------------------------
# default build type
#rapids_cmake_build_type(Release)
rapids_cmake_build_type(RelWithDebInfo)

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

message("CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
      message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

################################################################################
# - compiler options -----------------------------------------------------------

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
  BUILD_EXPORT_SET    wholegraph-exports
  INSTALL_EXPORT_SET  wholegraph-exports
)

set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WHOLEGRAPH_CXX_FLAGS "")
set(WHOLEGRAPH_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND WHOLEGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

#list(APPEND WHOLEGRAPH_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
#list(APPEND WHOLEGRAPH_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas=--disable-warnings)
#list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
#list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_CUDA_LINEINFO)
    list(APPEND WHOLEGRAPH_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND WHOLEGRAPH_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()

################################################################################
# - find CPM based dependencies  -----------------------------------------------

rapids_cpm_init()

include(./cmake/thirdparty/get_raft.cmake)
include(./cmake/thirdparty/get_nccl.cmake)
if(BUILD_WITH_NVSHMEM)
  include(./cmake/thirdparty/get_nvshmem.cmake)
endif()


file(GLOB WHOLEGRAPH_SOURCES
        "src/*.cpp"
        "src/wholememory/*.cpp"
        "src/wholememory_ops/*.cpp"
        "src/wholememory_ops/*.cu"
        "src/wholememory_ops/functions/*.cu"
        "src/wholegraph_ops/*.cpp"
        "src/wholegraph_ops/*.cu"
        "src/graph_ops/*.cu"
        "src/graph_ops/*.cpp")

add_library(wholegraph)
add_library(wholegraph::wholegraph ALIAS wholegraph)

target_sources(wholegraph PRIVATE ${WHOLEGRAPH_SOURCES})

set_target_properties(wholegraph
        PROPERTIES BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
        # set target compile options
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
        INTERFACE_POSITION_INDEPENDENT_CODE ON
        )

target_compile_options(wholegraph
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${WHOLEGRAPH_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${WHOLEGRAPH_CUDA_FLAGS}>"
)

#target_link_libraries(wholegraph PRIVATE -static-libgcc -static-libstdc++)

################################################################################
# - include paths --------------------------------------------------------------

set(WHOLEGRAPH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING
        "Path to wholegraph include directory")

target_include_directories(wholegraph
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${NCCL_INCLUDE_DIR}>"
        "$<BUILD_INTERFACE:${WHOLEGRAPH_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:include>"
)


################################################################################
# - link libraries -------------------------------------------------------------
target_link_libraries(wholegraph
PUBLIC
    CUDA::cuda_driver
    CUDA::cudart${_ctk_static_suffix}
    raft::raft
PRIVATE
    NCCL::NCCL
)

if(BUILD_WITH_NVSHMEM)
        set_source_files_properties("src/wholememory_ops/gather_op_impl_nvshmem.cu" "src/wholememory_ops/scatter_op_impl.nvshmem.cu"
        PROPERTIES
        # CUDA_SEPARABLE_COMPILATION ON
        COMPILE_FLAGS "-rdc=true"
        )
  target_compile_definitions(wholegraph PRIVATE WITH_NVSHMEM_SUPPORT)
  target_link_libraries(wholegraph PRIVATE nvshmem_host nvshmem_device)
endif()


# optionally build tests
if(BUILD_TESTS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(./cmake/thirdparty/get_gtest.cmake)
    include(CTest) # calls enable_testing()

    add_subdirectory(tests)
endif()

if(BUILD_WITH_NVSHMEM)
  add_library(nvshmem_wholememory_bootstrap SHARED)
  target_sources(nvshmem_wholememory_bootstrap PRIVATE src/nvshmem_plugins/nvshmem_wholememory_bootstrap.cpp)
  target_link_libraries(nvshmem_wholememory_bootstrap PRIVATE wholegraph nvshmem_host )
  target_include_directories(nvshmem_wholememory_bootstrap
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
get_target_property(NVSHMEM_LIBRARY_OUTPUT_DIR nvshmem LIBRARY_OUTPUT_DIRECTORY)

add_custom_command(TARGET nvshmem_wholememory_bootstrap POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${NVSHMEM_LIBRARY_OUTPUT_DIR}
                   ${CMAKE_BINARY_DIR}/nvshmem_lib)

endif()


##############################################################################
# - code checker -------------------------------------------------------------

include(./cmake/CodeChecker.cmake)
set(CLANG_FORMAT_EXE "clang-format")
set(CLANG_TIDY_EXE "clang-tidy")
set(FLAKE8_EXE "flake8")
if(CLANG_TOOL_PATH)
    set(CLANG_FORMAT_EXE ${CLANG_TOOL_PATH}/clang-format)
    set(CLANG_TIDY_EXE ${CLANG_TOOL_PATH}/clang-tidy)
endif()
add_code_checks(
        CWD          ${PROJECT_SOURCE_DIR}
        CLANG_FORMAT ${CLANG_FORMAT_EXE}
        CLANG_TIDY   ${CLANG_TIDY_EXE}
        FLAKE8       ${FLAKE8_EXE})

##############################################################################
# - install targets ----------------------------------------------------------

set(doc_string [=[
]=])
set(code_string
        [=[
]=])

set(WHOLEGRAPH_PUBLIC_HEADERS
        wholememory/device_reference.cuh
        wholememory/embedding.h
        wholememory/env_func_ptrs.h
        wholememory/global_reference.h
        wholememory/tensor_description.h
        wholememory/wholememory.h
        wholememory/wholememory_tensor.h
        wholememory/wholememory_op.h
        wholememory/wholegraph_op.h
        wholememory/graph_op.h
        )

foreach(file IN LISTS WHOLEGRAPH_PUBLIC_HEADERS)
    cmake_path(GET file PARENT_PATH file_out_)
    string(JOIN "/" out_path "include" ${file_out_})
    install(FILES "${WHOLEGRAPH_INCLUDE_DIR}/${file}"
            DESTINATION "${out_path}")
    #install(FILES "${WHOLEGRAPH_INCLUDE_DIR}/${file}"
    #        DESTINATION "${file}")
endforeach()

rapids_cmake_install_lib_dir( lib_dir )

set(PROGRAM_PERMISSIONS_DEFAULT
        OWNER_WRITE OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)

install(
        TARGETS     wholegraph
        DESTINATION ${lib_dir}
        PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT}
        EXPORT wholegraph-exports)

rapids_export(
        INSTALL wholegraph
        EXPORT_SET wholegraph-exports
        GLOBAL_TARGETS wholegraph
        NAMESPACE wholegraph::
        DOCUMENTATION doc_string
        FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
        BUILD wholegraph
        EXPORT_SET wholegraph-exports
        GLOBAL_TARGETS wholegraph
        NAMESPACE wholegraph::
        DOCUMENTATION doc_string
        FINAL_CODE_BLOCK code_string)

        ################################################################################
# - make documentation ---------------------------------------------------------
# requires doxygen and graphviz to be installed
# from build directory, run make doxygen

##############################################################################
# - doxygen docs -------------------------------------------------------------

find_package(Doxygen 1.8.11)
if(Doxygen_FOUND)
        add_custom_command(OUTPUT WHOLEGRAPH_DOXYGEN
                           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                           COMMAND doxygen Doxyfile
                           VERBATIM)

        add_custom_target(doxygen DEPENDS WHOLEGRAPH_DOXYGEN)
endif()
