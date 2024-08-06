#=============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
# option(BUILD_BENCHMARKS "Build wholegraph C++ benchmark tests" ON)
message(VERBOSE "WHOLEGRAPH: Building wholegraph C++ benchmarks: ${BUILD_BENCHMARKS}")

function(ConfigureBench)

    set(options OPTIONAL)
    set(oneValueArgs NAME)
    set(multiValueArgs PATH TARGETS CONFIGURATIONS)
    cmake_parse_arguments(ConfigureBench "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(BENCH_NAME ${ConfigureBench_NAME})

    add_executable(${BENCH_NAME} ${ConfigureBench_PATH})

    target_include_directories(${BENCH_NAME} PRIVATE "$<BUILD_INTERFACE:${WHOLEGRAPH_SOURCE_DIR}>/src")
    target_link_libraries(
            ${BENCH_NAME}
            PRIVATE wholegraph
            raft::raft
            rmm::rmm
            pthread
    )
    if(BUILD_WITH_NVSHMEM)
        target_compile_definitions(${BENCH_NAME} PRIVATE WITH_NVSHMEM_SUPPORT)
    endif()
    set_target_properties(
            ${BENCH_NAME}
            PROPERTIES # set target compile options
            INSTALL_RPATH "\$ORIGIN/../../../lib"
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            POSITION_INDEPENDENT_CODE ON
            RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${WHOLEGRAPH_BINARY_DIR}/gbench>"
            INTERFACE_POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(${BENCH_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Werror
            -Wno-error=deprecated-declarations>)

    install(
            TARGETS ${BENCH_NAME}
            COMPONENT testing
            DESTINATION bin/gbench/libwholegraph
            EXCLUDE_FROM_ALL
    )


endfunction()

if(BUILD_BENCHMARKS)
    ConfigureBench(
            NAME GATHER_SCATTER_BENCH
            PATH wholememory_ops/gather_scatter_bench.cu
            common/wholegraph_benchmark.cpp
    )

endif()
