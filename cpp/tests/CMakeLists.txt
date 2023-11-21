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
# Build options
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
option(CODE_COVERAGE "Enable generating code coverage with gcov." OFF)

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTestInternal TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_include_directories(${TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${WHOLEGRAPH_SOURCE_DIR}>/src")
    target_link_libraries(${TEST_NAME} PRIVATE GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main
            wholegraph raft::raft rmm::rmm pthread)
    if(BUILD_WITH_NVSHMEM)
            target_compile_definitions(${TEST_NAME} PRIVATE WITH_NVSHMEM_SUPPORT)
    endif()

    set_target_properties(
            ${TEST_NAME}
            PROPERTIES POSITION_INDEPENDENT_CODE ON
            RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${WHOLEGRAPH_BINARY_DIR}/gtests>"
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INSTALL_RPATH "\$ORIGIN/../../../lib")
    target_compile_definitions(${TEST_NAME}
            PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${WHOLEGRAPH_LOGGING_LEVEL}")
    target_compile_options(${TEST_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Werror
            -Wno-error=deprecated-declarations>)

    if(DISABLE_DEPRECATION_WARNING)
        target_compile_options(
                ${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
        target_compile_options(${TEST_NAME}
                PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
    endif()

    if(CODE_COVERAGE)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

            set(KEEP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp)
            make_directory(${KEEP_DIR})
            target_compile_options(${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--keep
                    --keep-dir=${KEEP_DIR}>)
            target_compile_options(
                    ${TEST_NAME}
                    PUBLIC
                    $<$<COMPILE_LANGUAGE:CUDA>:-O0
                    -Xcompiler=--coverage,-fprofile-abs-path,-fkeep-inline-functions,-fno-elide-constructors>)
            target_compile_options(
                    ${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-O0 --coverage -fprofile-abs-path
                    -fkeep-inline-functions -fno-elide-constructors>)
            target_link_options(${TEST_NAME} PRIVATE --coverage)
            target_link_libraries(${TEST_NAME} gcov)
        endif()

        # Add coverage-generated files to clean target
        list(APPEND COVERAGE_CLEAN_FILES "**/*.gcno" "**/*.gcda")
        set_property(
                TARGET ${TEST_NAME}
                APPEND
                PROPERTY ADDITIONAL_CLEAN_FILES ${COVERAGE_CLEAN_FILES})
    endif()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}")
    install(
            TARGETS ${TEST_NAME}
            COMPONENT testing
            DESTINATION bin/gtests/libwholegraph
            EXCLUDE_FROM_ALL)
endfunction()

# Wrapper around `ConfigureTestInternal` that builds tests both with and without per thread default
# stream
function(ConfigureTest TEST_NAME)
    # Test with legacy default stream.
    ConfigureTestInternal(${TEST_NAME} ${ARGN})
endfunction()

# parallel_utils tests
ConfigureTest(PARALLEL_UTILS_TEST parallel_utils_tests.cpp)

# wholememory communicator tests
ConfigureTest(WHOLEMEMORY_COMM_TEST wholememory/wholememory_comm_tests.cpp)

# wholememory handle tests
ConfigureTest(WHOLEMEMORY_HANDLE_TEST wholememory/wholememory_handle_tests.cpp)

# wholememory tensor tests
ConfigureTest(WHOLEMEMORY_TENSOR_TEST wholememory/wholememory_tensor_tests.cpp)

# wholememory gather op tests
ConfigureTest(WHOLEMEMORY_GATHER_TEST wholememory_ops/wholememory_gather_tests.cu wholememory_ops/embedding_test_utils.cu)

# wholememory scatter op tests
ConfigureTest(WHOLEMEMORY_SCATTER_TEST wholememory_ops/wholememory_scatter_tests.cu wholememory_ops/embedding_test_utils.cu)

#wholegraph unweighted samping op tests
ConfigureTest(WHOLEGRAPH_CSR_UNWEIGHTED_SAMPLE_WITHOUT_REPLACEMENT_TEST wholegraph_ops/wholegraph_csr_unweighted_sample_without_replacement_tests.cu wholegraph_ops/graph_sampling_test_utils.cu)

#wholegraph weighted samping op tests
ConfigureTest(WHOLEGRAPH_CSR_WEIGHTED_SAMPLE_WITHOUT_REPLACEMENT_TEST wholegraph_ops/wholegraph_csr_weighted_sample_without_replacement_tests.cu wholegraph_ops/graph_sampling_test_utils.cu)

#wholegraph cache set tests
ConfigureTest(WHOLEGRAPH_CACHESET_TEST wholememory_ops/cacheset_tests.cu)

#wholegraph embedding tests
ConfigureTest(WHOLEGRAPH_EMBEDDING_TEST wholememory_ops/wholememory_embedding_tests.cu wholememory_ops/embedding_test_utils.cu)

#wholegraph embedding gradient apply tests
ConfigureTest(WHOLEGRAPH_EMBEDDING_GRADIENT_APPLY_TEST wholememory_ops/wholememory_embedding_gradient_apply_tests.cu wholememory_ops/embedding_test_utils.cu)

#graph append unique op tests
ConfigureTest(GRAPH_APPEND_UNIQUE_TEST graph_ops/append_unique_tests.cu graph_ops/append_unique_test_utils.cu wholegraph_ops/graph_sampling_test_utils.cu)

#graph csr add self loop op tests
ConfigureTest(GRAPH_CSR_ADD_SELF_LOOP_TEST graph_ops/csr_add_self_loop_tests.cu graph_ops/csr_add_self_loop_utils.cu wholegraph_ops/graph_sampling_test_utils.cu)
