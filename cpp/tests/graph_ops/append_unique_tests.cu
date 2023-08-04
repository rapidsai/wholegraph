/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>

#include "../wholegraph_ops/graph_sampling_test_utils.hpp"
#include "../wholememory/wholememory_test_utils.hpp"
#include "append_unique_test_utils.hpp"
#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/env_func_ptrs.hpp"
#include "wholememory/initialize.hpp"
#include <wholememory/graph_op.h>
#include <wholememory/tensor_description.h>

typedef struct GraphAppendUniqueTestParam {
  GraphAppendUniqueTestParam& set_target_node_count(int new_target_node_count)
  {
    target_node_count = new_target_node_count;
    return *this;
  }
  GraphAppendUniqueTestParam& set_neighbor_node_count(int new_neighbor_node_count)
  {
    neighbor_node_count = new_neighbor_node_count;
    return *this;
  }

  GraphAppendUniqueTestParam& set_target_dtype(wholememory_dtype_t new_target_node_dtype)
  {
    target_node_dtype   = new_target_node_dtype;
    neighbor_node_dtype = new_target_node_dtype;
    return *this;
  }
  wholememory_array_description_t get_target_node_desc() const
  {
    return wholememory_create_array_desc(target_node_count, 0, target_node_dtype);
  }
  wholememory_array_description_t get_neighbor_node_desc() const
  {
    return wholememory_create_array_desc(neighbor_node_count, 0, neighbor_node_dtype);
  }

  int64_t get_target_node_count() const { return target_node_count; }
  int64_t get_neighbor_node_count() const { return neighbor_node_count; }
  wholememory_dtype_t target_node_dtype   = WHOLEMEMORY_DT_INT;
  wholememory_dtype_t neighbor_node_dtype = target_node_dtype;
  int64_t target_node_count               = 10;
  int64_t neighbor_node_count             = 100;
} GraphAppendUniqueTestParam;

class GraphAppendUniqueParameterTests
  : public ::testing::TestWithParam<GraphAppendUniqueTestParam> {};

TEST_P(GraphAppendUniqueParameterTests, AppendUniqueTest)
{
  auto params = GetParam();
  int dev_count;
  EXPECT_EQ(cudaGetDeviceCount(&dev_count), cudaSuccess);
  EXPECT_GE(dev_count, 1);

  cudaStream_t stream;
  EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  auto target_node_count   = params.get_target_node_count();
  auto neighbor_node_count = params.get_neighbor_node_count();
  auto target_node_desc    = params.get_target_node_desc();
  auto neighbor_node_desc  = params.get_neighbor_node_desc();

  size_t target_node_size   = wholememory_get_memory_size_from_array(&target_node_desc);
  size_t neighbor_node_size = wholememory_get_memory_size_from_array(&neighbor_node_desc);

  void *host_target_nodes_ptr = nullptr, *host_neighbor_nodes_ptr = nullptr;
  void *dev_target_nodes_ptr = nullptr, *dev_neighbor_nodes_ptr = nullptr;
  void *host_output_unique_nodes_ptr = nullptr, *ref_host_output_unique_nodes_ptr = nullptr;
  int *host_output_neighbor_raw_to_unique_mapping_ptr     = nullptr,
      *ref_host_output_neighbor_raw_to_unique_mapping_ptr = nullptr;
  int* dev_output_neighbor_raw_to_unique_mapping_ptr      = nullptr;
  wholememory_array_description_t neighbor_raw_to_unique_mapping_desc =
    wholememory_create_array_desc(neighbor_node_count, 0, WHOLEMEMORY_DT_INT);

  EXPECT_EQ(cudaMallocHost(&host_target_nodes_ptr, target_node_size), cudaSuccess);
  EXPECT_EQ(cudaMallocHost(&host_neighbor_nodes_ptr, neighbor_node_size), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_target_nodes_ptr, target_node_size), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_neighbor_nodes_ptr, neighbor_node_size), cudaSuccess);
  EXPECT_EQ(
    cudaMalloc(&dev_output_neighbor_raw_to_unique_mapping_ptr, neighbor_node_count * sizeof(int)),
    cudaSuccess);
  int64_t total_node_count = neighbor_node_count + target_node_count;
  graph_ops::testing::gen_node_ids(host_target_nodes_ptr, target_node_desc, total_node_count, true);
  graph_ops::testing::gen_node_ids(
    host_neighbor_nodes_ptr, neighbor_node_desc, total_node_count, false);

  EXPECT_EQ(cudaMemcpyAsync(dev_target_nodes_ptr,
                            host_target_nodes_ptr,
                            target_node_size,
                            cudaMemcpyHostToDevice,
                            stream),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpyAsync(dev_neighbor_nodes_ptr,
                            host_neighbor_nodes_ptr,
                            neighbor_node_size,
                            cudaMemcpyHostToDevice,
                            stream),
            cudaSuccess);
  wholememory_tensor_t target_node_tensor, neighbor_node_tensor,
    output_neighbor_raw_to_unique_mapping_tensor;
  wholememory_tensor_description_t target_node_tensor_desc, neighbor_node_tensor_desc,
    output_neighbor_raw_to_unique_mapping_tensor_desc;
  wholememory_copy_array_desc_to_tensor(&target_node_tensor_desc, &target_node_desc);
  wholememory_copy_array_desc_to_tensor(&neighbor_node_tensor_desc, &neighbor_node_desc);
  wholememory_copy_array_desc_to_tensor(&output_neighbor_raw_to_unique_mapping_tensor_desc,
                                        &neighbor_raw_to_unique_mapping_desc);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &target_node_tensor, dev_target_nodes_ptr, &target_node_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &neighbor_node_tensor, dev_neighbor_nodes_ptr, &neighbor_node_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(
    wholememory_make_tensor_from_pointer(&output_neighbor_raw_to_unique_mapping_tensor,
                                         dev_output_neighbor_raw_to_unique_mapping_ptr,
                                         &output_neighbor_raw_to_unique_mapping_tensor_desc),
    WHOLEMEMORY_SUCCESS);
  wholememory_env_func_t* default_env_func = wholememory::get_default_env_func();
  wholememory::default_memory_context_t output_unique_node_memory_ctx;
  EXPECT_EQ(graph_append_unique(target_node_tensor,
                                neighbor_node_tensor,
                                &output_unique_node_memory_ctx,
                                output_neighbor_raw_to_unique_mapping_tensor,
                                default_env_func,
                                stream),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  int total_unique_count = output_unique_node_memory_ctx.desc.sizes[0];

  host_output_unique_nodes_ptr =
    malloc(total_unique_count * wholememory_dtype_get_element_size(target_node_desc.dtype));
  host_output_neighbor_raw_to_unique_mapping_ptr = (int*)malloc(neighbor_node_count * sizeof(int));
  EXPECT_EQ(
    cudaMemcpyAsync(host_output_unique_nodes_ptr,
                    output_unique_node_memory_ctx.ptr,
                    total_unique_count * wholememory_dtype_get_element_size(target_node_desc.dtype),
                    cudaMemcpyDeviceToHost,
                    stream),
    cudaSuccess);
  EXPECT_EQ(cudaMemcpyAsync(host_output_neighbor_raw_to_unique_mapping_ptr,
                            dev_output_neighbor_raw_to_unique_mapping_ptr,
                            neighbor_node_count * sizeof(int),
                            cudaMemcpyDeviceToHost,
                            stream),
            cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  int ref_total_unique_node_count;
  graph_ops::testing::host_append_unique(host_target_nodes_ptr,
                                         target_node_desc,
                                         host_neighbor_nodes_ptr,
                                         neighbor_node_desc,
                                         &ref_total_unique_node_count,
                                         &ref_host_output_unique_nodes_ptr);

  EXPECT_EQ(total_unique_count, ref_total_unique_node_count);
  graph_ops::testing::host_gen_append_unique_neighbor_raw_to_unique(
    host_output_unique_nodes_ptr,
    wholememory_create_array_desc(total_unique_count, 0, target_node_desc.dtype),
    host_neighbor_nodes_ptr,
    neighbor_node_desc,
    (void**)&ref_host_output_neighbor_raw_to_unique_mapping_ptr,
    neighbor_raw_to_unique_mapping_desc);

  if (target_node_desc.dtype == WHOLEMEMORY_DT_INT) {
    std::sort(static_cast<int*>(host_output_unique_nodes_ptr) + target_node_count,
              static_cast<int*>(host_output_unique_nodes_ptr) + total_unique_count);
    std::sort(static_cast<int*>(ref_host_output_unique_nodes_ptr) + target_node_count,
              static_cast<int*>(ref_host_output_unique_nodes_ptr) + total_unique_count);
  } else if (target_node_desc.dtype == WHOLEMEMORY_DT_INT64) {
    std::sort(static_cast<int64_t*>(host_output_unique_nodes_ptr) + target_node_count,
              static_cast<int64_t*>(host_output_unique_nodes_ptr) + total_unique_count);
    std::sort(static_cast<int64_t*>(ref_host_output_unique_nodes_ptr) + target_node_count,
              static_cast<int64_t*>(ref_host_output_unique_nodes_ptr) + total_unique_count);
  }

  wholegraph_ops::testing::host_check_two_array_same(
    host_output_unique_nodes_ptr,
    wholememory_create_array_desc(total_unique_count, 0, target_node_desc.dtype),
    ref_host_output_unique_nodes_ptr,
    wholememory_create_array_desc(ref_total_unique_node_count, 0, target_node_desc.dtype));

  wholegraph_ops::testing::host_check_two_array_same(
    host_output_neighbor_raw_to_unique_mapping_ptr,
    neighbor_raw_to_unique_mapping_desc,
    ref_host_output_neighbor_raw_to_unique_mapping_ptr,
    neighbor_raw_to_unique_mapping_desc);

  (default_env_func->output_fns).free_fn(&output_unique_node_memory_ctx, nullptr);
  if (host_output_unique_nodes_ptr != nullptr) { free(host_output_unique_nodes_ptr); }
  if (host_output_neighbor_raw_to_unique_mapping_ptr != nullptr) {
    free(host_output_neighbor_raw_to_unique_mapping_ptr);
  }
  if (ref_host_output_unique_nodes_ptr != nullptr) { free(ref_host_output_unique_nodes_ptr); }
  if (ref_host_output_neighbor_raw_to_unique_mapping_ptr != nullptr) {
    free(ref_host_output_neighbor_raw_to_unique_mapping_ptr);
  }

  EXPECT_EQ(cudaFreeHost(host_target_nodes_ptr), cudaSuccess);
  EXPECT_EQ(cudaFreeHost(host_neighbor_nodes_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_target_nodes_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_neighbor_nodes_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_output_neighbor_raw_to_unique_mapping_ptr), cudaSuccess);

  WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
}

INSTANTIATE_TEST_SUITE_P(GraphAppendUniqueOpTests,
                         GraphAppendUniqueParameterTests,
                         ::testing::Values(GraphAppendUniqueTestParam()
                                             .set_target_node_count(3)
                                             .set_neighbor_node_count(10),
                                           GraphAppendUniqueTestParam()
                                             .set_target_node_count(53)
                                             .set_neighbor_node_count(123)
                                             .set_target_dtype(WHOLEMEMORY_DT_INT),
                                           GraphAppendUniqueTestParam()
                                             .set_target_node_count(57)
                                             .set_neighbor_node_count(1235)
                                             .set_target_dtype(WHOLEMEMORY_DT_INT64)));
