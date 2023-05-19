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
#include <gtest/gtest.h>

#include "wholememory/env_func_ptrs.hpp"
#include <wholememory/graph_op.h>
#include <wholememory/tensor_description.h>

#include "../wholegraph_ops/graph_sampling_test_utils.hpp"
#include "csr_add_self_loop_utils.hpp"
#include "error.hpp"

typedef struct CsrAddSelfLoopTestParam {
  CsrAddSelfLoopTestParam& set_graph_row_num(int new_graph_row_num)
  {
    graph_row_num = new_graph_row_num;
    return *this;
  }
  CsrAddSelfLoopTestParam& set_graph_col_num(int new_graph_col_num)
  {
    graph_col_num = new_graph_col_num;
    return *this;
  }
  CsrAddSelfLoopTestParam& set_graph_edge_num(int new_graph_edge_num)
  {
    graph_edge_num = new_graph_edge_num;
    return *this;
  }

  wholememory_array_description_t get_csr_row_ptr_array_desc() const
  {
    return wholememory_create_array_desc(graph_row_num + 1, 0, WHOLEMEMORY_DT_INT);
  }
  wholememory_array_description_t get_csr_col_ptr_array_desc() const
  {
    return wholememory_create_array_desc(graph_edge_num, 0, WHOLEMEMORY_DT_INT);
  }
  int get_graph_row_num() const { return graph_row_num; }
  int get_graph_col_num() const { return graph_col_num; }
  int get_graph_edge_num() const { return graph_edge_num; }

  int graph_row_num  = 3;
  int graph_col_num  = 5;
  int graph_edge_num = 9;
} CsrAddSelfLoopTestParam;

class GraphCsrAddSelfLoopParameterTests : public ::testing::TestWithParam<CsrAddSelfLoopTestParam> {
};

TEST_P(GraphCsrAddSelfLoopParameterTests, CsrAddSelfLoopParameterTest)
{
  auto params = GetParam();
  int dev_count;
  EXPECT_EQ(cudaGetDeviceCount(&dev_count), cudaSuccess);
  EXPECT_GE(dev_count, 1);

  cudaStream_t stream;
  EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  auto graph_row_num          = params.get_graph_row_num();
  auto graph_col_num          = params.get_graph_col_num();
  auto graph_edge_num         = params.get_graph_edge_num();
  auto csr_row_ptr_array_desc = params.get_csr_row_ptr_array_desc();
  auto csr_col_ptr_array_desc = params.get_csr_col_ptr_array_desc();
  void* host_csr_row_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc));
  void* host_csr_col_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&csr_col_ptr_array_desc));
  graph_ops::testing::gen_local_csr_graph(graph_row_num,
                                          graph_col_num,
                                          graph_edge_num,
                                          host_csr_row_ptr,
                                          csr_row_ptr_array_desc,
                                          host_csr_col_ptr,
                                          csr_col_ptr_array_desc);

  int output_edge_num = graph_edge_num + graph_row_num;
  auto output_csr_col_ptr_array_desc =
    wholememory_create_array_desc(output_edge_num, 0, WHOLEMEMORY_DT_INT);
  void* host_output_csr_row_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc));
  void* host_output_csr_col_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&output_csr_col_ptr_array_desc));

  void* host_ref_output_csr_row_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc));
  void* host_ref_output_csr_col_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&output_csr_col_ptr_array_desc));

  void *dev_csr_row_ptr, *dev_csr_col_ptr, *dev_output_csr_row_ptr, *dev_output_csr_col_ptr;
  EXPECT_EQ(
    cudaMalloc(&dev_csr_row_ptr, wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc)),
    cudaSuccess);
  EXPECT_EQ(
    cudaMalloc(&dev_csr_col_ptr, wholememory_get_memory_size_from_array(&csr_col_ptr_array_desc)),
    cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_output_csr_row_ptr,
                       wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc)),
            cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_output_csr_col_ptr,
                       wholememory_get_memory_size_from_array(&output_csr_col_ptr_array_desc)),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dev_csr_row_ptr,
                       host_csr_row_ptr,
                       wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dev_csr_col_ptr,
                       host_csr_col_ptr,
                       wholememory_get_memory_size_from_array(&csr_col_ptr_array_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  wholememory_tensor_description_t csr_row_ptr_tensor_desc, csr_col_ptr_tensor_desc,
    output_csr_row_ptr_tensor_desc, output_csr_col_ptr_tensor_desc;
  wholememory_tensor_t csr_row_ptr_tensor, csr_col_ptr_tensor, output_csr_row_ptr_tensor,
    output_csr_col_ptr_tensor;
  wholememory_copy_array_desc_to_tensor(&csr_row_ptr_tensor_desc, &csr_row_ptr_array_desc);
  wholememory_copy_array_desc_to_tensor(&csr_col_ptr_tensor_desc, &csr_col_ptr_array_desc);
  wholememory_copy_array_desc_to_tensor(&output_csr_row_ptr_tensor_desc, &csr_row_ptr_array_desc);
  wholememory_copy_array_desc_to_tensor(&output_csr_col_ptr_tensor_desc,
                                        &output_csr_col_ptr_array_desc);

  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &csr_row_ptr_tensor, dev_csr_row_ptr, &csr_row_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &csr_col_ptr_tensor, dev_csr_col_ptr, &csr_col_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &output_csr_row_ptr_tensor, dev_output_csr_row_ptr, &output_csr_row_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &output_csr_col_ptr_tensor, dev_output_csr_col_ptr, &output_csr_col_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(csr_add_self_loop(csr_row_ptr_tensor,
                              csr_col_ptr_tensor,
                              output_csr_row_ptr_tensor,
                              output_csr_col_ptr_tensor,
                              stream),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(host_output_csr_row_ptr,
                       dev_output_csr_row_ptr,
                       wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_EQ(cudaMemcpy(host_output_csr_col_ptr,
                       dev_output_csr_col_ptr,
                       wholememory_get_memory_size_from_array(&output_csr_col_ptr_array_desc),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  graph_ops::testing::host_csr_add_self_loop(host_csr_row_ptr,
                                             csr_row_ptr_array_desc,
                                             host_csr_col_ptr,
                                             csr_col_ptr_array_desc,
                                             host_ref_output_csr_row_ptr,
                                             csr_row_ptr_array_desc,
                                             host_ref_output_csr_col_ptr,
                                             output_csr_col_ptr_array_desc);
  wholegraph_ops::testing::host_check_two_array_same(host_output_csr_row_ptr,
                                                     csr_row_ptr_array_desc,
                                                     host_ref_output_csr_row_ptr,
                                                     csr_row_ptr_array_desc);
  wholegraph_ops::testing::host_check_two_array_same(host_output_csr_col_ptr,
                                                     output_csr_col_ptr_array_desc,
                                                     host_ref_output_csr_col_ptr,
                                                     output_csr_col_ptr_array_desc);
  EXPECT_EQ(cudaFree(dev_csr_row_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_csr_col_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_output_csr_row_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_output_csr_col_ptr), cudaSuccess);

  if (host_csr_row_ptr != nullptr) free(host_csr_row_ptr);
  if (host_csr_col_ptr != nullptr) free(host_csr_col_ptr);
  if (host_output_csr_row_ptr != nullptr) free(host_output_csr_row_ptr);
  if (host_output_csr_col_ptr != nullptr) free(host_output_csr_col_ptr);
  if (host_ref_output_csr_row_ptr != nullptr) free(host_ref_output_csr_row_ptr);
  if (host_ref_output_csr_col_ptr != nullptr) free(host_ref_output_csr_col_ptr);

  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
}

INSTANTIATE_TEST_SUITE_P(CsrAddSelfLoopOpTests,
                         GraphCsrAddSelfLoopParameterTests,
                         ::testing::Values(CsrAddSelfLoopTestParam()
                                             .set_graph_row_num(1357)
                                             .set_graph_col_num(2589)
                                             .set_graph_edge_num(19087)));
