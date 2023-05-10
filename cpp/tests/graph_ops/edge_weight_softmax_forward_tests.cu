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
#include "./spmm_csr_no_weight_utils.hpp"
#include "edge_weight_softmax_test_utils.hpp"
#include "spadd_gat_csr_utils.hpp"
#include "wholememory/initialize.hpp"
#include <error.hpp>
#include <gtest/gtest.h>
#include <wholememory/graph_op.h>
#include <wholememory/tensor_description.h>

typedef struct EdgeWeightSoftmaxForwardTestParam {
  EdgeWeightSoftmaxForwardTestParam& set_graph_row_num(int new_graph_row_num)
  {
    graph_row_num = new_graph_row_num;
    return *this;
  }
  EdgeWeightSoftmaxForwardTestParam& set_graph_col_num(int new_graph_col_num)
  {
    graph_col_num = new_graph_col_num;
    return *this;
  }
  EdgeWeightSoftmaxForwardTestParam& set_graph_edge_num(int new_graph_edge_num)
  {
    graph_edge_num = new_graph_edge_num;
    return *this;
  }
  EdgeWeightSoftmaxForwardTestParam& set_num_heads(int new_num_heads)
  {
    num_heads = new_num_heads;
    return *this;
  }
  EdgeWeightSoftmaxForwardTestParam& set_weight_dtype(wholememory_dtype_t new_weight_dtype)
  {
    weight_dtype = new_weight_dtype;
    return *this;
  }

  wholememory_array_description_t get_csr_row_ptr_array_desc()
  {
    return wholememory_create_array_desc(graph_row_num + 1, 0, WHOLEMEMORY_DT_INT);
  }
  wholememory_array_description_t get_csr_col_ptr_array_desc()
  {
    return wholememory_create_array_desc(graph_edge_num, 0, WHOLEMEMORY_DT_INT);
  }

  wholememory_matrix_description_t get_weight_matrix_desc() const
  {
    int64_t sizes[2] = {graph_edge_num, num_heads};
    return wholememory_create_matrix_desc(sizes, num_heads, 0, weight_dtype);
  }

  int get_graph_row_num() const { return graph_row_num; }
  int get_graph_col_num() const { return graph_col_num; }
  int get_graph_edge_num() const { return graph_edge_num; }
  int get_num_heads() const { return num_heads; }
  wholememory_dtype_t get_weight_dtype() const { return weight_dtype; }

  int graph_row_num                = 3;
  int graph_col_num                = 4;
  int graph_edge_num               = 10;
  int num_heads                    = 8;
  wholememory_dtype_t weight_dtype = WHOLEMEMORY_DT_FLOAT;

} EdgeWeightSoftmaxForwardTestParam;

class EdgeWeightSoftmaxForwardParameterTests
  : public ::testing::TestWithParam<EdgeWeightSoftmaxForwardTestParam> {};

TEST_P(EdgeWeightSoftmaxForwardParameterTests, EdgeWeightSoftmaxForwarParameterTest)
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
  auto num_heads              = params.get_num_heads();
  auto csr_row_ptr_array_desc = params.get_csr_row_ptr_array_desc();
  auto csr_col_ptr_array_desc = params.get_csr_col_ptr_array_desc();
  auto weight_matrix_desc     = params.get_weight_matrix_desc();
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

  void* host_weight_ptr =
    (void*)malloc(wholememory_get_memory_size_from_matrix(&weight_matrix_desc));
  graph_ops::testing::gen_features(host_weight_ptr, weight_matrix_desc);
  void* host_output_weight_ptr =
    (void*)malloc(wholememory_get_memory_size_from_matrix(&weight_matrix_desc));
  void* host_ref_output_weight_ptr =
    (void*)malloc(wholememory_get_memory_size_from_matrix(&weight_matrix_desc));
  void *dev_csr_row_ptr, *dev_weight_ptr, *dev_output_weight_ptr;
  EXPECT_EQ(
    cudaMalloc(&dev_csr_row_ptr, wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc)),
    cudaSuccess);
  EXPECT_EQ(
    cudaMalloc(&dev_weight_ptr, wholememory_get_memory_size_from_matrix(&weight_matrix_desc)),
    cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_output_weight_ptr,
                       wholememory_get_memory_size_from_matrix(&weight_matrix_desc)),
            cudaSuccess);

  EXPECT_EQ(cudaMemcpy(dev_csr_row_ptr,
                       host_csr_row_ptr,
                       wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dev_weight_ptr,
                       host_weight_ptr,
                       wholememory_get_memory_size_from_matrix(&weight_matrix_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  wholememory_tensor_description_t csr_row_ptr_tensor_desc, weight_tensor_desc,
    output_weight_tensor_desc;
  wholememory_tensor_t csr_row_ptr_tensor, weight_tensor, output_weight_tensor;
  wholememory_copy_array_desc_to_tensor(&csr_row_ptr_tensor_desc, &csr_row_ptr_array_desc);
  wholememory_copy_matrix_desc_to_tensor(&weight_tensor_desc, &weight_matrix_desc);
  wholememory_copy_matrix_desc_to_tensor(&output_weight_tensor_desc, &weight_matrix_desc);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &csr_row_ptr_tensor, dev_csr_row_ptr, &csr_row_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(
    wholememory_make_tensor_from_pointer(&weight_tensor, dev_weight_ptr, &weight_tensor_desc),
    WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &output_weight_tensor, dev_output_weight_ptr, &weight_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(edge_weight_softmax_csr_forward(
              csr_row_ptr_tensor, weight_tensor, output_weight_tensor, stream),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(cudaMemcpyAsync(host_output_weight_ptr,
                            dev_output_weight_ptr,
                            wholememory_get_memory_size_from_matrix(&weight_matrix_desc),
                            cudaMemcpyDeviceToHost,
                            stream),
            cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  graph_ops::testing::host_edge_weight_softmax_forward(host_csr_row_ptr,
                                                       csr_row_ptr_array_desc,
                                                       host_weight_ptr,
                                                       weight_matrix_desc,
                                                       host_ref_output_weight_ptr,
                                                       weight_matrix_desc);
  graph_ops::testing::host_check_float_matrix_same(
    host_output_weight_ptr, weight_matrix_desc, host_ref_output_weight_ptr, weight_matrix_desc);

  EXPECT_EQ(cudaFree(dev_csr_row_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_weight_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_output_weight_ptr), cudaSuccess);
  if (host_csr_row_ptr != nullptr) free(host_csr_row_ptr);
  if (host_weight_ptr != nullptr) free(host_weight_ptr);
  if (host_output_weight_ptr != nullptr) free(host_output_weight_ptr);
  if (host_ref_output_weight_ptr != nullptr) free(host_ref_output_weight_ptr);

  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
}

INSTANTIATE_TEST_SUITE_P(EdgeWeightSoftmaxForwardTests,
                         EdgeWeightSoftmaxForwardParameterTests,
                         ::testing::Values(EdgeWeightSoftmaxForwardTestParam()
                                             .set_graph_row_num(1059)
                                             .set_graph_col_num(5689)
                                             .set_graph_edge_num(23089)
                                             .set_num_heads(21)));
