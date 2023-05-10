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
#include <error.hpp>
#include <gtest/gtest.h>

#include "./spmm_csr_no_weight_utils.hpp"
#include "gspmm_csr_weighted_test_utils.hpp"
#include "wholememory/initialize.hpp"
#include <wholememory/graph_op.h>
#include <wholememory/tensor_description.h>

typedef struct gSpMMCsrWeightedForwardTestParam {
  gSpMMCsrWeightedForwardTestParam& set_graph_row_num(int new_graph_row_num)
  {
    graph_row_num = new_graph_row_num;
    return *this;
  }
  gSpMMCsrWeightedForwardTestParam& set_graph_col_num(int new_graph_col_num)
  {
    graph_col_num = new_graph_col_num;
    return *this;
  }
  gSpMMCsrWeightedForwardTestParam& set_graph_edge_num(int new_graph_edge_num)
  {
    graph_edge_num = new_graph_edge_num;
    return *this;
  }
  gSpMMCsrWeightedForwardTestParam& set_num_heads(int new_num_heads)
  {
    num_heads = new_num_heads;
    return *this;
  }
  gSpMMCsrWeightedForwardTestParam& set_feature_dim(int new_feature_dim)
  {
    feature_dim = new_feature_dim;
    return *this;
  }
  gSpMMCsrWeightedForwardTestParam& set_weight_dtype(wholememory_dtype_t new_weight_dtype)
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

  int get_graph_row_num() const { return graph_row_num; }
  int get_graph_col_num() const { return graph_col_num; }
  int get_graph_edge_num() const { return graph_edge_num; }
  int get_num_heads() const { return num_heads; }
  int get_feature_dim() const { return feature_dim; }
  wholememory_dtype_t get_weight_dtype() const { return weight_dtype; }

  int graph_row_num                = 3;
  int graph_col_num                = 4;
  int graph_edge_num               = 10;
  int num_heads                    = 8;
  int feature_dim                  = 8;
  wholememory_dtype_t weight_dtype = WHOLEMEMORY_DT_FLOAT;
} gSpMMCsrWeightedForwardTestParam;

class gSpMMCsrWeightedForwardParameterTests
  : public ::testing::TestWithParam<gSpMMCsrWeightedForwardTestParam> {};

TEST_P(gSpMMCsrWeightedForwardParameterTests, gSpmmCsrWeightedForwardParameterTest)
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
  auto feature_dim            = params.get_feature_dim();
  auto weight_type            = params.get_weight_dtype();
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
  wholememory_tensor_description_t csr_row_ptr_tensor_desc, csr_col_ptr_tensor_desc,
    edge_weight_tensor_desc, feature_tensor_desc, output_feature_tensor_desc;
  wholememory_copy_array_desc_to_tensor(&csr_row_ptr_tensor_desc, &csr_row_ptr_array_desc);
  wholememory_copy_array_desc_to_tensor(&csr_col_ptr_tensor_desc, &csr_col_ptr_array_desc);
  wholememory_initialize_tensor_desc(&edge_weight_tensor_desc);
  edge_weight_tensor_desc.dim            = 2;
  edge_weight_tensor_desc.sizes[0]       = graph_edge_num;
  edge_weight_tensor_desc.sizes[1]       = num_heads;
  edge_weight_tensor_desc.strides[0]     = num_heads;
  edge_weight_tensor_desc.strides[1]     = 1;
  edge_weight_tensor_desc.storage_offset = 0;
  edge_weight_tensor_desc.dtype          = weight_type;
  wholememory_initialize_tensor_desc(&feature_tensor_desc);
  feature_tensor_desc.dim            = 3;
  feature_tensor_desc.sizes[0]       = graph_col_num;
  feature_tensor_desc.sizes[1]       = num_heads;
  feature_tensor_desc.sizes[2]       = feature_dim;
  feature_tensor_desc.strides[0]     = feature_dim * num_heads;
  feature_tensor_desc.strides[1]     = feature_dim;
  feature_tensor_desc.strides[2]     = 1;
  feature_tensor_desc.storage_offset = 0;
  feature_tensor_desc.dtype          = weight_type;

  wholememory_initialize_tensor_desc(&output_feature_tensor_desc);
  output_feature_tensor_desc.dim            = 3;
  output_feature_tensor_desc.sizes[0]       = graph_row_num;
  output_feature_tensor_desc.sizes[1]       = num_heads;
  output_feature_tensor_desc.sizes[2]       = feature_dim;
  output_feature_tensor_desc.strides[0]     = feature_dim * num_heads;
  output_feature_tensor_desc.strides[1]     = feature_dim;
  output_feature_tensor_desc.strides[2]     = 1;
  output_feature_tensor_desc.storage_offset = 0;
  output_feature_tensor_desc.dtype          = weight_type;

  void* host_edge_weight_ptr =
    (void*)malloc(wholememory_get_memory_size_from_tensor(&edge_weight_tensor_desc));
  void* host_feature_ptr =
    (void*)malloc(wholememory_get_memory_size_from_tensor(&feature_tensor_desc));
  void* host_output_feature_ptr =
    (void*)malloc(wholememory_get_memory_size_from_tensor(&output_feature_tensor_desc));
  void* host_ref_output_feature_ptr =
    (void*)malloc(wholememory_get_memory_size_from_tensor(&output_feature_tensor_desc));
  graph_ops::testing::gen_features(host_edge_weight_ptr, edge_weight_tensor_desc);
  graph_ops::testing::gen_features(host_feature_ptr, feature_tensor_desc);
  void *dev_csr_row_ptr, *dev_csr_col_ptr, *dev_edge_weight_ptr, *dev_feature_ptr,
    *dev_output_feature_ptr;
  EXPECT_EQ(
    cudaMalloc(&dev_csr_row_ptr, wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc)),
    cudaSuccess);
  EXPECT_EQ(
    cudaMalloc(&dev_csr_col_ptr, wholememory_get_memory_size_from_array(&csr_col_ptr_array_desc)),
    cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_edge_weight_ptr,
                       wholememory_get_memory_size_from_tensor(&edge_weight_tensor_desc)),
            cudaSuccess);
  EXPECT_EQ(
    cudaMalloc(&dev_feature_ptr, wholememory_get_memory_size_from_tensor(&feature_tensor_desc)),
    cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_output_feature_ptr,
                       wholememory_get_memory_size_from_tensor(&output_feature_tensor_desc)),
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

  EXPECT_EQ(cudaMemcpy(dev_edge_weight_ptr,
                       host_edge_weight_ptr,
                       wholememory_get_memory_size_from_tensor(&edge_weight_tensor_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  EXPECT_EQ(cudaMemcpy(dev_feature_ptr,
                       host_feature_ptr,
                       wholememory_get_memory_size_from_tensor(&feature_tensor_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  wholememory_tensor_t csr_row_ptr_tensor, csr_col_ptr_tensor, edge_weight_tensor, feature_tensor,
    output_feature_tensor;
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &csr_row_ptr_tensor, dev_csr_row_ptr, &csr_row_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &csr_col_ptr_tensor, dev_csr_col_ptr, &csr_col_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &edge_weight_tensor, dev_edge_weight_ptr, &edge_weight_tensor_desc),
            WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(
    wholememory_make_tensor_from_pointer(&feature_tensor, dev_feature_ptr, &feature_tensor_desc),
    WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &output_feature_tensor, dev_output_feature_ptr, &output_feature_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(gspmm_csr_weighted_forward(csr_row_ptr_tensor,
                                       csr_col_ptr_tensor,
                                       edge_weight_tensor,
                                       feature_tensor,
                                       output_feature_tensor,
                                       stream),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(cudaMemcpyAsync(host_output_feature_ptr,
                            dev_output_feature_ptr,
                            wholememory_get_memory_size_from_tensor(&output_feature_tensor_desc),
                            cudaMemcpyDeviceToHost,
                            stream),
            cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  graph_ops::testing::host_gspmm_csr_weighted_forward(host_csr_row_ptr,
                                                      csr_row_ptr_array_desc,
                                                      host_csr_col_ptr,
                                                      csr_col_ptr_array_desc,
                                                      host_edge_weight_ptr,
                                                      edge_weight_tensor_desc,
                                                      host_feature_ptr,
                                                      feature_tensor_desc,
                                                      host_ref_output_feature_ptr,
                                                      output_feature_tensor_desc);
  graph_ops::testing::host_check_float_tensor_same(host_output_feature_ptr,
                                                   output_feature_tensor_desc,
                                                   host_ref_output_feature_ptr,
                                                   output_feature_tensor_desc);

  EXPECT_EQ(cudaFree(dev_csr_row_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_csr_col_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_edge_weight_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_feature_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_output_feature_ptr), cudaSuccess);
  if (host_csr_row_ptr != nullptr) free(host_csr_row_ptr);
  if (host_csr_col_ptr != nullptr) free(host_csr_col_ptr);
  if (host_edge_weight_ptr != nullptr) free(host_edge_weight_ptr);
  if (host_feature_ptr != nullptr) free(host_feature_ptr);
  if (host_output_feature_ptr != nullptr) free(host_output_feature_ptr);
  if (host_ref_output_feature_ptr != nullptr) free(host_ref_output_feature_ptr);

  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
}

INSTANTIATE_TEST_SUITE_P(gSpmmCsrWeightFrowardTests,
                         gSpMMCsrWeightedForwardParameterTests,
                         ::testing::Values(gSpMMCsrWeightedForwardTestParam()
                                             .set_graph_row_num(1025)
                                             .set_graph_col_num(2379)
                                             .set_graph_edge_num(10793)
                                             .set_num_heads(32)
                                             .set_feature_dim(128)
                                             .set_weight_dtype(WHOLEMEMORY_DT_FLOAT)));
