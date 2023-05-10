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
#include "gspmm_csr_weighted_test_utils.hpp"
#include <gtest/gtest.h>

namespace graph_ops {
namespace testing {

template <typename WeightType>
void host_get_gpsmm_csr_weighted_forward(
  int* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* host_edge_weight_ptr,
  wholememory_tensor_description_t edge_weight_tensor_desc,
  void* host_feature_ptr,
  wholememory_tensor_description_t feature_tensor_desc,
  void* host_ref_output_feature_ptr,
  wholememory_tensor_description_t output_feature_tensor_desc)
{
  int64_t row_num                = csr_row_ptr_array_desc.size - 1;
  int64_t head_num               = edge_weight_tensor_desc.sizes[1];
  int64_t emb_dim                = feature_tensor_desc.sizes[2];
  WeightType* edge_weight_ptr    = static_cast<WeightType*>(host_edge_weight_ptr);
  WeightType* feature_ptr        = static_cast<WeightType*>(host_feature_ptr);
  WeightType* output_feature_ptr = static_cast<WeightType*>(host_ref_output_feature_ptr);

  for (int64_t head_id = 0; head_id < head_num; head_id++) {
    for (int64_t emb_id = 0; emb_id < emb_dim; emb_id++) {
      for (int64_t row_id = 0; row_id < row_num; row_id++) {
        int start = host_csr_row_ptr[row_id];
        int end   = host_csr_row_ptr[row_id + 1];
        float sum = static_cast<float>(0);
        for (int64_t j = start; j < end; j++) {
          float score = static_cast<float>(edge_weight_ptr[j * head_num + head_id]);
          int col_id  = host_csr_col_ptr[j];
          sum += score * feature_ptr[col_id * (head_num * emb_dim) + head_id * emb_dim + emb_id];
        }
        output_feature_ptr[row_id * (head_num * emb_dim) + head_id * emb_dim + emb_id] =
          (WeightType)sum;
      }
    }
  }
}
void host_gspmm_csr_weighted_forward(void* host_csr_row_ptr,
                                     wholememory_array_description_t csr_row_ptr_array_desc,
                                     void* host_csr_col_ptr,
                                     wholememory_array_description_t csr_col_ptr_array_desc,
                                     void* host_edge_weight_ptr,
                                     wholememory_tensor_description_t edge_weight_tensor_desc,
                                     void* host_feature_ptr,
                                     wholememory_tensor_description_t feature_tensor_desc,
                                     void* host_ref_output_feature_ptr,
                                     wholememory_tensor_description_t output_feature_tensor_desc)
{
  EXPECT_EQ(csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_col_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(edge_weight_tensor_desc.dim, 2);
  EXPECT_EQ(feature_tensor_desc.dim, 3);
  EXPECT_EQ(output_feature_tensor_desc.dim, 3);
  EXPECT_EQ(edge_weight_tensor_desc.dtype, feature_tensor_desc.dtype);
  EXPECT_EQ(feature_tensor_desc.dtype, output_feature_tensor_desc.dtype);
  EXPECT_EQ(csr_col_ptr_array_desc.size, edge_weight_tensor_desc.sizes[0]);
  EXPECT_EQ(feature_tensor_desc.sizes[1], edge_weight_tensor_desc.sizes[1]);
  EXPECT_EQ(feature_tensor_desc.sizes[1], output_feature_tensor_desc.sizes[1]);
  EXPECT_EQ(feature_tensor_desc.sizes[2], output_feature_tensor_desc.sizes[2]);

  if (edge_weight_tensor_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    host_get_gpsmm_csr_weighted_forward<float>(static_cast<int*>(host_csr_row_ptr),
                                               csr_row_ptr_array_desc,
                                               static_cast<int*>(host_csr_col_ptr),
                                               csr_col_ptr_array_desc,
                                               host_edge_weight_ptr,
                                               edge_weight_tensor_desc,
                                               host_feature_ptr,
                                               feature_tensor_desc,
                                               host_ref_output_feature_ptr,
                                               output_feature_tensor_desc);
  }
}

template <typename WeightType>
void host_get_gpsmm_csr_weighted_backward(
  int* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* host_edge_weight_ptr,
  wholememory_tensor_description_t edge_weight_tensor_desc,
  void* host_feature_ptr,
  wholememory_tensor_description_t feature_tensor_desc,
  void* host_input_grad_feature_ptr,
  wholememory_tensor_description_t input_grad_feature_tensor_desc,
  void* host_ref_output_grad_edge_weight_ptr,
  wholememory_tensor_description_t output_grad_edge_weight_tensor_desc,
  void* host_ref_output_grad_feature_ptr,
  wholememory_tensor_description_t output_grad_feature_tensor_desc)
{
  int64_t row_num                    = csr_row_ptr_array_desc.size - 1;
  int64_t head_num                   = edge_weight_tensor_desc.sizes[1];
  int64_t emb_dim                    = feature_tensor_desc.sizes[2];
  WeightType* edge_weight_ptr        = static_cast<WeightType*>(host_edge_weight_ptr);
  WeightType* feature_ptr            = static_cast<WeightType*>(host_feature_ptr);
  WeightType* input_grad_feature_ptr = static_cast<WeightType*>(host_input_grad_feature_ptr);
  WeightType* output_grad_edge_weight_ptr =
    static_cast<WeightType*>(host_ref_output_grad_edge_weight_ptr);
  WeightType* output_grad_feature_ptr = static_cast<WeightType*>(host_ref_output_grad_feature_ptr);

  if (host_ref_output_grad_feature_ptr) {
    for (int64_t head_id = 0; head_id < head_num; head_id++) {
      for (int64_t row_id = 0; row_id < row_num; row_id++) {
        int start = host_csr_row_ptr[row_id];
        int end   = host_csr_row_ptr[row_id + 1];
        for (int64_t emb_id = 0; emb_id < emb_dim; emb_id++) {
          for (int j = start; j < end; j++) {
            int col_id  = host_csr_col_ptr[j];
            float score = static_cast<float>(edge_weight_ptr[j * head_num + head_id]);
            output_grad_feature_ptr[col_id * head_num * emb_dim + head_id * emb_dim + emb_id] +=
              static_cast<WeightType>(
                score * static_cast<float>(input_grad_feature_ptr[row_id * head_num * emb_dim +
                                                                  head_id * emb_dim + emb_id]));
          }
        }
      }
    }
  }
  if (host_ref_output_grad_edge_weight_ptr) {
    for (int64_t head_id = 0; head_id < head_num; head_id++) {
      for (int64_t row_id = 0; row_id < row_num; row_id++) {
        int start = host_csr_row_ptr[row_id];
        int end   = host_csr_row_ptr[row_id + 1];
        for (int j = start; j < end; j++) {
          int col_id      = host_csr_col_ptr[j];
          float agg_value = 0;
          for (int emb_id = 0; emb_id < emb_dim; emb_id++) {
            agg_value +=
              static_cast<float>(
                input_grad_feature_ptr[row_id * head_num * emb_dim + head_id * emb_dim + emb_dim]) *
              static_cast<float>(
                feature_ptr[col_id * head_num * emb_dim + head_id * emb_dim + emb_dim]);
          }
          output_grad_edge_weight_ptr[j * head_num + head_id] = static_cast<WeightType>(agg_value);
        }
      }
    }
  }
}

void host_gspmm_csr_weighted_backward(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* host_edge_weight_ptr,
  wholememory_tensor_description_t edge_weight_tensor_desc,
  void* host_feature_ptr,
  wholememory_tensor_description_t feature_tensor_desc,
  void* host_input_grad_feature_ptr,
  wholememory_tensor_description_t input_grad_feature_tensor_desc,
  void* host_ref_output_grad_edge_weight_ptr,
  wholememory_tensor_description_t output_grad_edge_weight_tensor_desc,
  void* host_ref_output_grad_feature_ptr,
  wholememory_tensor_description_t output_grad_feature_tensor_desc)
{
  EXPECT_EQ(csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_col_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(edge_weight_tensor_desc.dim, 2);
  EXPECT_EQ(feature_tensor_desc.dim, 3);
  EXPECT_EQ(input_grad_feature_tensor_desc.dim, 3);
  EXPECT_EQ(edge_weight_tensor_desc.dtype, feature_tensor_desc.dtype);
  EXPECT_EQ(feature_tensor_desc.dtype, input_grad_feature_tensor_desc.dtype);
  EXPECT_EQ(csr_col_ptr_array_desc.size, edge_weight_tensor_desc.sizes[0]);
  EXPECT_EQ(feature_tensor_desc.sizes[1], edge_weight_tensor_desc.sizes[1]);
  EXPECT_EQ(feature_tensor_desc.sizes[1], input_grad_feature_tensor_desc.sizes[1]);
  EXPECT_EQ(feature_tensor_desc.sizes[2], input_grad_feature_tensor_desc.sizes[2]);
  if (host_ref_output_grad_edge_weight_ptr) {
    memset(host_ref_output_grad_edge_weight_ptr,
           0,
           wholememory_get_memory_size_from_tensor(&output_grad_edge_weight_tensor_desc));
  }
  if (host_ref_output_grad_feature_ptr) {
    memset(host_ref_output_grad_feature_ptr,
           0,
           wholememory_get_memory_size_from_tensor(&output_grad_feature_tensor_desc));
  }

  if (edge_weight_tensor_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    host_get_gpsmm_csr_weighted_backward<float>(static_cast<int*>(host_csr_row_ptr),
                                                csr_row_ptr_array_desc,
                                                static_cast<int*>(host_csr_col_ptr),
                                                csr_col_ptr_array_desc,
                                                host_edge_weight_ptr,
                                                edge_weight_tensor_desc,
                                                host_feature_ptr,
                                                feature_tensor_desc,
                                                host_input_grad_feature_ptr,
                                                input_grad_feature_tensor_desc,
                                                host_ref_output_grad_edge_weight_ptr,
                                                output_grad_edge_weight_tensor_desc,
                                                host_ref_output_grad_feature_ptr,
                                                output_grad_feature_tensor_desc);
  }
}
}  // namespace testing
}  // namespace graph_ops
