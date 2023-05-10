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
#include "edge_weight_softmax_test_utils.hpp"
#include <gtest/gtest.h>

namespace graph_ops {
namespace testing {

template <typename WeightType>
void host_get_edge_weight_softmax_forward(
  int* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* host_weight_ptr,
  wholememory_matrix_description_t weight_matrix_desc,
  void* host_ref_output_weight_ptr,
  wholememory_matrix_description_t output_weight_matrix_desc)
{
  int64_t row_num               = csr_row_ptr_array_desc.size - 1;
  int64_t head_num              = weight_matrix_desc.sizes[1];
  WeightType* weight_ptr        = static_cast<WeightType*>(host_weight_ptr);
  WeightType* output_weight_ptr = static_cast<WeightType*>(host_ref_output_weight_ptr);
  for (int64_t row_id = 0; row_id < row_num; row_id++) {
    int start = host_csr_row_ptr[row_id];
    int end   = host_csr_row_ptr[row_id + 1];
    for (int64_t head_id = 0; head_id < head_num; head_id++) {
      float max_val = -1e38f;
      for (int k = start; k < end; k++) {
        float value = (float)weight_ptr[k * head_num + head_id];
        max_val     = std::max(max_val, value);
      }
      float total_exp_value = 0.0f;
      for (int k = start; k < end; k++) {
        float value = (float)weight_ptr[k * head_num + head_id];
        total_exp_value += expf(value - max_val);
      }
      for (int k = start; k < end; k++) {
        float value = expf((float)weight_ptr[k * head_num + head_id] - max_val);
        value /= total_exp_value;
        output_weight_ptr[k * head_num + head_id] = static_cast<WeightType>(value);
      }
    }
  }
}

void host_edge_weight_softmax_forward(void* host_csr_row_ptr,
                                      wholememory_array_description_t csr_row_ptr_array_desc,
                                      void* host_weight_ptr,
                                      wholememory_matrix_description_t weight_matrix_desc,
                                      void* host_ref_output_weight_ptr,
                                      wholememory_matrix_description_t output_weight_matrix_desc)
{
  EXPECT_EQ(csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(weight_matrix_desc.sizes[0], output_weight_matrix_desc.sizes[0]);
  EXPECT_EQ(weight_matrix_desc.sizes[1], output_weight_matrix_desc.sizes[1]);
  EXPECT_EQ(weight_matrix_desc.dtype, output_weight_matrix_desc.dtype);
  if (weight_matrix_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    host_get_edge_weight_softmax_forward<float>(static_cast<int*>(host_csr_row_ptr),
                                                csr_row_ptr_array_desc,
                                                host_weight_ptr,
                                                weight_matrix_desc,
                                                host_ref_output_weight_ptr,
                                                output_weight_matrix_desc);
  }
}

template <typename WeightType>
void host_get_edge_weight_softmax_backward(
  int* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* host_weight_ptr,
  wholememory_matrix_description_t weight_matrix_desc,
  void* host_grad_weight_softmax_ptr,
  wholememory_matrix_description_t grad_weight_softmax_matrix_desc,
  void* host_ref_output_grad_weight_ptr,
  wholememory_matrix_description_t output_grad_weight_matrix_desc)
{
  int64_t row_num                     = csr_row_ptr_array_desc.size - 1;
  int64_t head_num                    = weight_matrix_desc.sizes[1];
  WeightType* weight_ptr              = static_cast<WeightType*>(host_weight_ptr);
  WeightType* grad_weight_softmax_ptr = static_cast<WeightType*>(host_grad_weight_softmax_ptr);
  WeightType* output_grad_weight_ptr  = static_cast<WeightType*>(host_ref_output_grad_weight_ptr);

  for (int64_t row_id = 0; row_id < row_num; row_id++) {
    int start = host_csr_row_ptr[row_id];
    int end   = host_csr_row_ptr[row_id + 1];
    for (int64_t head_id = 0; head_id < head_num; head_id++) {
      float max_val = -1e38f;
      for (int k = start; k < end; k++) {
        float value = (float)weight_ptr[k * head_num + head_id];
        max_val     = std::max(max_val, value);
      }
      float total_exp_value    = 0.0f;
      float total_exp_dy_value = 0.0f;
      for (int k = start; k < end; k++) {
        float value     = (float)weight_ptr[k * head_num + head_id];
        float value_exp = expf(value - max_val);
        total_exp_value += value_exp;
        total_exp_dy_value += value_exp * (float)grad_weight_softmax_ptr[k * head_num + head_id];
      }
      for (int k = start; k < end; k++) {
        float y  = expf((float)weight_ptr[k * head_num + head_id] - max_val) / total_exp_value;
        float dy = (float)grad_weight_softmax_ptr[k * head_num + head_id];
        float dx = y * (dy - total_exp_dy_value / total_exp_value);
        output_grad_weight_ptr[k * head_num + head_id] = static_cast<WeightType>(dx);
      }
    }
  }
}
void host_edge_weight_softmax_backward(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* host_weight_ptr,
  wholememory_matrix_description_t weight_matrix_desc,
  void* host_grad_weight_softmax_ptr,
  wholememory_matrix_description_t grad_weight_softmax_matrix_desc,
  void* host_ref_output_grad_weight_ptr,
  wholememory_matrix_description_t output_grad_weight_matrix_desc)
{
  EXPECT_EQ(csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(weight_matrix_desc.sizes[0], output_grad_weight_matrix_desc.sizes[0]);
  EXPECT_EQ(weight_matrix_desc.sizes[1], output_grad_weight_matrix_desc.sizes[1]);
  EXPECT_EQ(weight_matrix_desc.dtype, output_grad_weight_matrix_desc.dtype);

  EXPECT_EQ(weight_matrix_desc.sizes[0], grad_weight_softmax_matrix_desc.sizes[0]);
  EXPECT_EQ(weight_matrix_desc.sizes[1], grad_weight_softmax_matrix_desc.sizes[1]);
  EXPECT_EQ(weight_matrix_desc.dtype, grad_weight_softmax_matrix_desc.dtype);

  if (weight_matrix_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    host_get_edge_weight_softmax_backward<float>(static_cast<int*>(host_csr_row_ptr),
                                                 csr_row_ptr_array_desc,
                                                 host_weight_ptr,
                                                 weight_matrix_desc,
                                                 host_grad_weight_softmax_ptr,
                                                 grad_weight_softmax_matrix_desc,
                                                 host_ref_output_grad_weight_ptr,
                                                 output_grad_weight_matrix_desc);
  }
}

}  // namespace testing
}  // namespace graph_ops
