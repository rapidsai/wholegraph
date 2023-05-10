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
#include "spadd_gat_csr_utils.hpp"
#include <gtest/gtest.h>

namespace graph_ops {
namespace testing {

template <typename WeightType>
void host_get_spadd_gat_csr_forward(int* host_csr_row_ptr,
                                    wholememory_array_description_t csr_row_ptr_array_desc,
                                    int* host_csr_col_ptr,
                                    wholememory_array_description_t csr_col_ptr_array_desc,
                                    void* host_weight_left_ptr,
                                    wholememory_matrix_description_t weight_left_matrix_desc,
                                    void* host_weight_right_ptr,
                                    wholememory_matrix_description_t weight_right_matrix_desc,
                                    void* host_output_score_ptr,
                                    wholememory_matrix_description_t output_score_matrix_desc)
{
  int64_t row_num = csr_row_ptr_array_desc.size - 1;
  // int64_t col_num = weight_right_matrix_desc.sizes[0];
  // int64_t edge_num = csr_col_ptr_array_desc.size;
  int64_t head_num             = weight_left_matrix_desc.sizes[1];
  WeightType* weight_left_ptr  = static_cast<WeightType*>(host_weight_left_ptr);
  WeightType* weight_right_ptr = static_cast<WeightType*>(host_weight_right_ptr);
  WeightType* output_score_ptr = static_cast<WeightType*>(host_output_score_ptr);
  for (int i = 0; i < row_num; i++) {
    int start = host_csr_row_ptr[i];
    int end   = host_csr_row_ptr[i + 1];
    for (int j = start; j < end; j++) {
      int col_id = host_csr_col_ptr[j];
      for (int k = 0; k < head_num; k++) {
        output_score_ptr[j * head_num + k] =
          weight_left_ptr[i * head_num + k] + weight_right_ptr[col_id * head_num + k];
      }
    }
  }
}

void host_spadd_gat_csr_forward(void* host_csr_row_ptr,
                                wholememory_array_description_t csr_row_ptr_array_desc,
                                void* host_csr_col_ptr,
                                wholememory_array_description_t csr_col_ptr_array_desc,
                                void* host_weight_left_ptr,
                                wholememory_matrix_description_t weight_left_matrix_desc,
                                void* host_weight_right_ptr,
                                wholememory_matrix_description_t weight_right_matrix_desc,
                                void* host_output_score_ptr,
                                wholememory_matrix_description_t output_score_matrix_desc)
{
  EXPECT_EQ(csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_col_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(weight_left_matrix_desc.sizes[1], weight_right_matrix_desc.sizes[1]);
  EXPECT_EQ(weight_left_matrix_desc.sizes[0], csr_row_ptr_array_desc.size - 1);
  EXPECT_EQ(weight_left_matrix_desc.dtype, weight_right_matrix_desc.dtype);
  EXPECT_EQ(weight_left_matrix_desc.dtype, output_score_matrix_desc.dtype);

  if (weight_left_matrix_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    host_get_spadd_gat_csr_forward<float>(static_cast<int*>(host_csr_row_ptr),
                                          csr_row_ptr_array_desc,
                                          static_cast<int*>(host_csr_col_ptr),
                                          csr_col_ptr_array_desc,
                                          host_weight_left_ptr,
                                          weight_left_matrix_desc,
                                          host_weight_right_ptr,
                                          weight_right_matrix_desc,
                                          host_output_score_ptr,
                                          output_score_matrix_desc);
  }
}

template <typename WeightType>
void host_get_spadd_gat_csr_backward(int* host_csr_row_ptr,
                                     wholememory_array_description_t csr_row_ptr_array_desc,
                                     int* host_csr_col_ptr,
                                     wholememory_array_description_t csr_col_ptr_array_desc,
                                     void* host_grad_score_ptr,
                                     wholememory_matrix_description_t grad_score_matrix_desc,
                                     void* host_ref_output_grad_weight_left_ptr,
                                     wholememory_matrix_description_t weight_left_matrix_desc,
                                     void* host_ref_output_grad_weight_right_ptr,
                                     wholememory_matrix_description_t weight_right_matrix_desc)
{
  int64_t row_num  = csr_row_ptr_array_desc.size - 1;
  int64_t head_num = weight_left_matrix_desc.sizes[1];
  WeightType* output_weight_left_ptr =
    static_cast<WeightType*>(host_ref_output_grad_weight_left_ptr);
  WeightType* output_weight_right_ptr =
    static_cast<WeightType*>(host_ref_output_grad_weight_right_ptr);
  WeightType* grad_score_ptr = static_cast<WeightType*>(host_grad_score_ptr);

  for (int64_t row_id = 0; row_id < row_num; row_id++) {
    int start = host_csr_row_ptr[row_id];
    int end   = host_csr_row_ptr[row_id + 1];
    for (int64_t head_id = 0; head_id < head_num; head_id++) {
      WeightType left_sum_value = (WeightType)0;
      for (int j = start; j < end; j++) {
        int col_id = host_csr_col_ptr[j];
        left_sum_value += grad_score_ptr[j * head_num + head_id];
        output_weight_right_ptr[col_id * head_num + head_id] +=
          grad_score_ptr[j * head_num + head_id];
      }
      output_weight_left_ptr[row_id * head_num + head_id] = left_sum_value;
    }
  }
}
void host_spadd_gat_csr_backward(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* host_grad_score_ptr,
  wholememory_matrix_description_t grad_score_matrix_desc,
  void* host_ref_output_grad_weight_left_ptr,
  wholememory_matrix_description_t output_grad_weight_left_matrix_desc,
  void* host_ref_output_grad_weight_right_ptr,
  wholememory_matrix_description_t output_grad_weight_right_matrix_desc)
{
  EXPECT_EQ(csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_col_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(output_grad_weight_left_matrix_desc.sizes[1],
            output_grad_weight_right_matrix_desc.sizes[1]);
  EXPECT_EQ(output_grad_weight_left_matrix_desc.sizes[0], csr_row_ptr_array_desc.size - 1);
  EXPECT_EQ(output_grad_weight_left_matrix_desc.dtype, output_grad_weight_right_matrix_desc.dtype);
  EXPECT_EQ(output_grad_weight_left_matrix_desc.dtype, grad_score_matrix_desc.dtype);
  memset(host_ref_output_grad_weight_right_ptr,
         0,
         wholememory_get_memory_size_from_matrix(&output_grad_weight_right_matrix_desc));

  if (output_grad_weight_left_matrix_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    host_get_spadd_gat_csr_backward<float>(static_cast<int*>(host_csr_row_ptr),
                                           csr_row_ptr_array_desc,
                                           static_cast<int*>(host_csr_col_ptr),
                                           csr_col_ptr_array_desc,
                                           host_grad_score_ptr,
                                           grad_score_matrix_desc,
                                           host_ref_output_grad_weight_left_ptr,
                                           output_grad_weight_left_matrix_desc,
                                           host_ref_output_grad_weight_right_ptr,
                                           output_grad_weight_right_matrix_desc);
  }
}

}  // namespace testing
}  // namespace graph_ops
