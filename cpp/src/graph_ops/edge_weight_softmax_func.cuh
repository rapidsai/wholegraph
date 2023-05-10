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
#pragma once

#include <assert.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include <raft/util/integer_utils.hpp>
#include <wholememory/tensor_description.h>

namespace graph_ops {

template <typename T = float>
__global__ void EdgeWeightSoftmaxForwardSimpleKernel(const int* csr_row_ptr,
                                                     const int64_t target_vec_count,
                                                     const T* edge_weight,
                                                     const int num_head,
                                                     T* edge_weight_softmax)
{
  int row_id  = blockIdx.x * blockDim.y + threadIdx.y;
  int head_id = threadIdx.x;
  if (row_id >= target_vec_count) return;

  int row_ptr_start = csr_row_ptr[row_id];
  int row_ptr_end   = csr_row_ptr[row_id + 1];
  float max_val     = -1e38f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value = (float)edge_weight[row_ptr_id * num_head + head_id];
    max_val     = max(max_val, value);
  }
  float total_exp_value = 0.0f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value = (float)edge_weight[row_ptr_id * num_head + head_id];
    total_exp_value += __expf(value - max_val);
  }

  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value = (float)edge_weight[row_ptr_id * num_head + head_id];
    value       = __expf(value - max_val);
    value /= total_exp_value;
    edge_weight_softmax[row_ptr_id * num_head + head_id] = (T)(value);
  }
}
template <typename WeightType>
void edge_weight_softmax_forward_func(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_matrix_description_t edge_weight_matrix_desc,
  void* output_edge_weight_ptr,
  wholememory_matrix_description_t output_edge_weight_matrix_desc,
  cudaStream_t stream)
{
  int target_vec_count = csr_row_ptr_array_desc.size - 1;
  int num_head         = edge_weight_matrix_desc.sizes[1];

  assert(num_head <= 512);
  int row_num_per_block = 512 / num_head;
  dim3 block(num_head, row_num_per_block);
  int block_count = raft::div_rounding_up_safe<int>(target_vec_count, row_num_per_block);
  EdgeWeightSoftmaxForwardSimpleKernel<WeightType>
    <<<block_count, block, 0, stream>>>(csr_row_ptr,
                                        target_vec_count,
                                        (const WeightType*)edge_weight_ptr,
                                        num_head,
                                        (WeightType*)output_edge_weight_ptr);
}

template <typename T = float>
__global__ void EdgeWeightSoftmaxBackwardSimpleKernel(const int* csr_row_ptr,
                                                      const T* edge_weight,
                                                      const T* grad_y,
                                                      const int64_t target_vec_count,
                                                      const int num_head,
                                                      T* grad_weight_softmax)
{
  int row_id  = blockIdx.x * blockDim.y + threadIdx.y;
  int head_id = threadIdx.x;
  if (row_id >= target_vec_count) return;

  int row_ptr_start = csr_row_ptr[row_id];
  int row_ptr_end   = csr_row_ptr[row_id + 1];

  float max_val = -1e38f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value = (float)edge_weight[row_ptr_id * num_head + head_id];
    max_val     = max(max_val, value);
  }
  float total_exp_value    = 0.0f;
  float total_exp_dy_value = 0.0f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float value     = (float)edge_weight[row_ptr_id * num_head + head_id];
    float value_exp = __expf(value - max_val);
    total_exp_value += value_exp;
    total_exp_dy_value += value_exp * (float)grad_y[row_ptr_id * num_head + head_id];
  }
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    float y =
      __expf((float)edge_weight[row_ptr_id * num_head + head_id] - max_val) / total_exp_value;
    float dy = (float)grad_y[row_ptr_id * num_head + head_id];
    float dx = y * (dy - total_exp_dy_value / total_exp_value);
    grad_weight_softmax[row_ptr_id * num_head + head_id] = dx;
  }
}

template <typename WeightType>
void edge_weight_softmax_backward_func(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_matrix_description_t edge_weight_matrix_desc,
  void* grad_edge_weight_softmax_ptr,
  wholememory_matrix_description_t grad_edge_weight_softmax_matrix_desc,
  void* output_grad_edge_weight_ptr,
  wholememory_matrix_description_t output_grad_edge_weight_matrix_desc,
  cudaStream_t stream)
{
  int target_vec_count = csr_row_ptr_array_desc.size - 1;
  int num_head         = edge_weight_matrix_desc.sizes[1];

  assert(num_head <= 512);
  int row_num_per_block = 512 / num_head;
  dim3 block(num_head, row_num_per_block);
  int block_count = raft::div_rounding_up_safe<int>(target_vec_count, row_num_per_block);

  EdgeWeightSoftmaxBackwardSimpleKernel<WeightType>
    <<<block_count, block, 0, stream>>>(csr_row_ptr,
                                        (const WeightType*)edge_weight_ptr,
                                        (const WeightType*)grad_edge_weight_softmax_ptr,
                                        target_vec_count,
                                        num_head,
                                        (WeightType*)output_grad_edge_weight_ptr);
}

}  // namespace graph_ops
