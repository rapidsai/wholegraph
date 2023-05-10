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
__global__ void SpAddGATCSRForwardSimpleKernel(const int* csr_row_ptr,
                                               int target_vec_count,
                                               const int* csr_col_ind,
                                               const T* edge_weight_left,
                                               const T* edge_weight_right,
                                               int num_head,
                                               T* output)
{
  int row_id  = blockIdx.x * blockDim.y + threadIdx.y;
  int head_id = threadIdx.x;
  if (row_id >= target_vec_count) return;

  int row_ptr_start = csr_row_ptr[row_id];
  int row_ptr_end   = csr_row_ptr[row_id + 1];

  T left_weight = edge_weight_left[row_id * num_head + head_id];
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    int col_id                              = csr_col_ind[row_ptr_id];
    T right_weight                          = edge_weight_right[col_id * num_head + head_id];
    T value                                 = left_weight + right_weight;
    output[row_ptr_id * num_head + head_id] = value;
  }
}

template <typename WeightType>
void spadd_gat_csr_forward_func(int* csr_row_ptr,
                                wholememory_array_description_t csr_row_ptr_array_desc,
                                int* csr_col_ptr,
                                wholememory_array_description_t csr_col_ptr_array_desc,
                                void* edge_weight_left_ptr,
                                wholememory_matrix_description_t edge_weight_left_matrix_desc,
                                void* edge_weight_right_ptr,
                                wholememory_matrix_description_t edge_weight_right_matrix_desc,
                                void* output_score_ptr,
                                wholememory_matrix_description_t output_score_matrix_desc,
                                cudaStream_t stream)
{
  int target_node_count = edge_weight_left_matrix_desc.sizes[0];
  int num_head          = edge_weight_left_matrix_desc.sizes[1];
  assert(num_head <= 512);
  int row_num_per_block = 512 / num_head;
  dim3 block(num_head, row_num_per_block);
  int block_count = (target_node_count + row_num_per_block - 1) / row_num_per_block;
  SpAddGATCSRForwardSimpleKernel<WeightType>
    <<<block_count, block, 0, stream>>>(csr_row_ptr,
                                        target_node_count,
                                        csr_col_ptr,
                                        (const WeightType*)edge_weight_left_ptr,
                                        (const WeightType*)edge_weight_right_ptr,
                                        num_head,
                                        (WeightType*)output_score_ptr);
}

template <typename T = float>
__global__ void SpAddGATCSRBackwardSimpleKernel(const int* csr_row_ptr,
                                                const int* csr_col_ind,
                                                const T* grad_y,
                                                int target_vec_count,
                                                int num_head,
                                                T* grad_weight_left,
                                                T* grad_weight_right)
{
  int row_id  = blockIdx.x * blockDim.y + threadIdx.y;
  int head_id = threadIdx.x;
  if (row_id >= target_vec_count) return;

  int row_ptr_start        = csr_row_ptr[row_id];
  int row_ptr_end          = csr_row_ptr[row_id + 1];
  T grad_weight_left_value = (T)0.0f;
  for (int row_ptr_id = row_ptr_start; row_ptr_id < row_ptr_end; row_ptr_id++) {
    T grad_y_value = grad_y[row_ptr_id * num_head + head_id];
    grad_weight_left_value += grad_y_value;
    int col_id = csr_col_ind[row_ptr_id];
    atomicAdd(grad_weight_right + col_id * num_head + head_id, (T)grad_y_value);
  }
  grad_weight_left[row_id * num_head + head_id] = grad_weight_left_value;
}

template <typename WeightType>
void spadd_gat_csr_backward_func(int* csr_row_ptr,
                                 wholememory_array_description_t csr_row_ptr_array_desc,
                                 int* csr_col_ptr,
                                 wholememory_array_description_t csr_col_ptr_array_desc,
                                 void* grad_score_ptr,
                                 wholememory_matrix_description_t grad_score_matrix_desc,
                                 void* output_grad_edge_weight_left_ptr,
                                 wholememory_matrix_description_t edge_weight_left_matrix_desc,
                                 void* output_grad_edge_weight_right_ptr,
                                 wholememory_matrix_description_t edge_weight_right_matrix_desc,
                                 cudaStream_t stream)
{
  int target_node_count = edge_weight_left_matrix_desc.sizes[0];
  // int neighbor_node_count = edge_weight_right_matrix_desc.sizes[0];
  int num_head = edge_weight_left_matrix_desc.sizes[1];
  WM_CUDA_CHECK(
    cudaMemsetAsync(output_grad_edge_weight_right_ptr,
                    0,
                    wholememory_get_memory_size_from_matrix(&edge_weight_right_matrix_desc),
                    stream));

  assert(num_head <= 512);
  int rows_per_block = 512 / num_head;
  dim3 block(num_head, rows_per_block);
  int block_count = (target_node_count + rows_per_block - 1) / rows_per_block;
  SpAddGATCSRBackwardSimpleKernel<WeightType>
    <<<block_count, block, 0, stream>>>(csr_row_ptr,
                                        csr_col_ptr,
                                        (const WeightType*)grad_score_ptr,
                                        target_node_count,
                                        num_head,
                                        (WeightType*)output_grad_edge_weight_left_ptr,
                                        (WeightType*)output_grad_edge_weight_right_ptr);
}

}  // namespace graph_ops
