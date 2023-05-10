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
__global__ void gSpmmCsrWeightedForwardKernel(const int* csr_row_ptr,
                                              const int* csr_col_ind,
                                              const T* edge_weight,
                                              int num_head,
                                              const T* x,
                                              int embedding_dim,
                                              int ldb,
                                              int ldb_head,
                                              int input_count,
                                              T* output,
                                              int ldc,
                                              int ldc_head)
{
  // ldb_head may be ldb * num_head
  int row_idx       = blockIdx.x;
  int head_idx      = blockIdx.y;
  int emb_idx       = threadIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end   = csr_row_ptr[row_idx + 1];
  for (; emb_idx < embedding_dim; emb_idx += blockDim.x) {
    float agg_value = 0.0f;
    for (int row_ptr = row_ptr_start; row_ptr < row_ptr_end; row_ptr++) {
      float ew    = (float)edge_weight[row_ptr * num_head + head_idx];
      int col_idx = csr_col_ind[row_ptr];
      assert(col_idx >= 0 && col_idx < input_count);
      float value = (float)x[(int64_t)col_idx * ldb_head + head_idx * ldb + emb_idx];
      agg_value += value * ew;
    }
    output[(int64_t)row_idx * ldc_head + head_idx * ldc + emb_idx] = (T)agg_value;
  }
}

template <typename WeightType>
void gspmm_csr_weighted_forward_func(int* csr_row_ptr,
                                     wholememory_array_description_t csr_row_ptr_array_desc,
                                     int* csr_col_ptr,
                                     wholememory_array_description_t csr_col_ptr_array_desc,
                                     void* edge_weight_ptr,
                                     wholememory_tensor_description_t edge_weight_tensor_desc,
                                     void* feature_ptr,
                                     wholememory_tensor_description_t feature_tensor_desc,
                                     void* output_feature_ptr,
                                     wholememory_tensor_description_t output_feature_tensor_desc,
                                     cudaStream_t stream)
{
  int m             = csr_row_ptr_array_desc.size - 1;
  int num_head      = feature_tensor_desc.sizes[1];
  int embedding_dim = feature_tensor_desc.sizes[2];
  int ldb           = feature_tensor_desc.strides[1];
  int ldb_head      = feature_tensor_desc.strides[0];
  int k             = feature_tensor_desc.sizes[0];
  int ldc           = output_feature_tensor_desc.strides[1];
  int ldc_head      = output_feature_tensor_desc.strides[0];

  int block_count  = m;
  int thread_count = embedding_dim;
  if (thread_count > 512) { thread_count = 512; }
  dim3 block(block_count, num_head);

  gSpmmCsrWeightedForwardKernel<WeightType>
    <<<block, thread_count, 0, stream>>>(csr_row_ptr,
                                         csr_col_ptr,
                                         (const WeightType*)edge_weight_ptr,
                                         num_head,
                                         (const WeightType*)feature_ptr,
                                         embedding_dim,
                                         ldb,
                                         ldb_head,
                                         k,
                                         (WeightType*)output_feature_ptr,
                                         ldc,
                                         ldc_head);
}

template <typename T = float, bool NeedBackwardX = true, bool NeedBackwardW = true>
__global__ void gSpmmCsrWeightedFusedSharedMemoryBackwardKernel(const int* csr_row_ptr,
                                                                int target_count,
                                                                const int* csr_col_ind,
                                                                int edge_count,
                                                                const T* edge_weight,
                                                                const T* x,
                                                                int embedding_dim,
                                                                int num_head,
                                                                int ldx,
                                                                int ldx_head,
                                                                int neighbor_count,
                                                                const T* output_grad,
                                                                int ld_grad_out,
                                                                int ld_grad_out_head,
                                                                T* x_grad,
                                                                int ld_grad_x,
                                                                int ld_grad_x_head,
                                                                T* edge_weight_grad)
{
  int row_idx       = blockIdx.x;  // target index
  int head_idx      = blockIdx.y;
  int emb_idx_start = threadIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end   = csr_row_ptr[row_idx + 1];
  assert(row_ptr_start >= 0 && row_ptr_start <= edge_count);
  assert(row_ptr_end >= 0 && row_ptr_end <= edge_count);
  assert(row_ptr_end >= row_ptr_start);
  extern __shared__ float w_grad_data[];
  for (int row_ptr = row_ptr_start; row_ptr < row_ptr_end; row_ptr++) {
    int col_idx = csr_col_ind[row_ptr];
    assert(col_idx >= 0 && col_idx < neighbor_count);
    float edge_weight_value      = (float)edge_weight[row_ptr * num_head + head_idx];
    float edge_weight_grad_value = 0.0f;
    for (int emb_idx = emb_idx_start; emb_idx < embedding_dim; emb_idx += blockDim.x) {
      float grad_out_value = (float)output_grad[(int64_t)row_idx * ld_grad_out_head +
                                                (int64_t)head_idx * ld_grad_out + emb_idx];
      float x_value = (float)x[(int64_t)col_idx * ldx_head + (int64_t)head_idx * ldx + emb_idx];
      // update x_grad here.
      float x_grad_value = edge_weight_value * grad_out_value;
      if (NeedBackwardX) {
        atomicAdd(
          x_grad + (int64_t)col_idx * ld_grad_x_head + (int64_t)head_idx * ld_grad_x + emb_idx,
          (T)x_grad_value);
      }
      edge_weight_grad_value += x_value * grad_out_value;
    }
    if (NeedBackwardW) {
      if (embedding_dim > 32) {
        w_grad_data[threadIdx.x] = edge_weight_grad_value;
        __syncthreads();
      }
      if (threadIdx.x < 32) {
        for (int widx = threadIdx.x + 32; widx < embedding_dim && widx < blockDim.x; widx += 32) {
          edge_weight_grad_value += w_grad_data[widx];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
          edge_weight_grad_value +=
            __shfl_down_sync(0xffffffff, edge_weight_grad_value, offset, 32);
        }
        if (threadIdx.x == 0) {
          edge_weight_grad[row_ptr * num_head + head_idx] = (T)edge_weight_grad_value;
        }
      }
    }
  }
}

template <typename T = float, bool NeedBackwardX = true, bool NeedBackwardW = true>
void gSpmmCsrWeightedFusedSharedMemoryBackwardFunc(const int* csr_row_ptr,
                                                   int target_count,
                                                   const int* csr_col_ind,
                                                   int edge_count,
                                                   const T* edge_weight,
                                                   const T* x,
                                                   int embedding_dim,
                                                   int num_head,
                                                   int ldx,
                                                   int ldx_head,
                                                   int neighbor_count,
                                                   const T* output_grad,
                                                   int ld_grad_out,
                                                   int ld_grad_out_head,
                                                   T* x_grad,
                                                   int ld_grad_x,
                                                   int ld_grad_x_head,
                                                   T* edge_weight_grad,
                                                   cudaStream_t stream)
{
  if (NeedBackwardX) {
    cudaMemsetAsync(x_grad, 0, neighbor_count * embedding_dim * num_head * sizeof(T), stream);
  }

  if (NeedBackwardW) {
    cudaMemsetAsync(edge_weight_grad, 0, edge_count * num_head * sizeof(T), stream);
  }
  dim3 block(target_count, num_head);
  int thread_count = std::min<int>(512, embedding_dim);
  thread_count     = std::max<int>(32, thread_count);  // at least one warp
  gSpmmCsrWeightedFusedSharedMemoryBackwardKernel<T, NeedBackwardX, NeedBackwardW>
    <<<block, thread_count, thread_count * sizeof(float), stream>>>(csr_row_ptr,
                                                                    target_count,
                                                                    csr_col_ind,
                                                                    edge_count,
                                                                    edge_weight,
                                                                    x,
                                                                    embedding_dim,
                                                                    num_head,
                                                                    ldx,
                                                                    ldx_head,
                                                                    neighbor_count,
                                                                    output_grad,
                                                                    ld_grad_out,
                                                                    ld_grad_out_head,
                                                                    x_grad,
                                                                    ld_grad_x,
                                                                    ld_grad_x_head,
                                                                    edge_weight_grad);
}

template <typename WeightType>
void gspmm_csr_weighted_backward_func(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_tensor_description_t edge_weight_tensor_desc,
  void* feature_ptr,
  wholememory_tensor_description_t feature_tensor_desc,
  void* grad_feature_ptr,
  wholememory_tensor_description_t grad_feature_tensor_desc,
  void* output_grad_edge_weight_ptr,
  wholememory_tensor_description_t output_grad_edge_weight_tensor_desc,
  void* output_grad_feature_ptr,
  wholememory_tensor_description_t output_grad_feature_tensor_desc,
  cudaStream_t stream)
{
  int target_count     = csr_row_ptr_array_desc.size - 1;
  int edge_count       = csr_col_ptr_array_desc.size;
  int num_head         = feature_tensor_desc.sizes[1];
  int embedding_dim    = feature_tensor_desc.sizes[2];
  int ldx              = feature_tensor_desc.strides[1];
  int ldx_head         = feature_tensor_desc.strides[0];
  int neighbor_count   = feature_tensor_desc.sizes[0];
  int ld_grad_out      = grad_feature_tensor_desc.strides[1];
  int ld_grad_out_head = grad_feature_tensor_desc.strides[0];
  int ld_grad_x        = embedding_dim;
  int ld_grad_x_head   = embedding_dim * num_head;

  auto fn = gSpmmCsrWeightedFusedSharedMemoryBackwardFunc<WeightType, true, true>;

  if (output_grad_feature_ptr) {
    if (!output_grad_edge_weight_ptr) {
      fn = gSpmmCsrWeightedFusedSharedMemoryBackwardFunc<WeightType, true, false>;
    }
  } else {
    if (output_grad_edge_weight_ptr) {
      fn = gSpmmCsrWeightedFusedSharedMemoryBackwardFunc<WeightType, false, true>;
    } else {
      // fn = nullptr;
      return;
    }
  }
  fn(csr_row_ptr,
     target_count,
     csr_col_ptr,
     edge_count,
     (const WeightType*)edge_weight_ptr,
     (const WeightType*)feature_ptr,
     embedding_dim,
     num_head,
     ldx,
     ldx_head,
     neighbor_count,
     (const WeightType*)grad_feature_ptr,
     ld_grad_out,
     ld_grad_out_head,
     (WeightType*)output_grad_feature_ptr,
     ld_grad_x,
     ld_grad_x_head,
     (WeightType*)output_grad_edge_weight_ptr,
     stream);
}
}  // namespace graph_ops
