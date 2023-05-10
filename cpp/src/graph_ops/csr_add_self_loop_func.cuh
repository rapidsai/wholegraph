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
#include "cuda_macros.hpp"
#include "error.hpp"
#include <raft/util/integer_utils.hpp>
#include <wholememory/tensor_description.h>

namespace graph_ops {

__global__ void AddSelfLoopKernel(const int* csr_row_ptr,
                                  const int* csr_col_ind,
                                  int* csr_row_ptr_looped,
                                  int* csr_col_ind_looped)
{
  int row_idx       = blockIdx.x;
  int row_ptr_start = csr_row_ptr[row_idx];
  int row_ptr_end   = csr_row_ptr[row_idx + 1];
  if (threadIdx.x == 0) {
    csr_row_ptr_looped[row_idx] = row_ptr_start + row_idx;
    if (blockIdx.x == gridDim.x - 1) {
      csr_row_ptr_looped[row_idx + 1] = row_ptr_end + row_idx + 1;
    }
  }
  for (int nidx = threadIdx.x; nidx <= row_ptr_end - row_ptr_start; nidx += blockDim.x) {
    int neighbor_idx = row_idx;
    if (nidx > 0) { neighbor_idx = csr_col_ind[row_ptr_start + nidx - 1]; }
    csr_col_ind_looped[row_ptr_start + row_idx + nidx] = neighbor_idx;
  }
}

void csr_add_self_loop_func(int* csr_row_ptr,
                            wholememory_array_description_t csr_row_ptr_array_desc,
                            int* csr_col_ptr,
                            wholememory_array_description_t csr_col_ptr_array_desc,
                            int* output_csr_row_ptr,
                            wholememory_array_description_t output_csr_row_ptr_array_desc,
                            int* output_csr_col_ptr,
                            wholememory_array_description_t output_csr_col_ptr_array_desc,
                            cudaStream_t stream)
{
  int target_count = csr_row_ptr_array_desc.size - 1;
  AddSelfLoopKernel<<<target_count, 64, 0, stream>>>(
    csr_row_ptr, csr_col_ptr, output_csr_row_ptr, output_csr_col_ptr);
}

}  // namespace graph_ops
