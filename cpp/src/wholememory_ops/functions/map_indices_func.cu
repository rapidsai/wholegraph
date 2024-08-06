/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "map_indices_func.h"
#include "wholememory/integer_utils.hpp"
#include "wholememory_ops/register.hpp"
#include <cuda_runtime.h>

namespace wholememory_ops {

template <typename IndexT>
__global__ void storage_idx2wm_emb_idx_kernel(IndexT* indice,
                                              IndexT* mapped_indice,
                                              int64_t indice_size,
                                              int world_size,
                                              int64_t entry_start,
                                              int round_robin_size)
{
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_t i = tid; i < indice_size; i += (blockDim.x * gridDim.x)) {
    IndexT target_idx  = indice[i];
    IndexT table_idx   = target_idx / round_robin_size;
    IndexT table_off   = target_idx % round_robin_size;
    int rank_id        = table_idx % world_size;
    int rank_table_idx = table_idx / world_size;
    IndexT wmidx       = entry_start + round_robin_size * rank_table_idx + table_off;
    mapped_indice[i]   = wmidx;
  }
  return;
}

template <typename IndexT>
void storage_idx2wm_emb_idx_temp_fn(void* indice_ptr,
                                    void* mapped_indice_ptr,
                                    int64_t indice_size,
                                    int world_size,
                                    int64_t entry_start,
                                    int round_robin_size,
                                    cudaStream_t stream)
{
  int block_size    = 256;
  int64_t block_num = (indice_size + block_size - 1) / block_size;
  if (block_num > 1568) block_num = 1568;
  IndexT* indice        = static_cast<IndexT*>(indice_ptr);
  IndexT* mapped_indice = static_cast<IndexT*>(mapped_indice_ptr);
  storage_idx2wm_emb_idx_kernel<<<block_num, block_size, 0, stream>>>(
    indice, mapped_indice, indice_size, world_size, entry_start, round_robin_size);
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  return;
}

REGISTER_DISPATCH_ONE_TYPE(storageidx2wmembidx, storage_idx2wm_emb_idx_temp_fn, SINT3264)

wholememory_error_code_t storage_index2wm_embedding_index(wholememory_tensor_t indices,
                                                          wholememory_tensor_t mapped_indices,
                                                          wholememory_tensor_t allocated_embedding,
                                                          int round_robin_size,
                                                          int64_t stream_int)
{
  if (round_robin_size == 0) return WHOLEMEMORY_SUCCESS;
  try {
    auto* indice_desc       = wholememory_tensor_get_tensor_description(indices);
    void* indice_ptr        = wholememory_tensor_get_data_pointer(indices);
    void* mapped_indice_ptr = wholememory_tensor_get_data_pointer(mapped_indices);
    int64_t indice_size     = indice_desc->sizes[0];

    wholememory_comm_t wm_comm;
    int world_size;
    auto* handle = wholememory_tensor_get_memory_handle(allocated_embedding);
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, handle));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));

    size_t entry_start = 0;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_tensor_get_local_entry_start(&entry_start, allocated_embedding));
    DISPATCH_ONE_TYPE(indice_desc->dtype,
                      storageidx2wmembidx,
                      indice_ptr,
                      mapped_indice_ptr,
                      indice_size,
                      world_size,
                      entry_start,
                      round_robin_size,
                      (cudaStream_t)stream_int);
    WM_CUDA_CHECK(cudaGetLastError());
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("index map CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
