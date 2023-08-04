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
#include "bucket_ids_func.h"

#include <cassert>
#include <cstdint>

#include <wholememory/wholememory.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "wholememory/integer_utils.hpp"
#include "wholememory_ops/register.hpp"

namespace wholememory_ops {

template <typename IndexT>
__global__ void bucket_ids_for_ranks_kernel(const IndexT* indices,
                                            size_t indice_count,
                                            int64_t* dev_rank_id_count_ptr,
                                            size_t embedding_entry_count_per_rank,
                                            int world_size)
{
  extern __shared__ int rank_count_shared[];
  for (int idx = threadIdx.x; idx < world_size; idx += blockDim.x) {
    rank_count_shared[idx] = 0;
  }
  __syncthreads();
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indice_count;
       idx += blockDim.x * gridDim.x) {
    IndexT node_idx = indices[idx];
    if (node_idx < 0) continue;
    int rank = node_idx / embedding_entry_count_per_rank;
    assert(rank >= 0 && rank < world_size);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    atomicAdd_block(&rank_count_shared[rank], 1);
#else
    atomicAdd(&rank_count_shared[rank], 1);
#endif
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < world_size; idx += blockDim.x) {
    atomicAdd(reinterpret_cast<unsigned long long*>(dev_rank_id_count_ptr) + idx,
              static_cast<unsigned long long>(rank_count_shared[idx]));
  }
}

template <typename IndexT>
void bucket_ids_for_ranks_temp_fn(void* indices,
                                  wholememory_array_description_t indice_desc,
                                  int64_t* dev_rank_id_count_ptr,
                                  size_t embedding_entry_count_per_rank,
                                  int world_size,
                                  int sm_count,
                                  cudaStream_t stream)
{
  static constexpr int BLOCK_SIZE = 128;
  int block_count     = wholememory::div_rounding_up_unsafe(indice_desc.size, BLOCK_SIZE);
  block_count         = std::min(block_count, sm_count * 4);
  IndexT* indices_ptr = static_cast<IndexT*>(indices);
  indices_ptr += indice_desc.storage_offset;
  bucket_ids_for_ranks_kernel<<<block_count, BLOCK_SIZE, sizeof(int) * world_size, stream>>>(
    indices_ptr,
    indice_desc.size,
    dev_rank_id_count_ptr,
    embedding_entry_count_per_rank,
    world_size);
}

REGISTER_DISPATCH_ONE_TYPE(BucketIdForRanks, bucket_ids_for_ranks_temp_fn, SINT3264)

wholememory_error_code_t bucket_ids_for_ranks(void* indices,
                                              wholememory_array_description_t indice_desc,
                                              int64_t* dev_rank_id_count_ptr,
                                              size_t embedding_entry_count_per_rank,
                                              int world_size,
                                              cudaDeviceProp* prop,
                                              cudaStream_t stream)
{
  try {
    WM_CUDA_CHECK(cudaMemsetAsync(dev_rank_id_count_ptr, 0, sizeof(int64_t) * world_size, stream));
    if (indice_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
    constexpr int K_DEFAULT_SM_COUNT = 108;
    int sm_count = (prop != nullptr) ? prop->multiProcessorCount : K_DEFAULT_SM_COUNT;
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      BucketIdForRanks,
                      indices,
                      indice_desc,
                      dev_rank_id_count_ptr,
                      embedding_entry_count_per_rank,
                      world_size,
                      sm_count,
                      stream);
    WM_CUDA_CHECK(cudaGetLastError());
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("bucket_ids_for_ranks CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
