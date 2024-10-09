/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#ifdef WITH_NVSHMEM_SUPPORT

#include "nvshmem_template.cuh"
#include "wholememory/device_reference.cuh"

namespace wholememory_ops {

template <typename DataTypeT>
class nvshmem_device_reference {
 public:
  __device__ __forceinline__ explicit nvshmem_device_reference(
    const wholememory_nvshmem_ref_t& nvshmem_ref)
    : pointer_(static_cast<DataTypeT*>(nvshmem_ref.pointer)),
      typed_stride_(nvshmem_ref.stride / sizeof(DataTypeT)),
      rank_memory_offsets_(nvshmem_ref.rank_memory_offsets),
      world_size_(nvshmem_ref.world_size),
      same_chunk_(nvshmem_ref.same_chunk)
  {
    assert(nvshmem_ref.stride % sizeof(DataTypeT) == 0);
    if (!same_chunk_) {
      estimated_stride_ = rank_memory_offsets_[world_size_] / world_size_;
      cache_rank_       = 0;
      cache_offset_     = 0;
      cache_size_       = rank_memory_offsets_[1] - rank_memory_offsets_[0];
    }
  }

  __device__ nvshmem_device_reference() = delete;

  __device__ __forceinline__ DataTypeT load(size_t index)
  {
    size_t rank = dest_rank(index);
    if (same_chunk_)
      return nvshmem_get<DataTypeT>(pointer_ + index - rank * typed_stride_, rank);
    else
      return nvshmem_get<DataTypeT>(
        pointer_ + index - rank_memory_offsets_[rank] / sizeof(DataTypeT), rank);
  }

  __device__ __forceinline__ void store(size_t index, DataTypeT val)
  {
    size_t rank = dest_rank(index);
    if (same_chunk_)
      return nvshmem_put<DataTypeT>(pointer_ + index - rank * typed_stride_, rank);
    else
      return nvshmem_put<DataTypeT>(
        pointer_ + index - rank_memory_offsets_[rank] / sizeof(DataTypeT), val, rank);
  }

  __device__ __forceinline__ DataTypeT* symmetric_address(size_t index)
  {
    size_t rank = dest_rank(index);
    if (same_chunk_)
      return pointer_ + index - rank * typed_stride_;
    else
      return pointer_ + index - rank_memory_offsets_[rank] / sizeof(DataTypeT);
  }

  __device__ __forceinline__ void mov_offsets_to_shmem(char* shmem)
  {
    if (same_chunk_) return;
    size_t* shmem_offsets = reinterpret_cast<size_t*>(shmem);
    for (int i = threadIdx.x; i <= world_size_; i += blockDim.x) {
      shmem_offsets[i] = rank_memory_offsets_[i];
    }
    __syncthreads();
    rank_memory_offsets_ = shmem_offsets;
  }

  __device__ __forceinline__ size_t dest_rank(size_t index)
  {
    if (same_chunk_) {
      return index / typed_stride_;
    } else {
      size_t rank   = 0;
      size_t offset = index * sizeof(DataTypeT);
      if (offset >= cache_offset_ && offset < cache_offset_ + cache_size_) {
        rank = cache_rank_;
      } else {
        int estimated_rank = max(world_size_ - 1, int(offset / estimated_stride_));
        if (rank_memory_offsets_[estimated_rank] > offset) {
          for (int i = estimated_rank - 1; i >= 0; i--) {
            if (rank_memory_offsets_[i] <= offset) {
              rank = i;
              break;
            }
          }
        } else {
          for (int i = estimated_rank + 1; i <= world_size_; i++) {
            if (rank_memory_offsets_[i] > offset) {
              rank = i - 1;
              break;
            }
          }
        }
        cache_rank_   = rank;
        cache_offset_ = rank_memory_offsets_[rank];
        cache_size_   = rank_memory_offsets_[rank + 1] - rank_memory_offsets_[rank];
      }
      return rank;
    }
  }

 private:
  DataTypeT* pointer_;
  size_t typed_stride_;
  size_t* rank_memory_offsets_;
  int world_size_;

  size_t estimated_stride_;
  bool same_chunk_;
  int cache_rank_;
  size_t cache_offset_;
  size_t cache_size_;
};
}  // namespace wholememory_ops

#endif
