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
#include <cassert>
#include <cstdint>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <wholememory/wholememory.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "wholememory/integer_utils.hpp"
#include "wholememory_ops/register.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

template <typename IndexT, int CROSS_OR_LOCAL = 0>
__global__ void bucket_ids_for_hierarchy_kernel(const IndexT* indices,
                                                size_t indice_count,
                                                int64_t* dev_rank_id_count_ptr,
                                                size_t embedding_entry_count_per_rank,
                                                int local_size,
                                                int bucket_size)
{
  extern __shared__ int rank_count_shared[];
  for (int idx = threadIdx.x; idx < bucket_size; idx += blockDim.x) {
    rank_count_shared[idx] = 0;
  }
  __syncthreads();
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indice_count;
       idx += blockDim.x * gridDim.x) {
    IndexT node_idx = indices[idx];
    if (node_idx < 0) continue;
    int rank   = node_idx / embedding_entry_count_per_rank;
    int bucket = 0;
    if (CROSS_OR_LOCAL == 0)  // bucket cross ranks
      bucket = rank % local_size;
    else  // bucket local ranks
      bucket = rank / local_size;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    atomicAdd_block(&rank_count_shared[bucket], 1);
#else
    atomicAdd(&rank_count_shared[bucket], 1);
#endif
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < bucket_size; idx += blockDim.x) {
    atomicAdd(reinterpret_cast<unsigned long long*>(dev_rank_id_count_ptr) + idx,
              static_cast<unsigned long long>(rank_count_shared[idx]));
  }
}

template <typename IndexT>
void bucket_ids_for_hierarchy_temp_func(const void* indices,
                                        wholememory_array_description_t indice_desc,
                                        int64_t* dev_rank_id_count_ptr,
                                        size_t embedding_entry_count_per_rank,
                                        int local_size,
                                        int cross_size,
                                        int bucket_cross_or_local,
                                        int sm_count,
                                        cudaStream_t stream)
{
  static constexpr int BLOCK_SIZE = 128;
  int block_count           = wholememory::div_rounding_up_unsafe(indice_desc.size, BLOCK_SIZE);
  block_count               = std::min(block_count, sm_count * 4);
  const IndexT* indices_ptr = static_cast<const IndexT*>(indices);
  indices_ptr += indice_desc.storage_offset;

  if (bucket_cross_or_local == 0) {
    int bucket_size = local_size;
    cudaMemsetAsync(dev_rank_id_count_ptr, 0, sizeof(int64_t) * bucket_size, stream);
    bucket_ids_for_hierarchy_kernel<IndexT, 0>
      <<<block_count, BLOCK_SIZE, sizeof(int) * bucket_size, stream>>>(
        indices_ptr,
        indice_desc.size,
        dev_rank_id_count_ptr,
        embedding_entry_count_per_rank,
        local_size,
        bucket_size);
  } else {
    int bucket_size = cross_size;
    cudaMemsetAsync(dev_rank_id_count_ptr, 0, sizeof(int64_t) * bucket_size, stream);
    bucket_ids_for_hierarchy_kernel<IndexT, 1>
      <<<block_count, BLOCK_SIZE, sizeof(int) * bucket_size, stream>>>(
        indices_ptr,
        indice_desc.size,
        dev_rank_id_count_ptr,
        embedding_entry_count_per_rank,
        local_size,
        bucket_size);
  }
}

REGISTER_DISPATCH_ONE_TYPE(BucketIdsForHierarchy, bucket_ids_for_hierarchy_temp_func, SINT3264)

template <typename IndexT>
__global__ void reorder_ids_for_hierarchy_kernel(const IndexT* indices,
                                                 size_t indice_count,
                                                 IndexT* dev_bucket_indices,
                                                 IndexT* dev_indice_map,
                                                 const int64_t* dev_rank_id_offset_ptr,
                                                 size_t embedding_entry_count_per_rank,
                                                 int local_size,
                                                 int64_t* dev_bucket_atomic_add_ptr)
{
  int nbucket                      = local_size;
  constexpr size_t shared_mem_size = 24576;
  __shared__ char shared_mem[shared_mem_size];
  int* block_bucket_count_shared      = reinterpret_cast<int*>(shared_mem);
  int* block_bucket_atomic_add_shared = reinterpret_cast<int*>(shared_mem) + nbucket;
  IndexT* block_bucket_offset_shared =
    reinterpret_cast<IndexT*>(shared_mem + 2 * sizeof(int) * nbucket);
  IndexT* global_bucket_offset_shared = block_bucket_offset_shared + nbucket;
  size_t buffer_size =
    (shared_mem_size - nbucket * 2 * (sizeof(IndexT) + sizeof(int))) / sizeof(IndexT) / 2;
  buffer_size = (buffer_size / blockDim.x) * blockDim.x;
  assert(buffer_size > 0);

  IndexT* buffer_load  = global_bucket_offset_shared + nbucket;
  IndexT* buffer_store = buffer_load + buffer_size;

  int warp_idx = threadIdx.x / warpSize;
  int lane_idx = threadIdx.x % warpSize;
  int nwarp    = blockDim.x / warpSize;
  for (IndexT load_offset = buffer_size * blockIdx.x; load_offset < indice_count;
       load_offset += gridDim.x * buffer_size) {
    for (int i = threadIdx.x; i < nbucket; i += blockDim.x) {
      block_bucket_count_shared[i]      = 0;
      block_bucket_atomic_add_shared[i] = 0;
    }
    __syncthreads();
    for (IndexT i = threadIdx.x; i < buffer_size; i += blockDim.x) {
      IndexT load_idx = i + load_offset;
      if (load_idx >= indice_count) break;
      IndexT indice = indices[load_idx];

      buffer_load[i] = indice;
      int bucket_idx = (indice / embedding_entry_count_per_rank) % local_size;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      atomicAdd_block(&block_bucket_count_shared[bucket_idx], 1);
#else
      atomicAdd(&block_bucket_count_shared[bucket_idx], 1);
#endif
    }
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {
      IndexT bucket_offset_tmp = 0;
      for (int bi = 0; bi < nbucket; bi++) {
        block_bucket_offset_shared[bi] = bucket_offset_tmp;
        bucket_offset_tmp += block_bucket_count_shared[bi];
      }
    }
    if (threadIdx.x < nbucket) {
      int bucket_idx = threadIdx.x;
      global_bucket_offset_shared[bucket_idx] =
        atomicAdd(reinterpret_cast<unsigned long long*>(dev_bucket_atomic_add_ptr) + bucket_idx,
                  block_bucket_count_shared[bucket_idx]);
    }
    __syncthreads();
    for (IndexT i = threadIdx.x; i < buffer_size; i += blockDim.x) {
      IndexT indice   = buffer_load[i];
      IndexT load_idx = i + load_offset;
      if (load_idx >= indice_count) break;
      int bucket_idx = (indice / embedding_entry_count_per_rank) % local_size;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      int block_bucket_inc = atomicAdd_block(&block_bucket_atomic_add_shared[bucket_idx], 1);
#else
      int block_bucket_inc = atomicAdd(&block_bucket_atomic_add_shared[bucket_idx], 1);
#endif
      buffer_store[block_bucket_offset_shared[bucket_idx] + block_bucket_inc] = indice;
      dev_indice_map[load_idx] = dev_rank_id_offset_ptr[bucket_idx] +
                                 global_bucket_offset_shared[bucket_idx] + block_bucket_inc;
    }
    __syncthreads();
    for (int bucket_idx = warp_idx; bucket_idx < nbucket; bucket_idx += nwarp) {
      int bucket_length = block_bucket_count_shared[bucket_idx];
      IndexT global_bucket_offset =
        dev_rank_id_offset_ptr[bucket_idx] + global_bucket_offset_shared[bucket_idx];
      for (int idx = lane_idx; idx < bucket_length; idx += warpSize) {
        dev_bucket_indices[global_bucket_offset + idx] =
          buffer_store[block_bucket_offset_shared[bucket_idx] + idx];
      }
    }
    __syncthreads();
  }
}

template <typename IndexT>
void reorder_ids_for_hierarchy_temp_func(const void* indices,
                                         wholememory_array_description_t indice_desc,
                                         void* dev_bucket_indices,
                                         void* dev_indice_map,
                                         const int64_t* dev_rank_id_count_ptr,
                                         size_t embedding_entry_count_per_rank,
                                         int local_size,
                                         wm_thrust_allocator* p_thrust_allocator,
                                         wholememory_env_func_t* p_env_fns,
                                         int sm_count,
                                         cudaStream_t stream)
{
  WHOLEMEMORY_CHECK(indice_desc.storage_offset == 0);
  WHOLEMEMORY_CHECK(indice_desc.dtype == WHOLEMEMORY_DT_INT ||
                    indice_desc.dtype == WHOLEMEMORY_DT_INT64);

  temp_memory_handle dev_rank_id_offset_handle(p_env_fns);
  int64_t* dev_rank_id_offset_ptr = static_cast<int64_t*>(
    dev_rank_id_offset_handle.device_malloc(local_size, WHOLEMEMORY_DT_INT64));
  void* cub_temp_storage    = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(cub_temp_storage,
                                temp_storage_bytes,
                                dev_rank_id_count_ptr,
                                dev_rank_id_offset_ptr,
                                local_size,
                                stream);
  cub_temp_storage = p_thrust_allocator->allocate(temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(cub_temp_storage,
                                temp_storage_bytes,
                                dev_rank_id_count_ptr,
                                dev_rank_id_offset_ptr,
                                local_size,
                                stream);
  p_thrust_allocator->deallocate(reinterpret_cast<char*>(cub_temp_storage), temp_storage_bytes);

  temp_memory_handle dev_bucket_atomic_add_handle(p_env_fns);
  int64_t* dev_bucket_atomic_add_ptr = static_cast<int64_t*>(
    dev_bucket_atomic_add_handle.device_malloc(local_size, WHOLEMEMORY_DT_INT64));
  cudaMemsetAsync((void*)dev_bucket_atomic_add_ptr, 0, sizeof(int64_t) * local_size, stream);
  static constexpr int BLOCK_SIZE = 128;
  int block_count = wholememory::div_rounding_up_unsafe(indice_desc.size, BLOCK_SIZE);
  block_count     = std::min(block_count, sm_count * 4);

  reorder_ids_for_hierarchy_kernel<<<block_count, BLOCK_SIZE, 0, stream>>>(
    static_cast<const IndexT*>(indices),
    indice_desc.size,
    static_cast<IndexT*>(dev_bucket_indices),
    static_cast<IndexT*>(dev_indice_map),
    dev_rank_id_offset_ptr,
    embedding_entry_count_per_rank,
    local_size,
    dev_bucket_atomic_add_ptr);
}

REGISTER_DISPATCH_ONE_TYPE(ReorderIdsForHierarchy, reorder_ids_for_hierarchy_temp_func, SINT3264)

wholememory_error_code_t bucket_and_reorder_ids_for_hierarchy_func(
  void* indices,
  wholememory_array_description_t indice_desc,
  void* dev_bucket_indices,
  void* dev_indice_map,
  int64_t* host_bucket_id_count,
  size_t embedding_entry_count_per_rank,
  wholememory_comm_t wm_global_comm,
  wholememory_comm_t wm_local_comm,
  wm_thrust_allocator* p_thrust_allocator,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  if (indice_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
  int world_size, local_size;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_global_comm));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&local_size, wm_local_comm));
  WHOLEMEMORY_CHECK_NOTHROW(world_size % local_size == 0);

  constexpr int K_DEFAULT_SM_COUNT = 108;
  auto prop                        = get_device_prop(-1);
  int sm_count = (prop != nullptr) ? prop->multiProcessorCount : K_DEFAULT_SM_COUNT;
  temp_memory_handle dev_rank_id_count_handle(p_env_fns);
  int64_t* dev_rank_id_count_ptr =
    static_cast<int64_t*>(dev_rank_id_count_handle.device_malloc(local_size, WHOLEMEMORY_DT_INT64));
  cudaMemsetAsync((void*)dev_rank_id_count_ptr, 0, sizeof(int64_t) * local_size, stream);
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      BucketIdsForHierarchy,
                      indices,
                      indice_desc,
                      dev_rank_id_count_ptr,
                      embedding_entry_count_per_rank,
                      local_size,
                      0,  // ignore
                      0,
                      sm_count,
                      stream);
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("bucket_ids_for_hierarchy_func CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  }
  WM_CUDA_CHECK_NO_THROW(cudaMemcpyAsync(host_bucket_id_count,
                                         dev_rank_id_count_ptr,
                                         local_size * sizeof(int64_t),
                                         cudaMemcpyDeviceToHost,
                                         stream));
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      ReorderIdsForHierarchy,
                      indices,
                      indice_desc,
                      dev_bucket_indices,
                      dev_indice_map,
                      dev_rank_id_count_ptr,
                      embedding_entry_count_per_rank,
                      local_size,
                      p_thrust_allocator,
                      p_env_fns,
                      sm_count,
                      stream);
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("reorder_ids_for_hierarchy CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("reorder_ids_for_hierarchy LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t bucket_local_ids_func(void* indices,
                                               wholememory_array_description_t indice_desc,
                                               int64_t* host_bucket_id_count,
                                               size_t embedding_entry_count_per_rank,
                                               wholememory_comm_t wm_local_comm,
                                               wholememory_comm_t wm_cross_comm,
                                               wm_thrust_allocator* p_thrust_allocator,
                                               wholememory_env_func_t* p_env_fns,
                                               cudaStream_t stream)
{
  if (indice_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
  int cross_size, local_size;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&cross_size, wm_cross_comm));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&local_size, wm_local_comm));

  constexpr int K_DEFAULT_SM_COUNT = 108;
  auto prop                        = get_device_prop(-1);
  int sm_count = (prop != nullptr) ? prop->multiProcessorCount : K_DEFAULT_SM_COUNT;
  temp_memory_handle dev_rank_id_count_handle(p_env_fns);
  int64_t* dev_rank_id_count_ptr =
    static_cast<int64_t*>(dev_rank_id_count_handle.device_malloc(cross_size, WHOLEMEMORY_DT_INT64));
  cudaMemsetAsync((void*)dev_rank_id_count_ptr, 0, sizeof(int64_t) * cross_size, stream);
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      BucketIdsForHierarchy,
                      indices,
                      indice_desc,
                      dev_rank_id_count_ptr,
                      embedding_entry_count_per_rank,
                      local_size,
                      cross_size,
                      1,
                      sm_count,
                      stream);
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("bucket_ids_for_hierarchy CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  }
  WM_CUDA_CHECK_NO_THROW(cudaMemcpyAsync(host_bucket_id_count,
                                         dev_rank_id_count_ptr,
                                         cross_size * sizeof(int64_t),
                                         cudaMemcpyDeviceToHost,
                                         stream));
  WM_CUDA_CHECK(cudaGetLastError());
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
