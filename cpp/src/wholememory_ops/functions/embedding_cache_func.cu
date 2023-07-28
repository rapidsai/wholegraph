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
#include "embedding_cache_func.h"

#include <cub/cub.cuh>

#include <wholememory/wholememory_op.h>
#include <wholememory/wholememory_tensor.h>

#include "cuda_macros.hpp"
#include "embedding_cache_func.cuh"
#include "error.hpp"
#include "exchange_ids_nccl_func.h"
#include "logger.hpp"
#include "wholememory/embedding_cache.hpp"
#include "wholememory/env_func_ptrs.h"
#include "wholememory/integer_utils.hpp"
#include "wholememory_ops/functions/embedding_cache_func.cuh"
#include "wholememory_ops/register.hpp"

namespace wholememory_ops {

/**
 * Sort local indices, do unique and return unique_indices and count of each indices
 * @tparam IndexT : data type of indices
 * @param indices : indices to process
 * @param indice_desc : description of indices
 * @param num_runs : return number of unique indices
 * @param unique_indices_handle : temp_memory_handle of unique indices
 * @param unique_count_handle : temp_memory_handle of count of each unique indices
 * @param p_thrust_allocator : thrust allocator
 * @param p_env_fns : env_fns
 * @param stream : CUDA stream to use
 */
template <typename IndexT>
void SortUniqueLocalIndicesTempFunc(const void* indices,
                                    wholememory_array_description_t indice_desc,
                                    int* num_runs,
                                    temp_memory_handle* unique_indices_handle,
                                    temp_memory_handle* unique_count_handle,
                                    wm_thrust_allocator* p_thrust_allocator,
                                    wholememory_env_func_t* p_env_fns,
                                    cudaStream_t stream)
{
  if (indice_desc.size == 0) return;
  wm_thrust_allocator& allocator = *p_thrust_allocator;
  WHOLEMEMORY_CHECK_NOTHROW(indice_desc.storage_offset == 0);
  const IndexT* indices_to_sort = static_cast<const IndexT*>(indices);
  temp_memory_handle sorted_indices_handle(p_env_fns);
  sorted_indices_handle.device_malloc(indice_desc.size, indice_desc.dtype);
  IndexT* sorted_indices    = static_cast<IndexT*>(sorted_indices_handle.pointer());
  void* cub_temp_storage    = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(cub_temp_storage,
                                 temp_storage_bytes,
                                 indices_to_sort,
                                 sorted_indices,
                                 indice_desc.size,
                                 0,
                                 sizeof(IndexT) * 8,
                                 stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceRadixSort::SortKeys(cub_temp_storage,
                                 temp_storage_bytes,
                                 indices_to_sort,
                                 sorted_indices,
                                 indice_desc.size,
                                 0,
                                 sizeof(IndexT) * 8,
                                 stream);
  unique_indices_handle->device_malloc(indice_desc.size, indice_desc.dtype);
  unique_count_handle->device_malloc(indice_desc.size, WHOLEMEMORY_DT_INT);
  IndexT* unique_indices = static_cast<IndexT*>(unique_indices_handle->pointer());
  int* unique_counts     = static_cast<int*>(unique_count_handle->pointer());
  temp_memory_handle number_runs_handle(p_env_fns);
  number_runs_handle.device_malloc(1, WHOLEMEMORY_DT_INT);
  int* number_runs   = static_cast<int*>(number_runs_handle.pointer());
  cub_temp_storage   = nullptr;
  temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(cub_temp_storage,
                                     temp_storage_bytes,
                                     sorted_indices,
                                     unique_indices,
                                     unique_counts,
                                     number_runs,
                                     indice_desc.size,
                                     stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceRunLengthEncode::Encode(cub_temp_storage,
                                     temp_storage_bytes,
                                     sorted_indices,
                                     unique_indices,
                                     unique_counts,
                                     number_runs,
                                     indice_desc.size,
                                     stream);
  WM_CUDA_CHECK_NO_THROW(
    cudaMemcpyAsync(num_runs, number_runs, sizeof(int), cudaMemcpyDeviceToHost, stream));
  WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
}

REGISTER_DISPATCH_ONE_TYPE(SortUniqueLocalIndicesTempFunc, SortUniqueLocalIndicesTempFunc, SINT3264)

template <typename IndexT>
__global__ void ComputeCacheSetLocalID(const IndexT* indices,
                                       int* cache_set_lid,
                                       int indices_num_run,
                                       int64_t rank_start_gid,
                                       int cache_set_coverage)
{
  int const thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_idx >= indices_num_run) return;
  int const cache_set_local_id = (indices[thread_idx] - rank_start_gid) / cache_set_coverage;
  cache_set_lid[thread_idx]    = cache_set_local_id;
}

template <typename IndexT>
void BucketByCacheSetTempFunc(const void* unique_indices,
                              int indices_num_run,
                              temp_memory_handle* unique_cache_set_lid_handle,
                              temp_memory_handle* unique_cache_set_start_handle,
                              temp_memory_handle* unique_cache_set_count_handle,
                              int* cache_set_num_run,
                              int64_t rank_start_gid,
                              int cache_set_coverage,
                              wm_thrust_allocator* p_thrust_allocator,
                              wholememory_env_func_t* p_env_fns,
                              cudaStream_t stream)
{
  if (indices_num_run == 0) return;
  wm_thrust_allocator& allocator = *p_thrust_allocator;
  temp_memory_handle cache_set_lid_handle(p_env_fns);
  cache_set_lid_handle.device_malloc(indices_num_run, WHOLEMEMORY_DT_INT);
  int* cache_set_lid    = static_cast<int*>(cache_set_lid_handle.pointer());
  int const block_count = wholememory::div_rounding_up_unsafe(indices_num_run, 32);
  ComputeCacheSetLocalID<<<block_count, 32, 0, stream>>>(static_cast<const IndexT*>(unique_indices),
                                                         cache_set_lid,
                                                         indices_num_run,
                                                         rank_start_gid,
                                                         cache_set_coverage);
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
  void* cub_temp_storage    = nullptr;
  size_t temp_storage_bytes = 0;
  temp_memory_handle cache_set_num_run_handle(p_env_fns);
  unique_cache_set_lid_handle->device_malloc(indices_num_run, WHOLEMEMORY_DT_INT);
  unique_cache_set_count_handle->device_malloc(indices_num_run, WHOLEMEMORY_DT_INT);
  unique_cache_set_start_handle->device_malloc(indices_num_run, WHOLEMEMORY_DT_INT);
  cache_set_num_run_handle.device_malloc(1, WHOLEMEMORY_DT_INT);
  int* unique_cache_set_lid   = static_cast<int*>(unique_cache_set_lid_handle->pointer());
  int* unique_cache_set_start = static_cast<int*>(unique_cache_set_start_handle->pointer());
  int* unique_cache_set_count = static_cast<int*>(unique_cache_set_count_handle->pointer());
  int* cache_set_num_run_d    = static_cast<int*>(cache_set_num_run_handle.pointer());
  cub::DeviceRunLengthEncode::Encode(cub_temp_storage,
                                     temp_storage_bytes,
                                     cache_set_lid,
                                     unique_cache_set_lid,
                                     unique_cache_set_count,
                                     cache_set_num_run_d,
                                     indices_num_run,
                                     stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceRunLengthEncode::Encode(cub_temp_storage,
                                     temp_storage_bytes,
                                     cache_set_lid,
                                     unique_cache_set_lid,
                                     unique_cache_set_count,
                                     cache_set_num_run_d,
                                     indices_num_run,
                                     stream);
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
  *cache_set_num_run = 0;
  WM_CUDA_CHECK_NO_THROW(cudaMemcpyAsync(
    cache_set_num_run, cache_set_num_run_d, sizeof(int), cudaMemcpyDeviceToHost, stream));
  WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
  if (*cache_set_num_run == 0) return;
  cub_temp_storage   = nullptr;
  temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(cub_temp_storage,
                                temp_storage_bytes,
                                unique_cache_set_count,
                                unique_cache_set_start,
                                *cache_set_num_run,
                                stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(cub_temp_storage,
                                temp_storage_bytes,
                                unique_cache_set_count,
                                unique_cache_set_start,
                                *cache_set_num_run,
                                stream);
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_ONE_TYPE(BucketByCacheSetTempFunc, BucketByCacheSetTempFunc, SINT3264)

template <typename IndexT>
__global__ void UpdateCacheDirectKernel(const int* unique_cache_set_lid,
                                        const int* unique_cache_set_update_start,
                                        const int* unique_cache_set_update_count,
                                        const IndexT* unique_indices,
                                        const int* unique_indices_count,
                                        uint16_t* local_cache_line_tag,
                                        uint16_t* local_cache_line_lfu_count,
                                        int64_t* local_access_count,
                                        int4* local_cached_data,
                                        int4* local_memory_data,
                                        int embedding_dim_in_int4,
                                        int64_t rank_start_gid,
                                        int cache_set_coverage)
{
  static_assert(wholememory::embedding_cache_base::kCacheSetSize == 32);
  int64_t const cache_set_lid = unique_cache_set_lid[blockIdx.x];
  local_cache_line_tag += cache_set_lid * wholememory::embedding_cache_base::kCacheSetSize;
  local_cache_line_lfu_count += cache_set_lid * wholememory::embedding_cache_base::kCacheSetSize;
  local_access_count += cache_set_lid * cache_set_coverage;
  local_cached_data +=
    cache_set_lid * wholememory::embedding_cache_base::kCacheSetSize * embedding_dim_in_int4;
  local_memory_data += cache_set_lid * cache_set_coverage * embedding_dim_in_int4;
  CacheLineInfo cache_line_info;
  cache_line_info.LoadInfo(local_cache_line_tag, local_cache_line_lfu_count);
  int cache_set_update_start_idx = unique_cache_set_update_start[blockIdx.x];
  int cache_set_update_count     = unique_cache_set_update_count[blockIdx.x];
  using Updater                  = wholememory_ops::CacheSetUpdater<IndexT>;
  Updater updater;
  __shared__ typename Updater::TempStorage temp_storage;
  __shared__ IndexT s_load_to_cache_ids[CacheSetUpdater<IndexT>::kCacheSetSize];
  __shared__ IndexT s_write_back_to_memory_ids[CacheSetUpdater<IndexT>::kCacheSetSize];
  s_load_to_cache_ids[threadIdx.x]        = -1;
  s_write_back_to_memory_ids[threadIdx.x] = -1;
  int old_cached_lid                      = cache_line_info.LocalID();
  __syncthreads();
  updater.template UpdateCache<true, true>(temp_storage,
                                           cache_line_info,
                                           local_access_count,
                                           unique_indices + cache_set_update_start_idx,
                                           unique_indices_count + cache_set_update_start_idx,
                                           &s_load_to_cache_ids[0],
                                           &s_write_back_to_memory_ids[0],
                                           rank_start_gid + cache_set_coverage * cache_set_lid,
                                           cache_set_update_count);
  __syncthreads();

  IndexT thread_node_id     = s_write_back_to_memory_ids[threadIdx.x];
  unsigned int valid_mask   = __ballot_sync(0xFFFFFFFF, thread_node_id >= 0);
  int need_write_back_count = __popc(valid_mask);
  assert(valid_mask == (1ULL << need_write_back_count) - 1);
  for (int i = 0; i < need_write_back_count; i++) {
    IndexT write_back_gid = s_write_back_to_memory_ids[i];
    int local_id          = write_back_gid - rank_start_gid - cache_set_coverage * cache_set_lid;
    uint32_t mask         = __ballot_sync(0xFFFFFFFF, static_cast<int>(old_cached_lid == local_id));
    int cache_line_idx    = __ffs(mask) - 1;
    assert(cache_line_idx >= 0 && cache_line_idx <= 32);
    assert(local_id >= 0 && local_id < cache_set_coverage);
    for (int idx = threadIdx.x; idx < embedding_dim_in_int4; idx += 32) {
      local_memory_data[local_id * embedding_dim_in_int4 + idx] =
        local_cached_data[cache_line_idx * embedding_dim_in_int4 + idx];
    }
  }

  thread_node_id      = s_load_to_cache_ids[threadIdx.x];
  valid_mask          = __ballot_sync(0xFFFFFFFF, thread_node_id >= 0);
  int need_load_count = __popc(valid_mask);
  assert(valid_mask == (1ULL << need_load_count) - 1);
  for (int i = 0; i < need_load_count; i++) {
    IndexT load_gid    = s_load_to_cache_ids[i];
    int local_id       = load_gid - rank_start_gid - cache_set_coverage * cache_set_lid;
    int cache_line_idx = cache_line_info.KeyIndexSync(local_id);
    assert(cache_line_idx >= 0 && cache_line_idx <= 32);
    assert(local_id >= 0 && local_id < cache_set_coverage);
    for (int idx = threadIdx.x; idx < embedding_dim_in_int4; idx += 32) {
      local_cached_data[cache_line_idx * embedding_dim_in_int4 + idx] =
        local_memory_data[local_id * embedding_dim_in_int4 + idx];
    }
  }

  cache_line_info.StoreInfo(local_cache_line_tag, local_cache_line_lfu_count);
}

template <typename IndexT>
void UpdateCacheDirectTempFunc(const int* unique_cache_set_lid,
                               const int* unique_cache_set_start,
                               const int* unique_cache_set_count,
                               const void* unique_indices,
                               const int* unique_indices_count,
                               uint16_t* local_cache_line_tag,
                               uint16_t* local_cache_line_lfu_count,
                               int64_t* local_access_count,
                               int4* local_cache_line_data,
                               int4* local_memory_data,
                               int embedding_dim_in_int4,
                               int cache_set_num_run,
                               int64_t rank_start_gid,
                               int cache_set_coverage,
                               cudaStream_t stream)
{
  if (cache_set_num_run > 0) {
    UpdateCacheDirectKernel<IndexT>
      <<<cache_set_num_run, 32, 0, stream>>>(unique_cache_set_lid,
                                             unique_cache_set_start,
                                             unique_cache_set_count,
                                             static_cast<const IndexT*>(unique_indices),
                                             unique_indices_count,
                                             local_cache_line_tag,
                                             local_cache_line_lfu_count,
                                             local_access_count,
                                             local_cache_line_data,
                                             local_memory_data,
                                             embedding_dim_in_int4,
                                             rank_start_gid,
                                             cache_set_coverage);
  }
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_ONE_TYPE(UpdateCacheDirectTempFunc, UpdateCacheDirectTempFunc, SINT3264)

wholememory_error_code_t update_cache_direct_same_comm(
  void* indices,
  wholememory_array_description_t indice_desc,
  wholememory_tensor_t wm_raw_memory_embedding,
  const wholememory::embedding_cache_local_data* cache_local_data,
  int cache_set_coverage,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  wm_thrust_allocator thrust_allocator(p_env_fns);
  int world_size = 1;
  int world_rank = 0;
  wholememory_handle_t wholememory_handle =
    wholememory_tensor_get_memory_handle(wm_raw_memory_embedding);
  wholememory_comm_t wm_comm;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, wholememory_handle));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&world_rank, wm_comm));

  auto* raw_embedding_desc =
    wholememory_tensor_get_tensor_description(wholememory_tensor_get_root(wm_raw_memory_embedding));
  size_t embedding_entry_count_per_rank = 0;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_determine_entry_partition_plan(
    &embedding_entry_count_per_rank, raw_embedding_desc->sizes[0], world_size));

  int indices_num_run = 0;
  temp_memory_handle unique_indice_handle(p_env_fns), unique_count_handle(p_env_fns);
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      SortUniqueLocalIndicesTempFunc,
                      indices,
                      indice_desc,
                      &indices_num_run,
                      &unique_indice_handle,
                      &unique_count_handle,
                      &thrust_allocator,
                      p_env_fns,
                      stream);
  } catch (...) {
    WHOLEMEMORY_ERROR("SortUniqueLocalIndicesTempFunc failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  temp_memory_handle unique_cache_set_lid_handle(p_env_fns),
    unique_cache_set_start_handle(p_env_fns), unique_cache_set_count_handle(p_env_fns);
  int cache_set_num_run;
  DISPATCH_ONE_TYPE(indice_desc.dtype,
                    BucketByCacheSetTempFunc,
                    unique_indice_handle.pointer(),
                    indices_num_run,
                    &unique_cache_set_lid_handle,
                    &unique_cache_set_start_handle,
                    &unique_cache_set_count_handle,
                    &cache_set_num_run,
                    world_rank * embedding_entry_count_per_rank,
                    cache_set_coverage,
                    &thrust_allocator,
                    p_env_fns,
                    stream);
  int* unique_cache_set_lid     = static_cast<int*>(unique_cache_set_lid_handle.pointer());
  int* unique_cache_set_start   = static_cast<int*>(unique_cache_set_start_handle.pointer());
  int* unique_cache_set_count   = static_cast<int*>(unique_cache_set_count_handle.pointer());
  void* embedding_local_pointer = nullptr;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_get_local_memory(&embedding_local_pointer,
                                 nullptr,
                                 nullptr,
                                 wholememory_tensor_get_memory_handle(wm_raw_memory_embedding)));
  int const embedding_dim = raw_embedding_desc->strides[0];
  size_t const dtype_size = wholememory_dtype_get_element_size(raw_embedding_desc->dtype);
  WHOLEMEMORY_CHECK_NOTHROW(embedding_dim * dtype_size % 16 == 0);
  int const embedding_dim_in_int4 = embedding_dim * dtype_size / 16;
  DISPATCH_ONE_TYPE(
    indice_desc.dtype,
    UpdateCacheDirectTempFunc,
    unique_cache_set_lid,
    unique_cache_set_start,
    unique_cache_set_count,
    unique_indice_handle.pointer(),
    static_cast<const int*>(unique_count_handle.pointer()),
    static_cast<uint16_t*>(wholememory_tensor_get_data_pointer(cache_local_data->cache_line_tag_)),
    static_cast<uint16_t*>(
      wholememory_tensor_get_data_pointer(cache_local_data->cache_line_lfu_count_)),
    static_cast<int64_t*>(wholememory_tensor_get_data_pointer(cache_local_data->access_count_)),
    static_cast<int4*>(wholememory_tensor_get_data_pointer(cache_local_data->cache_line_data_)),
    static_cast<int4*>(embedding_local_pointer),
    embedding_dim_in_int4,
    cache_set_num_run,
    world_rank * embedding_entry_count_per_rank,
    cache_set_coverage,
    stream);

  return WHOLEMEMORY_SUCCESS;
}

template <typename IndexT>
__global__ void DetermineLoadCacheKernel(const int* unique_cache_set_lid,
                                         const int* unique_cache_set_update_start,
                                         const int* unique_cache_set_update_count,
                                         const IndexT* unique_indices,
                                         const int* unique_indices_count,
                                         uint16_t* local_cache_line_tag,
                                         uint16_t* local_cache_line_lfu_count,
                                         int64_t* local_access_count,
                                         IndexT* output_local_write_cache_index,
                                         IndexT* output_global_load_gid,
                                         int64_t rank_start_gid,
                                         int cache_set_coverage)
{
  static_assert(wholememory::embedding_cache_base::kCacheSetSize == 32);
  int64_t const cache_set_lid = unique_cache_set_lid[blockIdx.x];
  local_cache_line_tag += cache_set_lid * wholememory::embedding_cache_base::kCacheSetSize;
  local_cache_line_lfu_count += cache_set_lid * wholememory::embedding_cache_base::kCacheSetSize;
  local_access_count += cache_set_lid * cache_set_coverage;
  CacheLineInfo cache_line_info;
  cache_line_info.LoadInfo(local_cache_line_tag, local_cache_line_lfu_count);
  int cache_set_update_start_idx = unique_cache_set_update_start[blockIdx.x];
  int cache_set_update_count     = unique_cache_set_update_count[blockIdx.x];
  output_local_write_cache_index += cache_set_update_start_idx;
  output_global_load_gid += cache_set_update_start_idx;
  using Updater = wholememory_ops::CacheSetUpdater<IndexT>;
  Updater updater;
  __shared__ typename Updater::TempStorage temp_storage;
  __shared__ IndexT s_load_to_cache_ids[CacheSetUpdater<IndexT>::kCacheSetSize];
  s_load_to_cache_ids[threadIdx.x] = -1;
  int old_cached_lid               = cache_line_info.LocalID();
  __syncthreads();
  updater.template UpdateCache<true, false>(temp_storage,
                                            cache_line_info,
                                            local_access_count,
                                            unique_indices + cache_set_update_start_idx,
                                            unique_indices_count + cache_set_update_start_idx,
                                            &s_load_to_cache_ids[0],
                                            nullptr,
                                            rank_start_gid + cache_set_coverage * cache_set_lid,
                                            cache_set_update_count);
  __syncthreads();

  IndexT thread_node_id   = s_load_to_cache_ids[threadIdx.x];
  unsigned int valid_mask = __ballot_sync(0xFFFFFFFF, thread_node_id >= 0);
  int need_load_count     = __popc(valid_mask);
  assert(valid_mask == (1ULL << need_load_count) - 1);
  for (int i = 0; i < need_load_count; i++) {
    IndexT load_gid    = s_load_to_cache_ids[i];
    int local_id       = load_gid - rank_start_gid - cache_set_coverage * cache_set_lid;
    int cache_line_idx = cache_line_info.KeyIndexSync(local_id);
    assert(cache_line_idx >= 0 && cache_line_idx <= 32);
    output_global_load_gid[i] = load_gid;
    output_local_write_cache_index[i] =
      cache_line_idx + cache_set_lid * wholememory::embedding_cache_base::kCacheSetSize;
  }
  for (int i = need_load_count + threadIdx.x; i < cache_set_update_count; i += 32) {
    output_global_load_gid[i] = output_local_write_cache_index[i] = -1;
  }

  cache_line_info.StoreInfo(local_cache_line_tag, local_cache_line_lfu_count);
}

template <typename IndexT>
void DetermineLoadCacheTempFunc(const int* unique_cache_set_lid,
                                const int* unique_cache_set_update_start,
                                const int* unique_cache_set_update_count,
                                const void* unique_indices,
                                const int* unique_indices_count,
                                uint16_t* local_cache_line_tag,
                                uint16_t* local_cache_line_lfu_count,
                                int64_t* local_access_count,
                                void* output_local_write_cache_index,
                                void* output_global_load_gid,
                                int64_t rank_start_gid,
                                int cache_set_coverage,
                                int cache_set_num_run,
                                cudaStream_t stream)
{
  if (cache_set_num_run > 0) {
    DetermineLoadCacheKernel<IndexT>
      <<<cache_set_num_run, 32, 0, stream>>>(unique_cache_set_lid,
                                             unique_cache_set_update_start,
                                             unique_cache_set_update_count,
                                             static_cast<const IndexT*>(unique_indices),
                                             unique_indices_count,
                                             local_cache_line_tag,
                                             local_cache_line_lfu_count,
                                             local_access_count,
                                             static_cast<IndexT*>(output_local_write_cache_index),
                                             static_cast<IndexT*>(output_global_load_gid),
                                             rank_start_gid,
                                             cache_set_coverage);
  }
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_ONE_TYPE(DetermineLoadCacheTempFunc, DetermineLoadCacheTempFunc, SINT3264)

wholememory_error_code_t update_cache_different_comm(
  void* indices,
  wholememory_array_description_t indice_desc,
  wholememory_tensor_t wm_raw_memory_embedding,
  wholememory_comm_t cache_comm,
  size_t embedding_entry_count_per_cache_rank,
  const wholememory::embedding_cache_local_data* cache_local_data,
  int cache_set_coverage,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  wm_thrust_allocator thrust_allocator(p_env_fns);
  int cache_world_size = 1;
  int cache_world_rank = 0;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&cache_world_size, cache_comm));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&cache_world_rank, cache_comm));

  int indices_num_run = 0;
  temp_memory_handle unique_indice_handle(p_env_fns), unique_count_handle(p_env_fns);
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      SortUniqueLocalIndicesTempFunc,
                      indices,
                      indice_desc,
                      &indices_num_run,
                      &unique_indice_handle,
                      &unique_count_handle,
                      &thrust_allocator,
                      p_env_fns,
                      stream);
  } catch (...) {
    WHOLEMEMORY_ERROR("SortUniqueLocalIndicesTempFunc failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  temp_memory_handle unique_cache_set_lid_handle(p_env_fns),
    unique_cache_set_start_handle(p_env_fns), unique_cache_set_count_handle(p_env_fns);
  int cache_set_num_run;
  DISPATCH_ONE_TYPE(indice_desc.dtype,
                    BucketByCacheSetTempFunc,
                    unique_indice_handle.pointer(),
                    indices_num_run,
                    &unique_cache_set_lid_handle,
                    &unique_cache_set_start_handle,
                    &unique_cache_set_count_handle,
                    &cache_set_num_run,
                    cache_world_rank * embedding_entry_count_per_cache_rank,
                    cache_set_coverage,
                    &thrust_allocator,
                    p_env_fns,
                    stream);
  int* unique_cache_set_lid   = static_cast<int*>(unique_cache_set_lid_handle.pointer());
  int* unique_cache_set_start = static_cast<int*>(unique_cache_set_start_handle.pointer());
  int* unique_cache_set_count = static_cast<int*>(unique_cache_set_count_handle.pointer());
  temp_memory_handle global_load_gid_handle(p_env_fns), local_write_cache_index_handle(p_env_fns);
  void* global_load_gid_ptr =
    global_load_gid_handle.device_malloc(indices_num_run, indice_desc.dtype);
  void* local_write_cache_index_ptr =
    local_write_cache_index_handle.device_malloc(indices_num_run, indice_desc.dtype);
  try {
    DISPATCH_ONE_TYPE(
      indice_desc.dtype,
      DetermineLoadCacheTempFunc,
      unique_cache_set_lid,
      unique_cache_set_start,
      unique_cache_set_count,
      unique_indice_handle.pointer(),
      static_cast<const int*>(unique_count_handle.pointer()),
      static_cast<uint16_t*>(
        wholememory_tensor_get_data_pointer(cache_local_data->cache_line_tag_)),
      static_cast<uint16_t*>(
        wholememory_tensor_get_data_pointer(cache_local_data->cache_line_lfu_count_)),
      static_cast<int64_t*>(wholememory_tensor_get_data_pointer(cache_local_data->access_count_)),
      local_write_cache_index_ptr,
      global_load_gid_ptr,
      cache_world_rank * embedding_entry_count_per_cache_rank,
      cache_set_coverage,
      cache_set_num_run,
      stream);
  } catch (...) {
    WHOLEMEMORY_ERROR("DetermineLoadCacheTempFunc failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  temp_memory_handle temp_cache_buffer_handle(p_env_fns);
  wholememory_tensor_description_t temp_cache_desc =
    *wholememory_tensor_get_tensor_description(wm_raw_memory_embedding);
  temp_cache_desc.storage_offset = 0;
  temp_cache_desc.sizes[0]       = indices_num_run;
  void* temp_cache_ptr           = temp_cache_buffer_handle.device_malloc(
    wholememory_get_memory_element_count_from_tensor(&temp_cache_desc), temp_cache_desc.dtype);
  wholememory_tensor_t temp_cache_tensor;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_make_tensor_from_pointer(&temp_cache_tensor, temp_cache_ptr, &temp_cache_desc));

  wholememory_tensor_description_t cache_indice_desc;
  wholememory_copy_array_desc_to_tensor(&cache_indice_desc, &indice_desc);
  cache_indice_desc.sizes[0]       = indices_num_run;
  cache_indice_desc.storage_offset = 0;

  wholememory_tensor_t gather_indice_tensor, scatter_indice_tensor;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_make_tensor_from_pointer(
    &gather_indice_tensor, global_load_gid_ptr, &cache_indice_desc));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_make_tensor_from_pointer(
    &scatter_indice_tensor, local_write_cache_index_ptr, &cache_indice_desc));

  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_gather(
    wm_raw_memory_embedding, gather_indice_tensor, temp_cache_tensor, p_env_fns, stream));

  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_scatter(temp_cache_tensor,
                                                 scatter_indice_tensor,
                                                 cache_local_data->cache_line_data_,
                                                 p_env_fns,
                                                 stream));

  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_destroy_tensor(temp_cache_tensor));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_destroy_tensor(gather_indice_tensor));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_destroy_tensor(scatter_indice_tensor));

  WM_CUDA_DEBUG_SYNC_STREAM(stream);

  return WHOLEMEMORY_SUCCESS;
}

__global__ void WriteBackCacheDirectKernel(uint16_t* local_cache_line_tag,
                                           int4* local_cached_data,
                                           int4* local_memory_data,
                                           int embedding_dim_in_int4,
                                           int cache_set_coverage,
                                           bool drop_all)
{
  static_assert(wholememory::embedding_cache_base::kCacheSetSize == 32);
  int64_t const cache_set_lid = blockIdx.x;
  local_cache_line_tag += cache_set_lid * wholememory::embedding_cache_base::kCacheSetSize;
  local_cached_data +=
    cache_set_lid * wholememory::embedding_cache_base::kCacheSetSize * embedding_dim_in_int4;
  local_memory_data += cache_set_lid * cache_set_coverage * embedding_dim_in_int4;
  CacheLineInfo cache_line_info;
  cache_line_info.LoadTag(local_cache_line_tag);

  bool is_modified = cache_line_info.IsModified() && cache_line_info.IsValid();
  auto modify_mask = __ballot_sync(0xFFFFFFFF, static_cast<int>(is_modified));
  while (modify_mask != 0) {
    int lane_idx = __ffs(modify_mask) - 1;
    int local_id = __shfl_sync(0xFFFFFFFF, cache_line_info.LocalID(), lane_idx, 32);
    for (int idx = threadIdx.x; idx < embedding_dim_in_int4; idx += 32) {
      local_memory_data[local_id * embedding_dim_in_int4 + idx] =
        local_cached_data[lane_idx * embedding_dim_in_int4 + idx];
    }
    modify_mask ^= (1 << lane_idx);
  }

  if (drop_all) {
    cache_line_info.ClearCacheLine();
  } else {
    cache_line_info.ClearModify();
  }
  cache_line_info.StoreTag(local_cache_line_tag);
}

wholememory_error_code_t writeback_cache_direct_same_comm(
  wholememory_tensor_t wm_raw_memory_embedding,
  const wholememory::embedding_cache_local_data* cache_local_data,
  int cache_set_coverage,
  bool drop_all,
  cudaStream_t stream)
{
  int world_size = 1;
  int world_rank = 0;
  wholememory_handle_t wholememory_handle =
    wholememory_tensor_get_memory_handle(wm_raw_memory_embedding);
  wholememory_comm_t wm_comm;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, wholememory_handle));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&world_rank, wm_comm));

  auto* raw_embedding_desc =
    wholememory_tensor_get_tensor_description(wholememory_tensor_get_root(wm_raw_memory_embedding));
  size_t embedding_entry_count_per_rank = 0;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_determine_entry_partition_plan(
    &embedding_entry_count_per_rank, raw_embedding_desc->sizes[0], world_size));

  WHOLEMEMORY_CHECK_NOTHROW(embedding_entry_count_per_rank % cache_set_coverage == 0);
  wholememory_tensor_t raw_local_tensor;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_tensor_map_local_tensor(wm_raw_memory_embedding, &raw_local_tensor));
  int cache_set_count = wholememory::div_rounding_up_unsafe(
    wholememory_tensor_get_tensor_description(raw_local_tensor)->sizes[0], cache_set_coverage);

  int const embedding_dim = raw_embedding_desc->strides[0];
  size_t const dtype_size = wholememory_dtype_get_element_size(raw_embedding_desc->dtype);
  WHOLEMEMORY_CHECK_NOTHROW(embedding_dim * dtype_size % 16 == 0);
  int const embedding_dim_in_int4 = embedding_dim * dtype_size / 16;

  if (cache_set_count > 0) {
    WriteBackCacheDirectKernel<<<cache_set_count, 32, 0, stream>>>(
      static_cast<uint16_t*>(
        wholememory_tensor_get_data_pointer(cache_local_data->cache_line_tag_)),
      static_cast<int4*>(wholememory_tensor_get_data_pointer(cache_local_data->cache_line_data_)),
      static_cast<int4*>(wholememory_tensor_get_data_pointer(raw_local_tensor)),
      embedding_dim_in_int4,
      cache_set_coverage,
      drop_all);
    WM_CUDA_CHECK_NO_THROW(cudaGetLastError());
  }

  WM_CUDA_DEBUG_SYNC_STREAM(stream);

  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_destroy_tensor(raw_local_tensor));

  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
