/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include "exchange_ids_nccl_func.h"

#include <cub/device/device_radix_sort.cuh>
#include <thrust/sequence.h>

#include "bucket_ids_func.h"
#include "error.hpp"
#include "logger.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory_ops/register.hpp"

namespace wholememory_ops {

template <typename IndexT>
struct UnsignedType {};

template <>
struct UnsignedType<int> {
  using UType = unsigned int;
};

template <>
struct UnsignedType<int64_t> {
  using UType = uint64_t;
};

template <typename IndexT>
void exchange_ids_temp_func(const void* indices_before_sort,
                            wholememory_array_description_t indices_desc,
                            const int64_t* host_recv_rank_id_count_ptr,
                            const int64_t* host_rank_id_count_ptr,
                            const int64_t* host_rank_id_offset_ptr,
                            temp_memory_handle* dev_recv_indice_buffer,
                            void* indices_after_sort,
                            int64_t* raw_indices,
                            wholememory_comm_t wm_comm,
                            wm_thrust_allocator* p_thrust_allocator,
                            cudaStream_t stream)
{
  auto index_type = indices_desc.dtype;
  WHOLEMEMORY_CHECK(indices_desc.storage_offset == 0);
  WHOLEMEMORY_CHECK(index_type == WHOLEMEMORY_DT_INT || index_type == WHOLEMEMORY_DT_INT64);
  wm_thrust_allocator& allocator = *p_thrust_allocator;

  int64_t* seq_indices = reinterpret_cast<int64_t*>(allocator.allocate(
    wholememory_get_memory_element_count_from_array(&indices_desc) * sizeof(int64_t)));
  thrust::sequence(thrust::cuda::par_nosync(allocator).on(stream),
                   seq_indices,
                   seq_indices + indices_desc.size,
                   0);
  // use UTypeT to put minus indices at last.
  using UTypeT                  = typename UnsignedType<IndexT>::UType;
  const UTypeT* indices_to_sort = static_cast<const UTypeT*>(indices_before_sort);
  UTypeT* sorted_indice         = static_cast<UTypeT*>(indices_after_sort);
  void* cub_temp_storage        = nullptr;
  size_t temp_storage_bytes     = 0;
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indices_to_sort,
                                  sorted_indice,
                                  seq_indices,
                                  raw_indices,
                                  indices_desc.size,
                                  0,
                                  sizeof(UTypeT) * 8,
                                  stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indices_to_sort,
                                  sorted_indice,
                                  seq_indices,
                                  raw_indices,
                                  indices_desc.size,
                                  0,
                                  sizeof(UTypeT) * 8,
                                  stream);
  int64_t total_recv_count = 0;
  int world_size;
  WHOLEMEMORY_CHECK(wholememory_communicator_get_size(&world_size, wm_comm) == WHOLEMEMORY_SUCCESS);
  std::vector<size_t> host_recv_offset(world_size);
  for (int i = 0; i < world_size; i++) {
    host_recv_offset[i] = total_recv_count;
    total_recv_count += host_recv_rank_id_count_ptr[i];
  }
  IndexT* dev_recv_indice_buffer_ptr =
    static_cast<IndexT*>(dev_recv_indice_buffer->device_malloc(total_recv_count, index_type));
  wm_comm->alltoallv(sorted_indice,
                     dev_recv_indice_buffer_ptr,
                     reinterpret_cast<const size_t*>(host_rank_id_count_ptr),
                     reinterpret_cast<const size_t*>(host_rank_id_offset_ptr),
                     reinterpret_cast<const size_t*>(host_recv_rank_id_count_ptr),
                     host_recv_offset.data(),
                     index_type,
                     stream);
  wm_comm->sync_stream(stream);
  allocator.deallocate(reinterpret_cast<char*>(seq_indices),
                       wholememory_get_memory_size_from_array(&indices_desc));
  allocator.deallocate(static_cast<char*>(cub_temp_storage), temp_storage_bytes);
}

REGISTER_DISPATCH_ONE_TYPE(ExchangeIDsNCCL, exchange_ids_temp_func, SINT3264)

wholememory_error_code_t exchange_ids_func(const void* indices_before_sort,
                                           wholememory_array_description_t indices_desc,
                                           const int64_t* host_recv_rank_id_count_ptr,
                                           const int64_t* host_rank_id_count_ptr,
                                           const int64_t* host_rank_id_offset_ptr,
                                           temp_memory_handle* dev_recv_indices_buffer_handle,
                                           void* indices_after_sort,
                                           int64_t* raw_indices,
                                           wholememory_comm_t wm_comm,
                                           wm_thrust_allocator* p_thrust_allocator,
                                           cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(indices_desc.dtype,
                      ExchangeIDsNCCL,
                      indices_before_sort,
                      indices_desc,
                      host_recv_rank_id_count_ptr,
                      host_rank_id_count_ptr,
                      host_rank_id_offset_ptr,
                      dev_recv_indices_buffer_handle,
                      indices_after_sort,
                      raw_indices,
                      wm_comm,
                      p_thrust_allocator,
                      stream);
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("exchange_ids_func CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("exchange_ids_func LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t bucket_and_exchange_ids_func(
  void* indices,
  wholememory_array_description_t indice_desc,
  int64_t* host_recv_rank_id_count_ptr,
  int64_t* host_rank_id_count_ptr,
  temp_memory_handle* dev_recv_indices_buffer_handle,
  int64_t* dev_raw_indice_ptr,
  size_t embedding_entry_count_per_rank,
  wholememory_comm_t wm_comm,
  wm_thrust_allocator* p_thrust_allocator,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  int world_size;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));

  // Exchange node count
  temp_memory_handle dev_rank_id_count(p_env_fns);
  int64_t* dev_rank_id_count_ptr =
    static_cast<int64_t*>(dev_rank_id_count.device_malloc(world_size, WHOLEMEMORY_DT_INT64));

  WHOLEMEMORY_RETURN_ON_FAIL(bucket_ids_for_ranks(indices,
                                                  indice_desc,
                                                  dev_rank_id_count_ptr,
                                                  embedding_entry_count_per_rank,
                                                  world_size,
                                                  get_device_prop(-1),
                                                  stream));

  WM_CUDA_CHECK(cudaGetLastError());

  temp_memory_handle host_rank_id_offset(p_env_fns);
  temp_memory_handle dev_sorted_indice(p_env_fns);
  int64_t* host_rank_id_offset_ptr =
    static_cast<int64_t*>(host_rank_id_offset.host_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));
  void* dev_sorted_indice_ptr =
    dev_sorted_indice.device_malloc(indice_desc.size, indice_desc.dtype);
  WM_CUDA_CHECK(cudaMemcpyAsync(host_rank_id_count_ptr,
                                dev_rank_id_count_ptr,
                                sizeof(int64_t) * world_size,
                                cudaMemcpyDeviceToHost,
                                stream));
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  wm_comm->host_alltoall(
    host_rank_id_count_ptr, host_recv_rank_id_count_ptr, 1, WHOLEMEMORY_DT_INT64);
  host_rank_id_offset_ptr[0] = 0;
  for (int i = 0; i < world_size; i++) {
    host_rank_id_offset_ptr[i + 1] = host_rank_id_offset_ptr[i] + host_rank_id_count_ptr[i];
  }
  WHOLEMEMORY_EXPECTS(wm_comm->sync_stream() == WHOLEMEMORY_SUCCESS,
                      "Rank id count AllToAll failed.");
  void* indice_ptr =
    static_cast<char*>(indices) +
    wholememory_dtype_get_element_size(indice_desc.dtype) * indice_desc.storage_offset;
  // Exchange ids
  WHOLEMEMORY_RETURN_ON_FAIL(exchange_ids_func(indice_ptr,
                                               indice_desc,
                                               host_recv_rank_id_count_ptr,
                                               host_rank_id_count_ptr,
                                               host_rank_id_offset_ptr,
                                               dev_recv_indices_buffer_handle,
                                               dev_sorted_indice_ptr,
                                               dev_raw_indice_ptr,
                                               wm_comm,
                                               p_thrust_allocator,
                                               stream));
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
