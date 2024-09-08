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
#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "logger.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/memory_handle.hpp"
#include "wholememory_ops/functions/bucket_ids_for_hierarchy_func.h"
#include "wholememory_ops/functions/exchange_embeddings_nccl_func.h"
#include "wholememory_ops/functions/gather_scatter_func.h"
#include "wholememory_ops/functions/sort_unique_ids_for_hierarchy_func.h"
#include "wholememory_ops/gather_op_impl.h"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

namespace wholememory_ops {

static wholememory_error_code_t wholememory_cross_gather(
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  int64_t* host_bucket_id_count_ptr,
  wholememory_comm_t wm_local_comm,
  wholememory_comm_t wm_cross_comm,
  wm_thrust_allocator* p_thrust_allocator,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms)
{
  int cross_size;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&cross_size, wm_cross_comm));
  std::vector<int64_t> host_bucket_id_offset(cross_size);
  std::vector<int64_t> host_recv_id_count(cross_size, 0);
  std::vector<int64_t> host_recv_id_offset(cross_size);
  // exchange node count
  wm_cross_comm->host_alltoall(
    host_bucket_id_count_ptr, host_recv_id_count.data(), 1, WHOLEMEMORY_DT_INT64);
  host_bucket_id_offset[0] = 0;
  for (int i = 1; i < cross_size; i++)
    host_bucket_id_offset[i] = host_bucket_id_offset[i - 1] + host_bucket_id_count_ptr[i - 1];
  wm_cross_comm->sync_stream();
  // exchange indices
  int64_t total_recv_count = 0;
  for (int i = 0; i < cross_size; i++) {
    host_recv_id_offset[i] = total_recv_count;
    total_recv_count += host_recv_id_count[i];
  }
  temp_memory_handle dev_recv_bucket_indices_handle(p_env_fns);
  void* dev_recv_bucket_indices_ptr =
    dev_recv_bucket_indices_handle.device_malloc(total_recv_count, indice_desc.dtype);
  wm_cross_comm->alltoallv(indices,
                           dev_recv_bucket_indices_ptr,
                           reinterpret_cast<const size_t*>(host_bucket_id_count_ptr),
                           reinterpret_cast<const size_t*>(host_bucket_id_offset.data()),
                           reinterpret_cast<const size_t*>(host_recv_id_count.data()),
                           reinterpret_cast<const size_t*>(host_recv_id_offset.data()),
                           indice_desc.dtype,
                           stream);
  wm_cross_comm->sync_stream(stream);
  // local gather
  temp_memory_handle dev_local_gather_buffer_handle(p_env_fns);
  void* dev_local_gather_buffer_ptr = dev_local_gather_buffer_handle.device_malloc(
    wholememory_desc.sizes[1] * total_recv_count, output_desc.dtype);
  int64_t local_gather_buffer_size[2] = {total_recv_count, wholememory_desc.sizes[1]};
  wholememory_matrix_description_t local_gather_buffer_desc = wholememory_create_matrix_desc(
    local_gather_buffer_size, wholememory_desc.sizes[1], 0, output_desc.dtype);
  void* local_fake_ptr = nullptr;
  size_t local_mem_offset, local_mem_size;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_local_memory(
    &local_fake_ptr, &local_mem_size, &local_mem_offset, wholememory_handle));
  local_fake_ptr = static_cast<char*>(local_fake_ptr) - local_mem_offset;
  wholememory_gref_t local_fake_gref =
    wholememory_create_continuous_global_reference(local_fake_ptr);
  auto local_gather_indice_desc =
    wholememory_create_array_desc(total_recv_count, 0, indice_desc.dtype);
  WHOLEMEMORY_RETURN_ON_FAIL(gather_func(local_fake_gref,
                                         wholememory_desc,
                                         dev_recv_bucket_indices_ptr,
                                         local_gather_indice_desc,
                                         dev_local_gather_buffer_ptr,
                                         local_gather_buffer_desc,
                                         stream,
                                         gather_sms));
  // exchange embeddings
  size_t output_embedding_size =
    wholememory_desc.sizes[1] * wholememory_dtype_get_element_size(output_desc.dtype);
  WHOLEMEMORY_RETURN_ON_FAIL(exchange_embeddings_nccl_func(dev_local_gather_buffer_ptr,
                                                           host_recv_id_count.data(),
                                                           host_bucket_id_count_ptr,
                                                           output,
                                                           output_embedding_size,
                                                           wm_cross_comm,
                                                           stream));
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_gather_hierarchy(
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms)
{
  try {
    if (wholememory_desc.storage_offset < 0 ||
        wholememory_desc.storage_offset + wholememory_desc.sizes[1] > wholememory_desc.stride) {
      return WHOLEMEMORY_INVALID_INPUT;
    }
    bool sort_unique_indices = true;

    wm_thrust_allocator thrust_allocator(p_env_fns);

    wholememory_comm_t wm_global_comm;
    int world_size, world_rank;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_global_comm, wholememory_handle));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_global_comm));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&world_rank, wm_global_comm));

    wholememory_comm_t wm_local_comm;
    int local_size, local_rank;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_local_communicator(&wm_local_comm, wholememory_handle));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&local_size, wm_local_comm));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&local_rank, wm_local_comm));

    wholememory_comm_t wm_cross_comm;
    int cross_size;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_cross_communicator(&wm_cross_comm, wholememory_handle));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&cross_size, wm_cross_comm));
    WHOLEMEMORY_CHECK_NOTHROW(world_size == local_size * cross_size);

    size_t element_size         = wholememory_dtype_get_element_size(wholememory_desc.dtype);
    size_t embedding_entry_size = element_size * wholememory_desc.stride;
    temp_memory_handle dev_embedding_entry_offsets_handle(p_env_fns);
    size_t* dev_embedding_entry_offsets_ptr = static_cast<size_t*>(
      dev_embedding_entry_offsets_handle.device_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));
    std::vector<size_t> host_embedding_entry_offsets(world_size + 1);
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_rank_partition_offsets(
      host_embedding_entry_offsets.data(), wholememory_handle));
    for (int i = 0; i < world_size + 1; i++) {
      size_t offset = host_embedding_entry_offsets[i];
      WHOLEMEMORY_EXPECTS_NOTHROW(
        offset % embedding_entry_size == 0,
        "embedding memory offset of rank%d=%ld is not multiple of embedding_entry_size=%ldx%ld",
        i,
        offset,
        element_size,
        wholememory_desc.stride);
      host_embedding_entry_offsets[i] /= embedding_entry_size;
    }

    WM_CUDA_CHECK(cudaMemcpyAsync(dev_embedding_entry_offsets_ptr,
                                  host_embedding_entry_offsets.data(),
                                  (world_size + 1) * sizeof(size_t),
                                  cudaMemcpyHostToDevice,
                                  stream));

    temp_memory_handle dev_bucket_indices_handle(p_env_fns);
    void* dev_bucket_indices_ptr =
      dev_bucket_indices_handle.device_malloc(indice_desc.size, indice_desc.dtype);
    temp_memory_handle dev_bucket_ids_map_handle(p_env_fns);
    void* dev_bucket_ids_map_ptr =
      dev_bucket_ids_map_handle.device_malloc(indice_desc.size, indice_desc.dtype);

    std::vector<int64_t> host_bucket_id_count(local_size, 0);
    std::vector<int64_t> host_bucket_id_offset(local_size);
    std::vector<int64_t> host_recv_id_count(local_size, 0);
    std::vector<int64_t> host_recv_id_offset(local_size);

    // bucket indices
    WHOLEMEMORY_RETURN_ON_FAIL(
      bucket_and_reorder_ids_for_hierarchy_func(indices,
                                                indice_desc,
                                                dev_bucket_indices_ptr,
                                                dev_bucket_ids_map_ptr,
                                                host_bucket_id_count.data(),
                                                dev_embedding_entry_offsets_ptr,
                                                wm_global_comm,
                                                wm_local_comm,
                                                0,
                                                &thrust_allocator,
                                                p_env_fns,
                                                stream));
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    // exchange node count
    wm_local_comm->host_alltoall(
      host_bucket_id_count.data(), host_recv_id_count.data(), 1, WHOLEMEMORY_DT_INT64);
    host_bucket_id_offset[0] = 0;
    for (int i = 1; i < local_size; i++)
      host_bucket_id_offset[i] = host_bucket_id_offset[i - 1] + host_bucket_id_count[i - 1];
    wm_local_comm->sync_stream();
    // exchange indices
    int64_t total_recv_count = 0;
    for (int i = 0; i < local_size; i++) {
      host_recv_id_offset[i] = total_recv_count;
      total_recv_count += host_recv_id_count[i];
    }
    temp_memory_handle dev_recv_bucket_indices_handle(p_env_fns);
    void* dev_recv_bucket_indices_ptr =
      dev_recv_bucket_indices_handle.device_malloc(total_recv_count, indice_desc.dtype);
    auto recv_bucket_indices_desc =
      wholememory_create_array_desc(total_recv_count, 0, indice_desc.dtype);
    wm_local_comm->alltoallv(dev_bucket_indices_ptr,
                             dev_recv_bucket_indices_ptr,
                             reinterpret_cast<const size_t*>(host_bucket_id_count.data()),
                             reinterpret_cast<const size_t*>(host_bucket_id_offset.data()),
                             reinterpret_cast<const size_t*>(host_recv_id_count.data()),
                             reinterpret_cast<const size_t*>(host_recv_id_offset.data()),
                             indice_desc.dtype,
                             stream);
    wm_local_comm->sync_stream(stream);
    WM_CUDA_CHECK(cudaGetLastError());
    // sort unique / bucket recv indices
    temp_memory_handle cross_gather_indices_handle(p_env_fns);
    wholememory_array_description_t cross_gather_indices_desc;
    temp_memory_handle dev_cross_gather_id_map_handle(p_env_fns);
    std::vector<int64_t> host_cross_bucket_id_count(cross_size, 0);
    if (sort_unique_indices) {
      sort_unique_ids_for_hierarchy_func(dev_recv_bucket_indices_ptr,
                                         recv_bucket_indices_desc,
                                         &cross_gather_indices_handle,
                                         &cross_gather_indices_desc,
                                         &dev_cross_gather_id_map_handle,
                                         &thrust_allocator,
                                         p_env_fns,
                                         stream);
      bucket_local_ids_func(cross_gather_indices_handle.pointer(),
                            cross_gather_indices_desc,
                            host_cross_bucket_id_count.data(),
                            dev_embedding_entry_offsets_ptr,
                            wm_local_comm,
                            wm_cross_comm,
                            &thrust_allocator,
                            p_env_fns,
                            stream);
    } else {
      void* cross_gather_indices_ptr = cross_gather_indices_handle.device_malloc(
        recv_bucket_indices_desc.size, recv_bucket_indices_desc.dtype);
      void* dev_cross_gather_id_map_ptr = dev_cross_gather_id_map_handle.device_malloc(
        recv_bucket_indices_desc.size, recv_bucket_indices_desc.dtype);
      cross_gather_indices_desc = recv_bucket_indices_desc;
      WHOLEMEMORY_RETURN_ON_FAIL(
        bucket_and_reorder_ids_for_hierarchy_func(dev_recv_bucket_indices_ptr,
                                                  recv_bucket_indices_desc,
                                                  cross_gather_indices_ptr,
                                                  dev_cross_gather_id_map_ptr,
                                                  host_cross_bucket_id_count.data(),
                                                  dev_embedding_entry_offsets_ptr,
                                                  wm_global_comm,
                                                  wm_local_comm,
                                                  1,
                                                  &thrust_allocator,
                                                  p_env_fns,
                                                  stream));
    }
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    // cross gather
    temp_memory_handle dev_cross_gather_buffer_handle(p_env_fns);
    void* dev_cross_gather_buffer_ptr = dev_cross_gather_buffer_handle.device_malloc(
      wholememory_desc.sizes[1] * cross_gather_indices_desc.size, output_desc.dtype);
    int64_t cross_gather_buffer_size[2]                       = {cross_gather_indices_desc.size,
                                                                 wholememory_desc.sizes[1]};
    wholememory_matrix_description_t cross_gather_buffer_desc = wholememory_create_matrix_desc(
      cross_gather_buffer_size, wholememory_desc.sizes[1], 0, output_desc.dtype);
    wholememory_cross_gather(wholememory_handle,
                             wholememory_desc,
                             cross_gather_indices_handle.pointer(),
                             cross_gather_indices_desc,
                             dev_cross_gather_buffer_ptr,
                             cross_gather_buffer_desc,
                             host_cross_bucket_id_count.data(),
                             wm_local_comm,
                             wm_cross_comm,
                             &thrust_allocator,
                             p_env_fns,
                             stream,
                             gather_sms);
    // cross gather reorder
    temp_memory_handle dev_embedding_map_buffer_handle(p_env_fns);
    void* dev_embedding_map_buffer_ptr = dev_embedding_map_buffer_handle.device_malloc(
      wholememory_desc.sizes[1] * total_recv_count, output_desc.dtype);
    int64_t embedding_map_buffer_size[2] = {total_recv_count, wholememory_desc.sizes[1]};
    wholememory_matrix_description_t embedding_map_buffer_desc = wholememory_create_matrix_desc(
      embedding_map_buffer_size, wholememory_desc.sizes[1], 0, output_desc.dtype);
    wholememory_gref_t cross_gather_fake_gref =
      wholememory_create_continuous_global_reference(dev_cross_gather_buffer_ptr);
    WHOLEMEMORY_RETURN_ON_FAIL(gather_func(cross_gather_fake_gref,
                                           cross_gather_buffer_desc,
                                           dev_cross_gather_id_map_handle.pointer(),
                                           recv_bucket_indices_desc,
                                           dev_embedding_map_buffer_ptr,
                                           embedding_map_buffer_desc,
                                           stream,
                                           gather_sms));
    // exchange embeddings
    size_t output_embedding_size =
      wholememory_desc.sizes[1] * wholememory_dtype_get_element_size(output_desc.dtype);
    temp_memory_handle dev_recv_embedding_buffer_handle(p_env_fns);
    void* dev_recv_embedding_buffer_ptr = dev_recv_embedding_buffer_handle.device_malloc(
      wholememory_desc.sizes[1] * indice_desc.size, output_desc.dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(exchange_embeddings_nccl_func(dev_embedding_map_buffer_ptr,
                                                             host_recv_id_count.data(),
                                                             host_bucket_id_count.data(),
                                                             dev_recv_embedding_buffer_ptr,
                                                             output_embedding_size,
                                                             wm_local_comm,
                                                             stream));
    // bucket reorder
    wholememory_gref_t recv_embedding_buffer_fake_gref =
      wholememory_create_continuous_global_reference(dev_recv_embedding_buffer_ptr);
    int64_t recv_embedding_buffer_size[2] = {indice_desc.size, wholememory_desc.sizes[1]};
    wholememory_matrix_description_t recv_embedding_buffer_desc = wholememory_create_matrix_desc(
      recv_embedding_buffer_size, wholememory_desc.sizes[1], 0, output_desc.dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(gather_func(recv_embedding_buffer_fake_gref,
                                           recv_embedding_buffer_desc,
                                           dev_bucket_ids_map_ptr,
                                           indice_desc,
                                           output,
                                           output_desc,
                                           stream,
                                           gather_sms));
    WM_CUDA_CHECK(cudaGetLastError());
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("CUDA logic Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }

  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
