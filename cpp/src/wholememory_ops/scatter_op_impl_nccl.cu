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
#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "logger.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/memory_handle.hpp"
#include "wholememory_ops/functions/bucket_ids_func.h"
#include "wholememory_ops/functions/exchange_embeddings_nccl_func.h"
#include "wholememory_ops/functions/exchange_ids_nccl_func.h"
#include "wholememory_ops/functions/gather_scatter_func.h"
#include "wholememory_ops/scatter_op_impl.h"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

namespace wholememory_ops {

wholememory_error_code_t wholememory_scatter_nccl(void* input,
                                                  wholememory_matrix_description_t input_desc,
                                                  void* indices,
                                                  wholememory_array_description_t indices_desc,
                                                  wholememory_handle_t wholememory_handle,
                                                  wholememory_matrix_description_t wholememory_desc,
                                                  wholememory_env_func_t* p_env_fns,
                                                  cudaStream_t stream,
                                                  int scatter_sms)
{
  try {
    if (wholememory_desc.storage_offset < 0 ||
        wholememory_desc.storage_offset + wholememory_desc.sizes[1] > wholememory_desc.stride) {
      WHOLEMEMORY_ERROR("invalid input offset=%ld, size[1]=%ld, stride=%ld\n",
                        wholememory_desc.storage_offset,
                        wholememory_desc.sizes[1],
                        wholememory_desc.stride);
      return WHOLEMEMORY_INVALID_INPUT;
    }

    wm_thrust_allocator thrust_allocator(p_env_fns);

    size_t element_size         = wholememory_dtype_get_element_size(wholememory_desc.dtype);
    size_t embedding_entry_size = element_size * wholememory_desc.stride;

    wholememory_comm_t wm_comm;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, wholememory_handle));

    int world_size;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));

    temp_memory_handle host_rank_id_count(p_env_fns), host_recv_rank_id_count(p_env_fns);
    int64_t* host_rank_id_count_ptr =
      static_cast<int64_t*>(host_rank_id_count.host_malloc(world_size, WHOLEMEMORY_DT_INT64));
    int64_t* host_recv_rank_id_count_ptr =
      static_cast<int64_t*>(host_recv_rank_id_count.host_malloc(world_size, WHOLEMEMORY_DT_INT64));

    temp_memory_handle dev_recv_indice_buffer(p_env_fns);
    temp_memory_handle dev_raw_indice(p_env_fns);
    int64_t* dev_raw_indice_ptr =
      static_cast<int64_t*>(dev_raw_indice.device_malloc(indices_desc.size, WHOLEMEMORY_DT_INT64));

    int64_t total_recv_count = 0;

    temp_memory_handle dev_embedding_entry_offsets_handle(p_env_fns);
    size_t* dev_embedding_entry_offsets_ptr = static_cast<size_t*>(
      dev_embedding_entry_offsets_handle.device_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));
    temp_memory_handle host_embedding_entry_offsets_handle(p_env_fns);
    size_t* host_embedding_entry_offsets_ptr = static_cast<size_t*>(
      host_embedding_entry_offsets_handle.host_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));

    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_rank_partition_offsets(host_embedding_entry_offsets_ptr, wholememory_handle));
    for (int i = 0; i < world_size + 1; i++) {
      size_t offset = host_embedding_entry_offsets_ptr[i];
      WHOLEMEMORY_EXPECTS_NOTHROW(
        offset % embedding_entry_size == 0,
        "embedding memory offset of rank%d=%ld is not multiple of embedding_entry_size=%ldx%ld",
        i,
        offset,
        element_size,
        wholememory_desc.stride);
      host_embedding_entry_offsets_ptr[i] /= embedding_entry_size;
    }
    WM_CUDA_CHECK(cudaMemcpyAsync(dev_embedding_entry_offsets_ptr,
                                  host_embedding_entry_offsets_ptr,
                                  (world_size + 1) * sizeof(size_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    WHOLEMEMORY_RETURN_ON_FAIL(bucket_and_exchange_ids_func(indices,
                                                            indices_desc,
                                                            host_recv_rank_id_count_ptr,
                                                            host_rank_id_count_ptr,
                                                            &dev_recv_indice_buffer,
                                                            dev_raw_indice_ptr,
                                                            dev_embedding_entry_offsets_ptr,
                                                            wm_comm,
                                                            &thrust_allocator,
                                                            p_env_fns,
                                                            stream));

    // Local Reorder
    for (int i = 0; i < world_size; i++) {
      total_recv_count += host_recv_rank_id_count_ptr[i];
    }
    temp_memory_handle dev_local_reorder_buffer(p_env_fns), dev_embedding_recv_buffer(p_env_fns);
    auto local_reorder_desc =
      wholememory_create_matrix_desc(input_desc.sizes, input_desc.sizes[1], 0, input_desc.dtype);
    void* dev_local_reorder_buffer_ptr = dev_local_reorder_buffer.device_malloc(
      wholememory_get_memory_element_count_from_matrix(&local_reorder_desc), input_desc.dtype);
    wholememory_gref_t input_gref = wholememory_create_continuous_global_reference(input);
    auto dev_raw_indice_desc =
      wholememory_create_array_desc(indices_desc.size, 0, WHOLEMEMORY_DT_INT64);
    WHOLEMEMORY_RETURN_ON_FAIL(gather_func(input_gref,
                                           input_desc,
                                           dev_raw_indice_ptr,
                                           dev_raw_indice_desc,
                                           dev_local_reorder_buffer_ptr,
                                           local_reorder_desc,
                                           stream));
    // AllToAllV for embeddings
    void* dev_embedding_recv_buffer_ptr = dev_embedding_recv_buffer.device_malloc(
      total_recv_count * input_desc.sizes[1], input_desc.dtype);
    size_t embedding_size =
      wholememory_desc.sizes[1] * wholememory_dtype_get_element_size(input_desc.dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(exchange_embeddings_nccl_func(dev_local_reorder_buffer_ptr,
                                                             host_rank_id_count_ptr,
                                                             host_recv_rank_id_count_ptr,
                                                             dev_embedding_recv_buffer_ptr,
                                                             embedding_size,
                                                             wm_comm,
                                                             stream));
    // Local scatter
    size_t local_mem_offset, local_mem_size;
    void* local_fake_ptr = nullptr;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_local_memory(
      &local_fake_ptr, &local_mem_size, &local_mem_offset, wholememory_handle));
    local_fake_ptr = static_cast<char*>(local_fake_ptr) - local_mem_offset;
    wholememory_gref_t local_fake_embedding_gref =
      wholememory_create_continuous_global_reference(local_fake_ptr);

    std::vector<int64_t> recv_embedding_sizes            = {total_recv_count, input_desc.sizes[1]};
    wholememory_matrix_description_t recv_embedding_desc = wholememory_create_matrix_desc(
      recv_embedding_sizes.data(), input_desc.sizes[1], 0, input_desc.dtype);
    auto recv_indices_desc = wholememory_create_array_desc(total_recv_count, 0, indices_desc.dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(scatter_func(dev_embedding_recv_buffer_ptr,
                                            recv_embedding_desc,
                                            dev_recv_indice_buffer.pointer(),
                                            recv_indices_desc,
                                            local_fake_embedding_gref,
                                            wholememory_desc,
                                            stream,
                                            scatter_sms));
    WM_CUDA_CHECK(cudaGetLastError());
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("CUDA logic Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    WHOLEMEMORY_ERROR("Unknown Error\n");
    return WHOLEMEMORY_UNKNOW_ERROR;
  }

  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_scatter_distributed(
  void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms)
{
#ifdef WITH_NVSHMEM_SUPPORT
  if (wholememory_get_distributed_backend(wholememory_handle) == WHOLEMEMORY_DB_NVSHMEM) {
    return wholememory_scatter_nvshmem(input,
                                       input_desc,
                                       indices,
                                       indices_desc,
                                       wholememory_handle,
                                       wholememory_desc,
                                       p_env_fns,
                                       stream,
                                       scatter_sms);
  }
#endif

  return wholememory_scatter_nccl(input,
                                  input_desc,
                                  indices,
                                  indices_desc,
                                  wholememory_handle,
                                  wholememory_desc,
                                  p_env_fns,
                                  stream,
                                  scatter_sms);
}
}  // namespace wholememory_ops
