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
#include "wholememory_ops/gather_op_impl.h"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

namespace wholememory_ops {

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

    wm_thrust_allocator thrust_allocator(p_env_fns);

    size_t embedding_size_per_rank;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_partition_plan(&embedding_size_per_rank, wholememory_handle));

    size_t element_size         = wholememory_dtype_get_element_size(wholememory_desc.dtype);
    size_t embedding_entry_size = element_size * wholememory_desc.stride;

    WHOLEMEMORY_EXPECTS_NOTHROW(
      embedding_size_per_rank % embedding_entry_size == 0,
      "embedding_size_per_rank=%ld is not multiple of embedding_entry_size=%ldx%ld",
      embedding_size_per_rank,
      element_size,
      wholememory_desc.stride);

    size_t embedding_entry_count_per_rank = embedding_size_per_rank / embedding_entry_size;

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
      static_cast<int64_t*>(dev_raw_indice.device_malloc(indice_desc.size, WHOLEMEMORY_DT_INT64));

    int64_t total_recv_count = 0;
    WHOLEMEMORY_RETURN_ON_FAIL(bucket_and_exchange_ids_func(indices,
                                                            indice_desc,
                                                            host_recv_rank_id_count_ptr,
                                                            host_rank_id_count_ptr,
                                                            &dev_recv_indice_buffer,
                                                            dev_raw_indice_ptr,
                                                            embedding_entry_count_per_rank,
                                                            wm_comm,
                                                            &thrust_allocator,
                                                            p_env_fns,
                                                            stream));

    // Local Gather
    for (int i = 0; i < world_size; i++) {
      total_recv_count += host_recv_rank_id_count_ptr[i];
    }
    size_t local_mem_offset, local_mem_size;
    temp_memory_handle dev_local_gather_buffer(p_env_fns);
    temp_memory_handle dev_embedding_recv_buffer(p_env_fns);
    void* dev_local_gather_buffer_ptr = dev_local_gather_buffer.device_malloc(
      wholememory_desc.sizes[1] * total_recv_count, output_desc.dtype);
    void* dev_embedding_recv_buffer_ptr = dev_embedding_recv_buffer.device_malloc(
      wholememory_desc.sizes[1] * indice_desc.size, output_desc.dtype);
    void* local_fake_ptr = nullptr;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_local_memory(
      &local_fake_ptr, &local_mem_size, &local_mem_offset, wholememory_handle));
    local_fake_ptr = static_cast<char*>(local_fake_ptr) - local_mem_offset;
    wholememory_gref_t local_fake_gref =
      wholememory_create_continuous_global_reference(local_fake_ptr);
    int64_t local_buffer_size[2] = {total_recv_count, wholememory_desc.sizes[1]};
    wholememory_matrix_description_t local_gather_buffer_desc = wholememory_create_matrix_desc(
      local_buffer_size, wholememory_desc.sizes[1], 0, output_desc.dtype);
    auto dev_recv_indice_desc =
      wholememory_create_array_desc(total_recv_count, 0, indice_desc.dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(gather_func(local_fake_gref,
                                           wholememory_desc,
                                           dev_recv_indice_buffer.pointer(),
                                           dev_recv_indice_desc,
                                           dev_local_gather_buffer_ptr,
                                           local_gather_buffer_desc,
                                           stream,
                                           gather_sms));
    // AllToAllV for embeddings
    size_t embedding_size =
      wholememory_desc.sizes[1] * wholememory_dtype_get_element_size(output_desc.dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(exchange_embeddings_nccl_func(dev_local_gather_buffer_ptr,
                                                             host_recv_rank_id_count_ptr,
                                                             host_rank_id_count_ptr,
                                                             dev_embedding_recv_buffer_ptr,
                                                             embedding_size,
                                                             wm_comm,
                                                             stream));
    // Local reorder
    int64_t total_need_indice_count = 0;
    for (int i = 0; i < world_size; i++) {
      total_need_indice_count += host_rank_id_count_ptr[i];
    }
    wholememory_gref_t output_gref = wholememory_create_continuous_global_reference(output);
    wholememory_matrix_description_t local_recv_buffer_desc =
      wholememory_create_matrix_desc(output_desc.sizes, output_desc.sizes[1], 0, output_desc.dtype);
    local_recv_buffer_desc.sizes[0] = total_need_indice_count;
    auto raw_indice_desc =
      wholememory_create_array_desc(total_need_indice_count, 0, WHOLEMEMORY_DT_INT64);
    WHOLEMEMORY_RETURN_ON_FAIL(scatter_func(dev_embedding_recv_buffer_ptr,
                                            local_recv_buffer_desc,
                                            dev_raw_indice_ptr,
                                            raw_indice_desc,
                                            output_gref,
                                            output_desc,
                                            stream));
    WM_CUDA_CHECK(cudaGetLastError());
    // WM_CUDA_CHECK(cudaStreamSynchronize(stream));
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
