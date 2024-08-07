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
#ifdef WITH_NVSHMEM_SUPPORT

#include <cuda_runtime_api.h>
#include <wholememory/wholememory.h>

#include <wholememory/env_func_ptrs.h>

#include "logger.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/integer_utils.hpp"
#include "wholememory/memory_handle.hpp"
#include "wholememory_ops/functions/bucket_ids_func.h"
#include "wholememory_ops/functions/exchange_embeddings_nccl_func.h"
#include "wholememory_ops/functions/exchange_ids_nccl_func.h"
#include "wholememory_ops/functions/gather_scatter_func.cuh"
#include "wholememory_ops/functions/gather_scatter_func.h"

#include "wholememory_ops/register.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"
#include <wholememory/tensor_description.h>

#include <nvshmem.h>
#include <nvshmemx.h>

namespace wholememory_ops {

wholememory_error_code_t nvshmem_scatter_floating_int32_func(
  wholememory_comm_t wm_comm,
  void* input,
  void* temp_input,
  wholememory_matrix_description_t input_desc,
  const void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
  wholememory_matrix_description_t embedding_desc,
  size_t* embedding_entry_offsets,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms);

wholememory_error_code_t nvshmem_scatter_floating_int64_func(
  wholememory_comm_t wm_comm,
  void* input,
  void* temp_input,
  wholememory_matrix_description_t input_desc,
  const void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
  wholememory_matrix_description_t embedding_desc,
  size_t* embedding_entry_offsets,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms);

wholememory_error_code_t nvshmem_scatter_integer_int32_func(
  wholememory_comm_t wm_comm,
  void* input,
  void* temp_input,
  wholememory_matrix_description_t input_desc,
  const void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
  wholememory_matrix_description_t embedding_desc,
  size_t* embedding_entry_offsets,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms);

wholememory_error_code_t nvshmem_scatter_integer_int64_func(
  wholememory_comm_t wm_comm,
  void* input,
  void* temp_input,
  wholememory_matrix_description_t input_desc,
  const void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
  wholememory_matrix_description_t embedding_desc,
  size_t* embedding_entry_offsets,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms);

wholememory_error_code_t wholememory_scatter_nvshmem(
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
  try {
    bool embedding_is_float = wholememory_dtype_is_floating_number(wholememory_desc.dtype);
    WHOLEMEMORY_CHECK(embedding_is_float ||
                      wholememory_dtype_is_integer_number(wholememory_desc.dtype));
    bool input_is_float = wholememory_dtype_is_floating_number(input_desc.dtype);
    WHOLEMEMORY_CHECK(input_is_float || wholememory_dtype_is_integer_number(input_desc.dtype));
    WHOLEMEMORY_EXPECTS(
      embedding_is_float == input_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indices_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
    if (wholememory_desc.storage_offset < 0 ||
        wholememory_desc.storage_offset + wholememory_desc.sizes[1] > wholememory_desc.stride) {
      return WHOLEMEMORY_INVALID_INPUT;
    }

    wholememory_comm_t wm_comm;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, wholememory_handle));

    int world_size;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));

    temp_memory_handle dev_embedding_entry_offsets_handle(p_env_fns);
    size_t* dev_embedding_entry_offsets_ptr = static_cast<size_t*>(
      dev_embedding_entry_offsets_handle.device_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));
    temp_memory_handle host_embedding_entry_offsets_handle(p_env_fns);
    size_t* host_embedding_entry_offsets_ptr = static_cast<size_t*>(
      host_embedding_entry_offsets_handle.host_malloc(world_size + 1, WHOLEMEMORY_DT_INT64));

    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_rank_partition_offsets(host_embedding_entry_offsets_ptr, wholememory_handle));

    size_t element_size         = wholememory_dtype_get_element_size(wholememory_desc.dtype);
    size_t embedding_entry_size = element_size * wholememory_desc.stride;
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

    wholememory_nvshmem_ref_t embedding_nvshmem_ref;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_nvshmem_reference(&embedding_nvshmem_ref, wholememory_handle));

    temp_memory_handle device_temp_input_handle(p_env_fns);
    size_t temp_input_ele_size = wholememory_get_memory_element_count_from_matrix(&input_desc);
    void* temp_input_ptr =
      device_temp_input_handle.device_malloc(temp_input_ele_size, wholememory_desc.dtype);
    size_t temp_input_byte_size =
      temp_input_ele_size * wholememory_dtype_get_element_size(wholememory_desc.dtype);
    // register
    if (nvshmemx_buffer_register(temp_input_ptr, temp_input_byte_size) != 0) {
      WHOLEMEMORY_ERROR("nvshmemx_buffer_register error in wholememory_gather_nvshmem");
    }

    wholememory_error_code_t (*p_nvshmem_scatter_func)(wholememory_comm_t,
                                                       void*,
                                                       void*,
                                                       wholememory_matrix_description_t,
                                                       const void*,
                                                       wholememory_array_description_t,
                                                       wholememory_nvshmem_ref_t,
                                                       wholememory_matrix_description_t,
                                                       size_t*,
                                                       wholememory_env_func_t*,
                                                       cudaStream_t,
                                                       int);

    if (embedding_is_float) {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_nvshmem_scatter_func = nvshmem_scatter_floating_int32_func;
      } else {
        p_nvshmem_scatter_func = nvshmem_scatter_floating_int64_func;
      }
    } else {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_nvshmem_scatter_func = nvshmem_scatter_integer_int32_func;
      } else {
        p_nvshmem_scatter_func = nvshmem_scatter_integer_int64_func;
      }
    }

    auto ret = p_nvshmem_scatter_func(wm_comm,
                                      input,
                                      temp_input_ptr,
                                      input_desc,
                                      indices,
                                      indices_desc,
                                      embedding_nvshmem_ref,
                                      wholememory_desc,
                                      dev_embedding_entry_offsets_ptr,
                                      p_env_fns,
                                      stream,
                                      scatter_sms);
    if (nvshmemx_buffer_unregister(temp_input_ptr) != 0) {
      WHOLEMEMORY_ERROR("nvshmemx_buffer_unregister error in wholememory_gather_nvshmem");
    }

    WM_CUDA_CHECK(cudaGetLastError());
    return ret;
  } catch (const wholememory::cuda_error& wle) {
    WHOLEMEMORY_ERROR("scatter CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_ERROR("scatter LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops

#endif
