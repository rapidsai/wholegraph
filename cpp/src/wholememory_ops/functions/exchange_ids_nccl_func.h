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
#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include <wholememory_ops/temp_memory_handle.hpp>
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

/**
 * Bucket and exchange ids using collective communication
 *
 * @param indices : pointer to indices array
 * @param indice_desc : indices array description, should have storage offset = 0, indice can be
 * int32 or int64
 * @param host_recv_rank_id_count_ptr : pointer to int64_t array of received id count from each rank
 * @param host_rank_id_count_ptr : pointer to int64_t array of id count to send to each rank.
 * @param dev_recv_indices_buffer_handle : temp_memory_handle to create buffer for received indices.
 * @param dev_raw_indice_ptr : pointer to allocated int64_t array to storage raw indices mapping of
 * sort
 * @param embedding_entry_offsets : embedding entry offsets
 * @param wm_comm : WholeMemory Communicator
 * @param p_thrust_allocator : thrust allocator
 * @param p_env_fns : EnvFns
 * @param stream : CUDA stream to use.
 * @return : WHOLEMEMORY_SUCCESS on success, others on failure
 */
wholememory_error_code_t bucket_and_exchange_ids_func(
  void* indices,
  wholememory_array_description_t indice_desc,
  int64_t* host_recv_rank_id_count_ptr,
  int64_t* host_rank_id_count_ptr,
  temp_memory_handle* dev_recv_indices_buffer_handle,
  int64_t* dev_raw_indice_ptr,
  size_t* embedding_entry_offsets,
  wholememory_comm_t wm_comm,
  wm_thrust_allocator* p_thrust_allocator,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

}  // namespace wholememory_ops
