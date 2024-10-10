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

#include <wholememory/wholememory.h>

#include <cuda_runtime_api.h>

#include "communicator.hpp"

namespace wholememory {

class wholememory_impl;

}

struct wholememory_handle_ {
  int handle_id;
  wholememory::wholememory_impl* impl = nullptr;
  ~wholememory_handle_();
};

namespace wholememory {

wholememory_error_code_t create_wholememory(wholememory_handle_t* wholememory_handle_ptr,
                                            size_t total_size,
                                            wholememory_comm_t comm,
                                            wholememory_memory_type_t memory_type,
                                            wholememory_memory_location_t memory_location,
                                            size_t data_granularity,
                                            size_t* rank_entry_partition = nullptr) noexcept;

wholememory_error_code_t destroy_wholememory_with_comm_locked(
  wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t destroy_wholememory(wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_communicator_from_handle(
  wholememory_comm_t* comm, wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_local_communicator_from_handle(
  wholememory_comm_t* comm, wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_cross_communicator_from_handle(
  wholememory_comm_t* comm, wholememory_handle_t wholememory_handle) noexcept;

wholememory_memory_type_t get_memory_type(wholememory_handle_t wholememory_handle) noexcept;

wholememory_memory_location_t get_memory_location(wholememory_handle_t wholememory_handle) noexcept;

size_t get_total_size(wholememory_handle_t wholememory_handle) noexcept;

size_t get_data_granularity(wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_local_memory_from_handle(
  void** local_ptr,
  size_t* local_size,
  size_t* local_offset,
  wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_local_node_memory_from_handle(
  void** local_ptr,
  size_t* local_size,
  size_t* local_offset,
  wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_rank_memory_from_handle(
  void** rank_memory_ptr,
  size_t* rank_memory_size,
  size_t* rank_memory_offset,
  int rank,
  wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_local_size_from_handle(
  size_t* size, wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_local_offset_from_handle(
  size_t* offset, wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_global_pointer_from_handle(
  void** global_ptr, wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_global_reference_from_handle(
  wholememory_gref_t* wholememory_gref, wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t equal_partition_plan(size_t* entry_per_rank,
                                              size_t total_entry_count,
                                              int world_size) noexcept;

wholememory_error_code_t get_rank_partition_sizes_from_handle(
  size_t* rank_sizes, wholememory_handle_t wholememory_handle) noexcept;

wholememory_error_code_t get_rank_partition_offsets_from_handle(
  size_t* rank_offsets, wholememory_handle_t wholememory_handle) noexcept;

wholememory_distributed_backend_t get_distributed_backend_t(
  wholememory_handle_t wholememory_handle) noexcept;

#ifdef WITH_NVSHMEM_SUPPORT

wholememory_error_code_t get_nvshmem_reference_frome_handle(
  wholememory_nvshmem_ref_t* wholememory_nvshmem_ref,
  wholememory_handle_t wholememory_handle) noexcept;
#endif

}  // namespace wholememory
