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
#include <wholememory/wholememory.h>

#include "communicator.hpp"
#include "file_io.h"
#include "initialize.hpp"
#include "memory_handle.hpp"
#include "parallel_utils.hpp"

#ifdef __cplusplus
extern "C" {
#endif

wholememory_error_code_t wholememory_init(unsigned int flags, LogLevel log_level)
{
  return wholememory::init(flags, log_level);
}

wholememory_error_code_t wholememory_finalize() { return wholememory::finalize(); }

wholememory_error_code_t wholememory_create_unique_id(wholememory_unique_id_t* unique_id)
{
  return wholememory::create_unique_id(unique_id);
}

wholememory_error_code_t wholememory_create_communicator(wholememory_comm_t* comm,
                                                         wholememory_unique_id_t unique_id,
                                                         int rank,
                                                         int size)
{
  return wholememory::create_communicator(comm, unique_id, rank, size);
}

wholememory_error_code_t wholememory_split_communicator(wholememory_comm_t* new_comm,
                                                        wholememory_comm_t comm,
                                                        int color,
                                                        int key)
{
  return wholememory::split_communicator(new_comm, comm, color, key);
}

wholememory_error_code_t wholememory_destroy_communicator(wholememory_comm_t comm)
{
  return wholememory::destroy_communicator(comm);
}

wholememory_error_code_t wholememory_communicator_support_type_location(
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location)
{
  return wholememory::communicator_support_type_location(comm, memory_type, memory_location);
}

wholememory_error_code_t wholememory_communicator_get_rank(int* rank, wholememory_comm_t comm)
{
  return wholememory::communicator_get_rank(rank, comm);
}

wholememory_error_code_t wholememory_communicator_get_size(int* size, wholememory_comm_t comm)
{
  return wholememory::communicator_get_size(size, comm);
}

wholememory_error_code_t wholememory_communicator_get_local_size(int* local_size,
                                                                 wholememory_comm_t comm)
{
  return wholememory::communicator_get_local_size(local_size, comm);
}

bool wholememory_communicator_is_bind_to_nvshmem(wholememory_comm_t comm)
{
#ifdef WITH_NVSHMEM_SUPPORT
  return wholememory::communicator_is_bind_to_nvshmem(comm);
#else
  return false;
#endif
}

wholememory_error_code_t wholememory_communicator_set_distributed_backend(
  wholememory_comm_t comm, wholememory_distributed_backend_t distributed_backend)
{
  return wholememory::communicator_set_distributed_backend(comm, distributed_backend);
}

wholememory_distributed_backend_t wholememory_communicator_get_distributed_backend(
  wholememory_comm_t comm)
{
  return wholememory::communicator_get_distributed_backend(comm);
}

wholememory_error_code_t wholememory_communicator_barrier(wholememory_comm_t comm)
{
  wholememory::communicator_barrier(comm);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_malloc(wholememory_handle_t* wholememory_handle_ptr,
                                            size_t total_size,
                                            wholememory_comm_t comm,
                                            wholememory_memory_type_t memory_type,
                                            wholememory_memory_location_t memory_location,
                                            size_t data_granularity,
                                            size_t* rank_entry_partition)
{
  return wholememory::create_wholememory(wholememory_handle_ptr,
                                         total_size,
                                         comm,
                                         memory_type,
                                         memory_location,
                                         data_granularity,
                                         rank_entry_partition);
}

wholememory_error_code_t wholememory_free(wholememory_handle_t wholememory_handle)
{
  return wholememory::destroy_wholememory(wholememory_handle);
}

wholememory_error_code_t wholememory_get_communicator(wholememory_comm_t* comm,
                                                      wholememory_handle_t wholememory_handle)
{
  return wholememory::get_communicator_from_handle(comm, wholememory_handle);
}

wholememory_error_code_t wholememory_get_local_communicator(wholememory_comm_t* comm,
                                                            wholememory_handle_t wholememory_handle)
{
  return wholememory::get_local_communicator_from_handle(comm, wholememory_handle);
}

wholememory_error_code_t wholememory_get_cross_communicator(wholememory_comm_t* comm,
                                                            wholememory_handle_t wholememory_handle)
{
  return wholememory::get_cross_communicator_from_handle(comm, wholememory_handle);
}

wholememory_memory_type_t wholememory_get_memory_type(wholememory_handle_t wholememory_handle)
{
  return wholememory::get_memory_type(wholememory_handle);
}

wholememory_memory_location_t wholememory_get_memory_location(
  wholememory_handle_t wholememory_handle)
{
  return wholememory::get_memory_location(wholememory_handle);
}

wholememory_distributed_backend_t wholememory_get_distributed_backend(
  wholememory_handle_t wholememory_handle)
{
  return wholememory::get_distributed_backend_t(wholememory_handle);
}

size_t wholememory_get_total_size(wholememory_handle_t wholememory_handle)
{
  return wholememory::get_total_size(wholememory_handle);
}

size_t wholememory_get_data_granularity(wholememory_handle_t wholememory_handle)
{
  return wholememory::get_data_granularity(wholememory_handle);
}

wholememory_error_code_t wholememory_get_local_memory(void** local_ptr,
                                                      size_t* local_size,
                                                      size_t* local_offset,
                                                      wholememory_handle_t wholememory_handle)
{
  return wholememory::get_local_memory_from_handle(
    local_ptr, local_size, local_offset, wholememory_handle);
}

wholememory_error_code_t wholememory_get_rank_memory(void** rank_memory_ptr,
                                                     size_t* rank_memory_size,
                                                     size_t* rank_memory_offset,
                                                     int rank,
                                                     wholememory_handle_t wholememory_handle)
{
  return wholememory::get_rank_memory_from_handle(
    rank_memory_ptr, rank_memory_size, rank_memory_offset, rank, wholememory_handle);
}

wholememory_error_code_t wholememory_equal_entry_partition_plan(size_t* entry_per_rank,
                                                                size_t total_entry_count,
                                                                int world_size)
{
  return wholememory::equal_partition_plan(entry_per_rank, total_entry_count, world_size);
}

wholememory_error_code_t wholememory_get_global_pointer(void** global_ptr,
                                                        wholememory_handle_t wholememory_handle)
{
  return wholememory::get_global_pointer_from_handle(global_ptr, wholememory_handle);
}

wholememory_error_code_t wholememory_get_global_reference(wholememory_gref_t* wholememory_gref,
                                                          wholememory_handle_t wholememory_handle)
{
  return wholememory::get_global_reference_from_handle(wholememory_gref, wholememory_handle);
}

#ifdef WITH_NVSHMEM_SUPPORT

wholememory_error_code_t wholememory_get_nvshmem_reference(
  wholememory_nvshmem_ref_t* wholememory_nvshmem_ref, wholememory_handle_t wholememory_handle)
{
  return wholememory::get_nvshmem_reference_frome_handle(wholememory_nvshmem_ref,
                                                         wholememory_handle);
}

#endif

wholememory_error_code_t wholememory_get_rank_partition_sizes(
  size_t* rank_sizes, wholememory_handle_t wholememory_handle)
{
  return wholememory::get_rank_partition_sizes_from_handle(rank_sizes, wholememory_handle);
}

wholememory_error_code_t wholememory_get_rank_partition_offsets(
  size_t* rank_offsets, wholememory_handle_t wholememory_handle)
{
  return wholememory::get_rank_partition_offsets_from_handle(rank_offsets, wholememory_handle);
}

wholememory_error_code_t wholememory_get_local_size(size_t* local_size,
                                                    wholememory_handle_t wholememory_handle)
{
  return wholememory::get_local_size_from_handle(local_size, wholememory_handle);
}

wholememory_error_code_t wholememory_get_local_offset(size_t* local_size,
                                                      wholememory_handle_t wholememory_handle)
{
  return wholememory::get_local_offset_from_handle(local_size, wholememory_handle);
}

int fork_get_device_count()
{
  try {
    return ForkGetDeviceCount();
  } catch (...) {
    WHOLEMEMORY_ERROR("fork_get_device_count failed.");
    return -1;
  }
}

wholememory_error_code_t wholememory_load_from_file(wholememory_handle_t wholememory_handle,
                                                    size_t memory_offset,
                                                    size_t memory_entry_size,
                                                    size_t file_entry_size,
                                                    const char** file_names,
                                                    int file_count,
                                                    int round_robin_size)
{
  return wholememory::load_file_to_handle(wholememory_handle,
                                          memory_offset,
                                          memory_entry_size,
                                          file_entry_size,
                                          file_names,
                                          file_count,
                                          round_robin_size);
}

wholememory_error_code_t wholememory_store_to_file(wholememory_handle_t wholememory_handle,
                                                   size_t memory_offset,
                                                   size_t memory_entry_stride,
                                                   size_t file_entry_size,
                                                   const char* local_file_name)
{
  return wholememory::store_handle_to_file(
    wholememory_handle, memory_offset, memory_entry_stride, file_entry_size, local_file_name);
}

wholememory_error_code_t wholememory_load_hdfs_support() { return WHOLEMEMORY_NOT_IMPLEMENTED; }

wholememory_error_code_t wholememory_load_from_hdfs_file(wholememory_handle_t wholememory_handle,
                                                         size_t memory_offset,
                                                         size_t memory_entry_size,
                                                         size_t file_entry_size,
                                                         const char* hdfs_host,
                                                         int hdfs_port,
                                                         const char* hdfs_user,
                                                         const char* hdfs_path,
                                                         const char* hdfs_prefix)
{
  return WHOLEMEMORY_NOT_IMPLEMENTED;
}

bool wholememory_is_intranode_communicator(wholememory_comm_t comm)
{
  return wholememory::is_intranode_communicator(comm);
}

bool wholememory_is_intra_mnnvl_communicator(wholememory_comm_t comm)
{
  return wholememory::is_intra_mnnvl_communicator(comm);
}

wholememory_error_code_t wholememory_communicator_get_clique_info(clique_info_t* clique_info,
                                                                  wholememory_comm_t comm)
{
  return wholememory::communicator_get_clique_info(clique_info, comm);
}

bool wholememory_is_build_with_nvshmem()
{
#ifdef WITH_NVSHMEM_SUPPORT

  return true;

#else
  return false;
#endif
}
#ifdef __cplusplus
}
#endif
