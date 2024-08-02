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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to WholeMemoryTensor
 *
 * An Opaque handle to WholeMemoryTensor
 */
typedef struct wholememory_tensor_* wholememory_tensor_t;

/**
 * Create WholeMemory Tensor
 * @param wholememory_tensor : returned WholeMemory Tensor handle
 * @param tensor_description : description of the WholeMemory Tensor, should be 1-D or 2-D
 * continuous tensor without offset.
 * @param comm : WholeMemory Communicator
 * @param memory_type : Memory Type of the underlying WholeMemory
 * @param memory_location : Memory Location of the underlying WholeMemory
 * @param tensor_entry_partition : Tensor entry count of each rank, the length must be world_size.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_create_tensor(
  wholememory_tensor_t* wholememory_tensor,
  wholememory_tensor_description_t* tensor_description,
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location,
  size_t* tensor_entry_partition = nullptr);

/**
 * Destroy WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor to destroy
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_destroy_tensor(wholememory_tensor_t wholememory_tensor);

/**
 * Make WholeMemory Tensor from local memory
 * @param wholememory_tensor : returned WholeMemory Tensor handle
 * @param storage_ptr : pointer to underlying storage memory. Note: storage pointer may be not same
 * as data pointer.
 * @param tensor_description : description of the WholeMemory Tensor, should be 1-D or 2-D
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_make_tensor_from_pointer(
  wholememory_tensor_t* wholememory_tensor,
  void* storage_ptr,
  wholememory_tensor_description_t* tensor_description);

/**
 * Make WholeMemory Tensor from local memory
 * @param wholememory_tensor : returned WholeMemory Tensor handle
 * @param wholememory_handle : WholeMemory Handle
 * @param tensor_description : description of the WholeMemory Tensor, should be 1-D or 2-D
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_make_tensor_from_handle(
  wholememory_tensor_t* wholememory_tensor,
  wholememory_handle_t wholememory_handle,
  wholememory_tensor_description_t* tensor_description);

/**
 * Check if has WholeMemory Handle, WholeMemory Tensor created by wholememory_make_tensor has no
 * Handle
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : if has WholeMemory Handle
 */
bool wholememory_tensor_has_handle(wholememory_tensor_t wholememory_tensor);

/**
 * Get WholeMemory handle from WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : WholeMemory handle
 */
wholememory_handle_t wholememory_tensor_get_memory_handle(wholememory_tensor_t wholememory_tensor);

/**
 * Get tensor description from WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : pointer to the underlying wholememory_tensor_description_t
 */
wholememory_tensor_description_t* wholememory_tensor_get_tensor_description(
  wholememory_tensor_t wholememory_tensor);

/**
 * Get global reference from WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor
 * @param wholememory_gref : global reference
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_tensor_get_global_reference(
  wholememory_tensor_t wholememory_tensor, wholememory_gref_t* wholememory_gref);

/**
 * Map local tensor of WholeMemory Tensor.
 * Only support 1D and 2D tensor with WholeMemory Handle.
 * For 1D tensor, storage_offset should be 0
 * For 2D tensor, storage_offset + size[1] should <= stride[0]
 *
 * @param wholememory_tensor : WholeMemory Tensor.
 * @param local_tensor : returned local tensor, need to be destroyed.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_tensor_map_local_tensor(
  wholememory_tensor_t wholememory_tensor, wholememory_tensor_t* local_tensor);

/**
 * Get data pointer from WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : Pointer to first data for CONTINUOUS WholeMemory or not WholeMemory.
 */
void* wholememory_tensor_get_data_pointer(wholememory_tensor_t wholememory_tensor);

/**
 * Get entry offset of each rank from WholeMemory Tensor
 * @param entry_offsets : returned entry offset of each rank
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_tensor_get_entry_offsets(
  size_t* entry_offsets, wholememory_tensor_t wholememory_tensor);

/**
 * Get entry count of each rank from WholeMemory Tensor
 * @param entry_partition : returned entry count of each rank
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_tensor_get_entry_partition_sizes(
  size_t* entry_partition, wholememory_tensor_t wholememory_tensor);

/**
 * Get entry count of current rank from WholeMemory Tensor
 * @param local_entry_count  : returned entry count of current rank
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_tensor_get_local_entry_count(
  size_t* local_entry_count, wholememory_tensor_t wholememory_tensor);

/**
 * Get entry start of current rank from WholeMemory Tensor
 * @param local_entry_start  : returned entry start id of current rank
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_tensor_get_local_entry_start(
  size_t* local_entry_start, wholememory_tensor_t wholememory_tensor);

/**
 * Get sub tensor of a WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor
 * @param starts : starts of each dim, length should be the dim of wholememory_tensor.
 * @param ends : ends of each dim, length should be the dim of wholememory_tensor
 * @param sub_wholememory_tensor : pointer to returned sub tensor
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_tensor_get_subtensor(
  wholememory_tensor_t wholememory_tensor,
  int64_t* starts,
  int64_t* ends,
  wholememory_tensor_t* sub_wholememory_tensor);

/**
 * Get root tensor of a WholeMemory Tensor, root means it is not a sub tensor of any WholeMemory
 * Tensor.
 * @param wholememory_tensor : WholeMemory Tensor
 * @return : the root of current WholeMemory tensor, maybe same as wholememory_tensor.
 */
wholememory_tensor_t wholememory_tensor_get_root(wholememory_tensor_t wholememory_tensor);

#define WM_TENSOR_COUNT_DEBUG
int64_t get_wholememory_tensor_count();

#ifdef __cplusplus
}
#endif
