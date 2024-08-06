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
#include "wholememory/wholememory_tensor.h"

#include <atomic>
#include <cstdlib>

#include "logger.hpp"

#ifdef WM_TENSOR_COUNT_DEBUG
static std::atomic<int64_t> wm_tensor_count;
static void inc_tensor_count() { wm_tensor_count.fetch_add(1); }
static void dec_tensor_count() { wm_tensor_count.fetch_add(-1); }
static int64_t get_tensor_count() { return wm_tensor_count.load(); }
#else
static void inc_tensor_count() {}
static void dec_tensor_count() {}
static int64_t get_tensor_count() { return 0; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct wholememory_tensor_ {
  union {
    wholememory_handle_t wholememory_handle;
    void* storage_ptr;
  };
  wholememory_tensor_description_t tensor_description;
  wholememory_tensor_t root_tensor;
  bool is_wholememory;
  bool own_handle;
};

int64_t get_wholememory_tensor_count() { return get_tensor_count(); }

wholememory_error_code_t wholememory_create_tensor(
  wholememory_tensor_t* p_wholememory_tensor,
  wholememory_tensor_description_t* tensor_description,
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location,
  size_t* tensor_entry_partition)
{
  if (p_wholememory_tensor == nullptr) {
    WHOLEMEMORY_ERROR("p_wholememory_tensor is nullptr");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description == nullptr) {
    WHOLEMEMORY_ERROR("tensor_description is nullptr");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dim <= 0 || tensor_description->dim > 2) {
    WHOLEMEMORY_ERROR("tensor_description->dim=%d", tensor_description->dim);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->storage_offset != 0) {
    WHOLEMEMORY_ERROR("tensor_description->storage_offset=%ld", tensor_description->storage_offset);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  int const dim = tensor_description->dim;
  if (tensor_description->strides[dim - 1] != 1) {
    WHOLEMEMORY_ERROR("tensor_description->strides[dim - 1]", tensor_description->strides[dim - 1]);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN ||
      tensor_description->dtype >= WHOLEMEMORY_DT_COUNT) {
    WHOLEMEMORY_ERROR("tensor_description is unknown");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  size_t elt_count   = wholememory_get_memory_element_count_from_tensor(tensor_description);
  size_t elt_size    = wholememory_dtype_get_element_size(tensor_description->dtype);
  size_t malloc_size = elt_count * elt_size;
  size_t granularity = elt_size * tensor_description->strides[0];

  auto* wholememory_tensor = static_cast<wholememory_tensor_*>(malloc(sizeof(wholememory_tensor_)));

  wholememory_tensor->tensor_description = *tensor_description;
  wholememory_tensor->own_handle         = true;
  wholememory_tensor->is_wholememory     = true;
  wholememory_tensor->root_tensor        = wholememory_tensor;
  *p_wholememory_tensor                  = wholememory_tensor;
  auto ret_code = wholememory_malloc(&wholememory_tensor->wholememory_handle,
                                     malloc_size,
                                     comm,
                                     memory_type,
                                     memory_location,
                                     granularity,
                                     tensor_entry_partition);
  inc_tensor_count();
  if (ret_code != WHOLEMEMORY_SUCCESS) { free(wholememory_tensor); }
  return ret_code;
}

wholememory_error_code_t wholememory_destroy_tensor(wholememory_tensor_t wholememory_tensor)
{
  if (wholememory_tensor->own_handle) {
    if (wholememory_tensor->is_wholememory) {
      WHOLEMEMORY_RETURN_ON_FAIL(wholememory_free(wholememory_tensor->wholememory_handle));
    } else {
      free(wholememory_tensor->storage_ptr);
    }
  }
  dec_tensor_count();
  free(wholememory_tensor);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_make_tensor_from_pointer(
  wholememory_tensor_t* p_wholememory_tensor,
  void* storage_ptr,
  wholememory_tensor_description_t* tensor_description)
{
  if (storage_ptr == nullptr || tensor_description->dim == 0) {
    auto* wholememory_tensor =
      static_cast<wholememory_tensor_*>(malloc(sizeof(wholememory_tensor_)));
    wholememory_tensor->storage_ptr        = storage_ptr;
    wholememory_tensor->tensor_description = *tensor_description;
    wholememory_tensor->own_handle         = false;
    wholememory_tensor->is_wholememory     = false;
    wholememory_tensor->root_tensor        = wholememory_tensor;
    *p_wholememory_tensor                  = wholememory_tensor;
    inc_tensor_count();
    return WHOLEMEMORY_SUCCESS;
  }

  if (p_wholememory_tensor == nullptr || tensor_description == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dim < 0) {
    WHOLEMEMORY_ERROR("tensor_description->dim=%d", tensor_description->dim);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  int const dim = tensor_description->dim;
  if (tensor_description->strides[dim - 1] != 1) {
    WHOLEMEMORY_ERROR("tensor_description->strides[dim - 1]", tensor_description->strides[dim - 1]);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN ||
      tensor_description->dtype >= WHOLEMEMORY_DT_COUNT) {
    WHOLEMEMORY_ERROR("tensor_description is unknown");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* wholememory_tensor = static_cast<wholememory_tensor_*>(malloc(sizeof(wholememory_tensor_)));
  wholememory_tensor->storage_ptr        = storage_ptr;
  wholememory_tensor->tensor_description = *tensor_description;
  wholememory_tensor->own_handle         = false;
  wholememory_tensor->is_wholememory     = false;
  wholememory_tensor->root_tensor        = wholememory_tensor;
  *p_wholememory_tensor                  = wholememory_tensor;
  inc_tensor_count();
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_make_tensor_from_handle(
  wholememory_tensor_t* p_wholememory_tensor,
  wholememory_handle_t wholememory_handle,
  wholememory_tensor_description_t* tensor_description)
{
  if (wholememory_handle == nullptr || p_wholememory_tensor == nullptr ||
      tensor_description == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dim <= 0 || tensor_description->dim > 2) {
    WHOLEMEMORY_ERROR("tensor_description->dim=%d", tensor_description->dim);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  int const dim = tensor_description->dim;
  if (tensor_description->strides[dim - 1] != 1) {
    WHOLEMEMORY_ERROR("tensor_description->strides[dim - 1]", tensor_description->strides[dim - 1]);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN ||
      tensor_description->dtype >= WHOLEMEMORY_DT_COUNT) {
    WHOLEMEMORY_ERROR("tensor_description is unknown");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* wholememory_tensor = static_cast<wholememory_tensor_*>(malloc(sizeof(wholememory_tensor_)));
  wholememory_tensor->wholememory_handle = wholememory_handle;
  wholememory_tensor->tensor_description = *tensor_description;
  wholememory_tensor->own_handle         = false;
  wholememory_tensor->is_wholememory     = true;
  wholememory_tensor->root_tensor        = wholememory_tensor;
  *p_wholememory_tensor                  = wholememory_tensor;
  inc_tensor_count();
  return WHOLEMEMORY_SUCCESS;
}

bool wholememory_tensor_has_handle(wholememory_tensor_t wholememory_tensor)
{
  return wholememory_tensor->is_wholememory;
}

wholememory_handle_t wholememory_tensor_get_memory_handle(wholememory_tensor_t wholememory_tensor)
{
  if (wholememory_tensor->is_wholememory) { return wholememory_tensor->wholememory_handle; }
  return nullptr;
}

wholememory_tensor_description_t* wholememory_tensor_get_tensor_description(
  wholememory_tensor_t wholememory_tensor)
{
  return &wholememory_tensor->tensor_description;
}

wholememory_error_code_t wholememory_tensor_get_global_reference(
  wholememory_tensor_t wholememory_tensor, wholememory_gref_t* wholememory_gref)
{
  if (wholememory_gref == nullptr || wholememory_tensor == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (wholememory_tensor->is_wholememory) {
    return wholememory_get_global_reference(wholememory_gref,
                                            wholememory_tensor->wholememory_handle);
  }
  *wholememory_gref =
    wholememory_create_continuous_global_reference(wholememory_tensor->storage_ptr);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_tensor_map_local_tensor(
  wholememory_tensor_t wholememory_tensor, wholememory_tensor_t* local_tensor)
{
  // NOTE: wholememory_tensor should NOT skip entry from front, but can skip from tail.
  if (local_tensor == nullptr || wholememory_tensor == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (!wholememory_tensor->is_wholememory) { return WHOLEMEMORY_INVALID_VALUE; }
  auto* wm_desc = wholememory_tensor_get_tensor_description(wholememory_tensor);
  if (wm_desc->dim != 1 && wm_desc->dim != 2) { return WHOLEMEMORY_INVALID_VALUE; }
  if (wm_desc->dim == 1 && wm_desc->storage_offset != 0) { return WHOLEMEMORY_INVALID_VALUE; }
  if (wm_desc->dim == 2 && wm_desc->storage_offset + wm_desc->sizes[1] > wm_desc->strides[0]) {
    return WHOLEMEMORY_INVALID_VALUE;
  }

  wholememory_comm_t wm_comm;
  int world_rank;

  void* local_ptr;
  size_t local_size, local_offset;
  auto* handle = wholememory_tensor_get_memory_handle(wholememory_tensor);
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, handle));
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&world_rank, wm_comm));

  size_t total_handle_memory_size = wholememory_get_total_size(handle);
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_get_local_memory(&local_ptr, &local_size, &local_offset, handle));
  size_t const element_size = wholememory_dtype_get_element_size(wm_desc->dtype);
  size_t const gran_size    = wm_desc->dim == 1 ? element_size : element_size * wm_desc->strides[0];
  local_size = std::min<size_t>(local_size, wm_desc->sizes[0] * gran_size - local_offset);
  if (local_size % gran_size != 0) return WHOLEMEMORY_LOGIC_ERROR;
  wholememory_tensor_description_t local_desc = *wm_desc;
  local_desc.sizes[0]                         = local_size / gran_size;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_make_tensor_from_pointer(local_tensor, local_ptr, &local_desc));

  return WHOLEMEMORY_SUCCESS;
}

void* wholememory_tensor_get_data_pointer(wholememory_tensor_t wholememory_tensor)
{
  char* data_ptr = nullptr;
  if (wholememory_tensor->is_wholememory &&
      wholememory_get_memory_type(wholememory_tensor->wholememory_handle) !=
        WHOLEMEMORY_MT_CONTINUOUS) {
    return nullptr;
  }
  if (!wholememory_tensor->is_wholememory) {
    data_ptr = static_cast<char*>(wholememory_tensor->storage_ptr);
  } else {
    if (wholememory_get_global_pointer(reinterpret_cast<void**>(&data_ptr),
                                       wholememory_tensor->wholememory_handle) !=
        WHOLEMEMORY_SUCCESS) {
      return nullptr;
    }
  }
  return data_ptr +
         wholememory_dtype_get_element_size(wholememory_tensor->tensor_description.dtype) *
           wholememory_tensor->tensor_description.storage_offset;
}

wholememory_error_code_t wholememory_tensor_get_entry_offsets(
  size_t* entry_offsets, wholememory_tensor_t wholememory_tensor)
{
  wholememory_tensor_t root_tensor = wholememory_tensor_get_root(wholememory_tensor);
  WHOLEMEMORY_CHECK_NOTHROW(
    (root_tensor->tensor_description.dim == 1 || root_tensor->tensor_description.dim == 2));
  if (wholememory_tensor->is_wholememory) {
    size_t embedding_stride = 1;
    size_t const element_size =
      wholememory_dtype_get_element_size(wholememory_tensor->tensor_description.dtype);
    if (root_tensor->tensor_description.dim == 2) {
      embedding_stride = root_tensor->tensor_description.strides[0];
    }

    int world_size;
    wholememory_comm_t comm;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(
      &comm, wholememory_tensor_get_memory_handle(wholememory_tensor)));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, comm));

    wholememory_get_rank_partition_offsets(
      entry_offsets, wholememory_tensor_get_memory_handle(wholememory_tensor));
    for (int i = 0; i < world_size + 1; i++) {
      WHOLEMEMORY_CHECK_NOTHROW(entry_offsets[i] % (embedding_stride * element_size) == 0);
      entry_offsets[i] /= (embedding_stride * element_size);
    }
    return WHOLEMEMORY_SUCCESS;
  }
  entry_offsets[0] = 0;
  entry_offsets[1] = root_tensor->tensor_description.sizes[0];
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_tensor_get_entry_partition_sizes(
  size_t* entry_partition, wholememory_tensor_t wholememory_tensor)
{
  wholememory_tensor_t root_tensor = wholememory_tensor_get_root(wholememory_tensor);
  WHOLEMEMORY_CHECK_NOTHROW(
    (root_tensor->tensor_description.dim == 1 || root_tensor->tensor_description.dim == 2));
  if (wholememory_tensor->is_wholememory) {
    size_t embedding_stride = 1;
    size_t const element_size =
      wholememory_dtype_get_element_size(wholememory_tensor->tensor_description.dtype);
    if (root_tensor->tensor_description.dim == 2) {
      embedding_stride = root_tensor->tensor_description.strides[0];
    }

    int world_size;
    wholememory_comm_t comm;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(
      &comm, wholememory_tensor_get_memory_handle(wholememory_tensor)));
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, comm));

    wholememory_get_rank_partition_sizes(entry_partition,
                                         wholememory_tensor_get_memory_handle(wholememory_tensor));
    for (int i = 0; i < world_size; i++) {
      WHOLEMEMORY_CHECK_NOTHROW(entry_partition[i] % (embedding_stride * element_size) == 0);
      entry_partition[i] /= (embedding_stride * element_size);
    }
    return WHOLEMEMORY_SUCCESS;
  }
  entry_partition[0] = root_tensor->tensor_description.sizes[0];
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_tensor_get_local_entry_count(
  size_t* local_entry_count, wholememory_tensor_t wholememory_tensor)
{
  wholememory_tensor_t root_tensor = wholememory_tensor_get_root(wholememory_tensor);
  WHOLEMEMORY_CHECK_NOTHROW(
    (root_tensor->tensor_description.dim == 1 || root_tensor->tensor_description.dim == 2));
  if (wholememory_tensor->is_wholememory) {
    size_t embedding_stride = 1;
    size_t const element_size =
      wholememory_dtype_get_element_size(wholememory_tensor->tensor_description.dtype);
    if (root_tensor->tensor_description.dim == 2) {
      embedding_stride = root_tensor->tensor_description.strides[0];
    }

    size_t entry_cnt;
    wholememory_get_local_size(&entry_cnt,
                               wholememory_tensor_get_memory_handle(wholememory_tensor));
    WHOLEMEMORY_CHECK_NOTHROW(entry_cnt % (embedding_stride * element_size) == 0);
    entry_cnt /= (embedding_stride * element_size);
    *local_entry_count = entry_cnt;
    return WHOLEMEMORY_SUCCESS;
  }
  *local_entry_count = root_tensor->tensor_description.sizes[0];
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_tensor_get_local_entry_start(
  size_t* local_entry_start, wholememory_tensor_t wholememory_tensor)
{
  wholememory_tensor_t root_tensor = wholememory_tensor_get_root(wholememory_tensor);
  WHOLEMEMORY_CHECK_NOTHROW(
    (root_tensor->tensor_description.dim == 1 || root_tensor->tensor_description.dim == 2));
  if (wholememory_tensor->is_wholememory) {
    size_t embedding_stride = 1;
    size_t const element_size =
      wholememory_dtype_get_element_size(wholememory_tensor->tensor_description.dtype);
    if (root_tensor->tensor_description.dim == 2) {
      embedding_stride = root_tensor->tensor_description.strides[0];
    }
    size_t entry_start;
    wholememory_get_local_offset(&entry_start,
                                 wholememory_tensor_get_memory_handle(wholememory_tensor));
    WHOLEMEMORY_CHECK_NOTHROW(entry_start % (embedding_stride * element_size) == 0);
    entry_start /= (embedding_stride * element_size);
    *local_entry_start = entry_start;
    return WHOLEMEMORY_SUCCESS;
  }
  *local_entry_start = 0;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t wholememory_tensor_get_subtensor(
  wholememory_tensor_t wholememory_tensor,
  int64_t* starts,
  int64_t* ends,
  wholememory_tensor_t* p_sub_wholememory_tensor)
{
  if (p_sub_wholememory_tensor == nullptr || wholememory_tensor == nullptr || starts == nullptr ||
      ends == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (wholememory_tensor->tensor_description.dim > 2) { return WHOLEMEMORY_NOT_IMPLEMENTED; }
  int const dim      = wholememory_tensor->tensor_description.dim;
  int64_t offsets[2] = {0, 0};
  if (dim == 1) {
    offsets[0] = wholememory_tensor->tensor_description.storage_offset;
  } else {
    offsets[0] = wholememory_tensor->tensor_description.storage_offset /
                 wholememory_tensor->tensor_description.strides[0];
    offsets[1] = wholememory_tensor->tensor_description.storage_offset %
                 wholememory_tensor->tensor_description.strides[0];
  }
  int64_t new_size[2] = {0, 0};
  int64_t new_offset  = wholememory_tensor->tensor_description.storage_offset;
  for (int i = 0; i < dim; i++) {
    int64_t starts_i = starts[i];
    int64_t ends_i   = ends[i];
    if (starts[i] == -1) starts_i = 0;
    if (ends[i] == -1) ends_i = wholememory_tensor->tensor_description.sizes[i];
    if (ends_i <= starts_i) return WHOLEMEMORY_INVALID_INPUT;
    if (starts_i >= wholememory_tensor->tensor_description.sizes[i])
      return WHOLEMEMORY_INVALID_INPUT;
    if (ends_i <= 0) return WHOLEMEMORY_INVALID_INPUT;
    new_offset += wholememory_tensor->tensor_description.strides[i] * starts_i;
    new_size[i] = ends_i - starts_i;
  }
  auto* sub_wholememory_tensor =
    static_cast<wholememory_tensor_*>(malloc(sizeof(wholememory_tensor_)));
  *sub_wholememory_tensor                                   = *wholememory_tensor;
  sub_wholememory_tensor->own_handle                        = false;
  sub_wholememory_tensor->tensor_description.storage_offset = new_offset;
  sub_wholememory_tensor->tensor_description.dim            = dim;
  sub_wholememory_tensor->tensor_description.dtype =
    sub_wholememory_tensor->tensor_description.dtype;
  for (int i = 0; i < dim; i++) {
    sub_wholememory_tensor->tensor_description.sizes[i] = new_size[i];
    sub_wholememory_tensor->tensor_description.strides[i] =
      wholememory_tensor->tensor_description.strides[i];
  }
  *p_sub_wholememory_tensor = sub_wholememory_tensor;
  inc_tensor_count();

  return WHOLEMEMORY_SUCCESS;
}

wholememory_tensor_t wholememory_tensor_get_root(wholememory_tensor_t wholememory_tensor)
{
  return wholememory_tensor->root_tensor;
}

#ifdef __cplusplus
}
#endif
