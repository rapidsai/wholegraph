/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <wholememory/tensor_description.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t wholememory_dtype_get_element_size(wholememory_dtype_t dtype)
{
  switch (dtype) {
    case WHOLEMEMORY_DT_UNKNOWN: return 0;
    case WHOLEMEMORY_DT_INT8: return 1;
    case WHOLEMEMORY_DT_INT16:
    case WHOLEMEMORY_DT_BF16:
    case WHOLEMEMORY_DT_HALF: return 2;
    case WHOLEMEMORY_DT_INT:
    case WHOLEMEMORY_DT_FLOAT: return 4;
    case WHOLEMEMORY_DT_INT64:
    case WHOLEMEMORY_DT_DOUBLE: return 8;
    default: return -1;
  }
}

bool wholememory_dtype_is_floating_number(wholememory_dtype_t dtype)
{
  if (dtype == WHOLEMEMORY_DT_FLOAT || dtype == WHOLEMEMORY_DT_HALF ||
      dtype == WHOLEMEMORY_DT_DOUBLE || dtype == WHOLEMEMORY_DT_BF16)
    return true;
  return false;
}

bool wholememory_dtype_is_integer_number(wholememory_dtype_t dtype)
{
  if (dtype == WHOLEMEMORY_DT_INT || dtype == WHOLEMEMORY_DT_INT64 ||
      dtype == WHOLEMEMORY_DT_INT16 || dtype == WHOLEMEMORY_DT_INT8)
    return true;
  return false;
}

wholememory_array_description_t wholememory_create_array_desc(int64_t size,
                                                              int64_t storage_offset,
                                                              wholememory_dtype_t dtype)
{
  wholememory_array_description_t wm_array_desc;
  wm_array_desc.size           = size;
  wm_array_desc.storage_offset = storage_offset;
  wm_array_desc.dtype          = dtype;
  return wm_array_desc;
}

wholememory_matrix_description_t wholememory_create_matrix_desc(int64_t sizes[2],
                                                                int64_t stride,
                                                                int64_t storage_offset,
                                                                wholememory_dtype_t dtype)
{
  wholememory_matrix_description_t wm_matrix_desc;
  wm_matrix_desc.sizes[0]       = sizes[0];
  wm_matrix_desc.sizes[1]       = sizes[1];
  wm_matrix_desc.stride         = stride;
  wm_matrix_desc.storage_offset = storage_offset;
  wm_matrix_desc.dtype          = dtype;
  return wm_matrix_desc;
}

void wholememory_initialize_tensor_desc(wholememory_tensor_description_t* p_tensor_description)
{
  p_tensor_description->dim = 0;
  for (int i = 0; i < WHOLEMEMORY_MAX_TENSOR_DIM; i++) {
    p_tensor_description->sizes[i]   = 1;
    p_tensor_description->strides[i] = 1;
  }
  p_tensor_description->storage_offset = 0;
  p_tensor_description->dtype          = WHOLEMEMORY_DT_UNKNOWN;
}

void wholememory_copy_array_desc_to_matrix(wholememory_matrix_description_t* p_matrix_description,
                                           wholememory_array_description_t* p_array_description)
{
  p_matrix_description->sizes[0]       = p_array_description->size;
  p_matrix_description->sizes[1]       = 1;
  p_matrix_description->dtype          = p_array_description->dtype;
  p_matrix_description->stride         = 1;
  p_matrix_description->storage_offset = p_array_description->storage_offset;
}

void wholememory_copy_array_desc_to_tensor(wholememory_tensor_description_t* p_tensor_description,
                                           wholememory_array_description_t* p_array_description)
{
  wholememory_initialize_tensor_desc(p_tensor_description);
  p_tensor_description->dim            = 1;
  p_tensor_description->sizes[0]       = p_array_description->size;
  p_tensor_description->strides[0]     = 1;
  p_tensor_description->dtype          = p_array_description->dtype;
  p_tensor_description->storage_offset = p_array_description->storage_offset;
}

void wholememory_copy_matrix_desc_to_tensor(wholememory_tensor_description_t* p_tensor_description,
                                            wholememory_matrix_description_t* p_matrix_description)
{
  wholememory_initialize_tensor_desc(p_tensor_description);
  p_tensor_description->dim            = 2;
  p_tensor_description->sizes[0]       = p_matrix_description->sizes[0];
  p_tensor_description->sizes[1]       = p_matrix_description->sizes[1];
  p_tensor_description->strides[0]     = p_matrix_description->stride;
  p_tensor_description->strides[1]     = 1;
  p_tensor_description->dtype          = p_matrix_description->dtype;
  p_tensor_description->storage_offset = p_matrix_description->storage_offset;
}

bool wholememory_convert_tensor_desc_to_array(
  wholememory_array_description_t* p_array_description,
  wholememory_tensor_description_t* p_tensor_description)
{
  if (p_tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN ||
      p_tensor_description->dtype >= WHOLEMEMORY_DT_COUNT)
    return false;
  if (p_tensor_description->dim != 1) return false;
  if (p_tensor_description->strides[0] != 1) return false;
  p_array_description->dtype          = p_tensor_description->dtype;
  p_array_description->storage_offset = p_tensor_description->storage_offset;
  p_array_description->size           = p_tensor_description->sizes[0];
  return true;
}

bool wholememory_convert_tensor_desc_to_matrix(
  wholememory_matrix_description_t* p_matrix_description,
  wholememory_tensor_description_t* p_tensor_description)
{
  if (p_tensor_description->dtype <= WHOLEMEMORY_DT_UNKNOWN ||
      p_tensor_description->dtype >= WHOLEMEMORY_DT_COUNT)
    return false;
  if (p_tensor_description->dim > 2 || p_tensor_description->dim <= 0) return false;
  if (p_tensor_description->dim == 2 && p_tensor_description->strides[1] != 1) return false;
  p_matrix_description->dtype          = p_tensor_description->dtype;
  p_matrix_description->storage_offset = p_tensor_description->storage_offset;
  p_matrix_description->sizes[0]       = p_tensor_description->sizes[0];
  if (p_tensor_description->dim == 2) {
    p_matrix_description->sizes[1] = p_tensor_description->sizes[1];
    p_matrix_description->stride   = p_tensor_description->strides[0];
  } else {
    p_matrix_description->sizes[1] = 1;
    p_matrix_description->stride   = 1;
  }
  return true;
}

int64_t wholememory_get_memory_element_count_from_array(
  wholememory_array_description_t* p_array_description)
{
  return p_array_description->size;
}

int64_t wholememory_get_memory_size_from_array(wholememory_array_description_t* p_array_description)
{
  return wholememory_get_memory_element_count_from_array(p_array_description) *
         wholememory_dtype_get_element_size(p_array_description->dtype);
}

int64_t wholememory_get_memory_element_count_from_matrix(
  wholememory_matrix_description_t* p_matrix_description)
{
  return p_matrix_description->sizes[0] * p_matrix_description->stride;
}

int64_t wholememory_get_memory_size_from_matrix(
  wholememory_matrix_description_t* p_matrix_description)
{
  return wholememory_get_memory_element_count_from_matrix(p_matrix_description) *
         wholememory_dtype_get_element_size(p_matrix_description->dtype);
}

int64_t wholememory_get_memory_element_count_from_tensor(
  wholememory_tensor_description_t* p_tensor_description)
{
  if (p_tensor_description->dim == 0) return 1;
  if (p_tensor_description->dim < 0 || p_tensor_description->dim >= WHOLEMEMORY_MAX_TENSOR_DIM)
    return -1;
  return p_tensor_description->strides[0] * p_tensor_description->sizes[0];
}

int64_t wholememory_get_memory_size_from_tensor(
  wholememory_tensor_description_t* p_tensor_description)
{
  return wholememory_get_memory_element_count_from_tensor(p_tensor_description) *
         wholememory_dtype_get_element_size(p_tensor_description->dtype);
}

bool wholememory_squeeze_tensor(wholememory_tensor_description_t* p_tensor_description, int dim)
{
  if (p_tensor_description == nullptr) return false;
  if (dim < 0 || dim >= p_tensor_description->dim) return false;
  if (p_tensor_description->sizes[dim] != 1) return false;
  if (dim != p_tensor_description->dim - 1 &&
      p_tensor_description->strides[dim] != p_tensor_description->strides[dim + 1]) {
    return false;
  }
  for (int idx = dim; idx < p_tensor_description->dim - 1; idx++) {
    p_tensor_description->sizes[idx]   = p_tensor_description->sizes[idx + 1];
    p_tensor_description->strides[idx] = p_tensor_description->strides[idx + 1];
  }
  p_tensor_description->dim--;
  return true;
}

bool wholememory_unsqueeze_tensor(wholememory_tensor_description_t* p_tensor_description, int dim)
{
  if (p_tensor_description == nullptr) return false;
  if (dim < 0 || dim > p_tensor_description->dim) return false;
  int idx             = p_tensor_description->dim;
  int64_t last_stride = p_tensor_description->strides[p_tensor_description->dim - 1];
  for (; idx > dim; idx--) {
    p_tensor_description->sizes[idx] = p_tensor_description->sizes[idx - 1];
    last_stride = p_tensor_description->strides[idx] = p_tensor_description->strides[idx - 1];
  }
  p_tensor_description->sizes[dim]   = 1;
  p_tensor_description->strides[dim] = last_stride;
  p_tensor_description->dim++;
  return true;
}

#ifdef __cplusplus
}
#endif
