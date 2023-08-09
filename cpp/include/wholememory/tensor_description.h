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
#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum wholememory_dtype_t
 * @brief defines WholeMemory data type for tensors
 */
enum wholememory_dtype_t {
  WHOLEMEMORY_DT_UNKNOWN = 0, /*!< Unknown type */
  WHOLEMEMORY_DT_FLOAT,       /*!< 32-bit float type */
  WHOLEMEMORY_DT_HALF,        /*!< 16-bit half float type */
  WHOLEMEMORY_DT_DOUBLE,      /*!< 64-bit double type */
  WHOLEMEMORY_DT_BF16,        /*!< 16-bit bfloat type */
  WHOLEMEMORY_DT_INT,         /*!< 32-bit signed integer type */
  WHOLEMEMORY_DT_INT64,       /*!< 64-bit signed integer type */
  WHOLEMEMORY_DT_INT16,       /*!< 16-bit signed integer type */
  WHOLEMEMORY_DT_INT8,        /*!< 8-bit signed integer type */
  WHOLEMEMORY_DT_COUNT,       /*!< total count if types */
};

/**
 * Get element size of wholememory_dtype_t
 * @param dtype : wholememory_dtype_t
 * @return : element size of dtype, -1 on invalid dtype.
 */
size_t wholememory_dtype_get_element_size(wholememory_dtype_t dtype);

/**
 * Check if dtype is floating number
 * @param dtype : wholememory_dtype_t
 * @return : True if dtype is WHOLEMEMORY_DT_FLOAT, WHOLEMEMORY_DT_HALF, WHOLEMEMORY_DT_DOUBLE or
 * WHOLEMEMORY_DT_BF16. False otherwise.
 */
bool wholememory_dtype_is_floating_number(wholememory_dtype_t dtype);

/**
 * Check if dtype is integer number
 * @param dtype : wholememory_dtype_t
 * @return : True if dtype is WHOLEMEMORY_DT_INT, WHOLEMEMORY_DT_INT64, WHOLEMEMORY_DT_INT16 or
 * WHOLEMEMORY_DT_INT8, False otherwise.
 */
bool wholememory_dtype_is_integer_number(wholememory_dtype_t dtype);

/**
 * @struct wholememory_array_description_t
 * @brief wrapper for array in WholeMemory
 */
struct wholememory_array_description_t {
  int64_t size;              /*!< size of the array in elements. */
  int64_t storage_offset;    /*!< offset in number of elements, NOT in bytes. */
  wholememory_dtype_t dtype; /*!< data type of the array */
};

/**
 * @struct wholememory_matrix_description_t
 * @brief wrapper for matrix in WholeMemory
 */
struct wholememory_matrix_description_t {
  int64_t sizes[2];          /*!< sizes[0] is row of the matrix, sizes[1] is column of the matrix */
  int64_t stride;            /*!< stride of first dimension, in number of elements */
  int64_t storage_offset;    /*!< offset in number of elements, NOT in bytes. */
  wholememory_dtype_t dtype; /*!< data type of the matrix */
};

#define WHOLEMEMORY_MAX_TENSOR_DIM (8)

/**
 * @struct wholememory_tensor_description_t
 * @brief Tensor description in WholeMemory, dimension 0 is the slowest changed dimension
 */
struct wholememory_tensor_description_t {
  int64_t sizes[WHOLEMEMORY_MAX_TENSOR_DIM]; /*!< size of each dimension of the tensor, in number of
                                                elements */
  int64_t strides[WHOLEMEMORY_MAX_TENSOR_DIM]; /*!< stride of the tensor, in number of elements */
  int64_t storage_offset;                      /*!< offset in number of elements, NOT in bytes. */
  int dim;                                     /*!< dim of the tensor */
  wholememory_dtype_t dtype;                   /*!< data type of the tensor */
};

/*!
 * Create wholememory_array_description_t object
 * @param size : array size in number of elements
 * @param storage_offset : storage offset in number of elements
 * @param dtype : data type of array elements
 * @return created wholememory_array_description_t
 */
wholememory_array_description_t wholememory_create_array_desc(int64_t size,
                                                              int64_t storage_offset,
                                                              wholememory_dtype_t dtype);

/*!
 * Create wholememory_matrix_description_t object
 * @param sizes : matrix sizes array, counted in number of elements, sizes[1] changes fastest.
 * @param stride : stride of first dimension(slower changed dimension), stride is counted in number
 * of elements
 * @param storage_offset : storage offset in number of elements
 * @param dtype : data type of matrix elements
 * @return created wholememory_matrix_description_t
 */
wholememory_matrix_description_t wholememory_create_matrix_desc(int64_t sizes[2],
                                                                int64_t stride,
                                                                int64_t storage_offset,
                                                                wholememory_dtype_t dtype);

/*!
 * Initialize wholememory_tensor_description_t, set sizes and strides to all ones, and set
 * storage_offset to 0, set dtype to WHOLEMEMORY_DT_UNKNOWN, set dim to 0.
 * @param p_tensor_description : pointer to wholememory_tensor_description_t.
 */
void wholememory_initialize_tensor_desc(wholememory_tensor_description_t* p_tensor_description);

/**
 * Copy array description to tensor description
 * @param p_matrix_description : pointer to wholememory_matrix_description_t.
 * @param p_array_description : pointer to wholememory_array_description_t.
 */
void wholememory_copy_array_desc_to_matrix(wholememory_matrix_description_t* p_matrix_description,
                                           wholememory_array_description_t* p_array_description);

/*!
 * Copy array description to tensor description
 * @param p_tensor_description : pointer to wholememory_tensor_description_t.
 * @param p_array_description : pointer to wholememory_array_description_t.
 */
void wholememory_copy_array_desc_to_tensor(wholememory_tensor_description_t* p_tensor_description,
                                           wholememory_array_description_t* p_array_description);

/*!
 * Copy matrix description to tensor description
 * @param p_tensor_description : pointer to wholememory_tensor_description_t.
 * @param p_matrix_description : pointer to wholememory_matrix_description_t.
 */
void wholememory_copy_matrix_desc_to_tensor(wholememory_tensor_description_t* p_tensor_description,
                                            wholememory_matrix_description_t* p_matrix_description);
/*!
 * Convert tensor description to array description
 * @param p_array_description : pointer to wholememory_array_description_t.
 * @param p_tensor_description : pointer to wholememory_tensor_description_t.
 * @return : Return true if convertible else false.
 */
bool wholememory_convert_tensor_desc_to_array(
  wholememory_array_description_t* p_array_description,
  wholememory_tensor_description_t* p_tensor_description);

/*!
 * Convert tensor description to matrix description
 * @param p_matrix_description : pointer to wholememory_matrix_description_t.
 * @param p_tensor_description : pointer to wholememory_tensor_description_t.
 * @return : Return true if convertible else false.
 */
bool wholememory_convert_tensor_desc_to_matrix(
  wholememory_matrix_description_t* p_matrix_description,
  wholememory_tensor_description_t* p_tensor_description);

/*!
 * Get total element count from array description.
 * @param p_array_description : pointer to wholememory_array_description_t.
 * @return : Return element count.
 */
int64_t wholememory_get_memory_element_count_from_array(
  wholememory_array_description_t* p_array_description);

/*!
 * Get total memory size from array description.
 * @param p_array_description : pointer to wholememory_array_description_t.
 * @return : Return memory size.
 */
int64_t wholememory_get_memory_size_from_array(
  wholememory_array_description_t* p_array_description);

/*!
 * Get total element count from matrix description.
 * @param p_matrix_description : pointer to wholememory_matrix_description_t.
 * @return : Return element count.
 */
int64_t wholememory_get_memory_element_count_from_matrix(
  wholememory_matrix_description_t* p_matrix_description);

/*!
 * Get total memory size from matrix description.
 * @param p_matrix_description : pointer to wholememory_matrix_description_t.
 * @return : Return memory size.
 */
int64_t wholememory_get_memory_size_from_matrix(
  wholememory_matrix_description_t* p_matrix_description);

/*!
 * Get total element count from tensor description.
 * @param p_tensor_description : pointer to wholememory_tensor_description_t.
 * @return : Return element count.
 */
int64_t wholememory_get_memory_element_count_from_tensor(
  wholememory_tensor_description_t* p_tensor_description);

/*!
 * Get total memory size from tensor description.
 * @param p_tensor_description : pointer to wholememory_tensor_description_t.
 * @return : Return memory size.
 */
int64_t wholememory_get_memory_size_from_tensor(
  wholememory_tensor_description_t* p_tensor_description);

/**
 * Squeeze tensor
 * @param p_tensor_description : pointer to wholememory_tensor_description_t
 * @param dim : which dim to squeeze
 * @return : true if success else false
 */
bool wholememory_squeeze_tensor(wholememory_tensor_description_t* p_tensor_description, int dim);

/**
 * Unsqueeze tensor
 * @param p_tensor_description : pointer to wholememory_tensor_description_t
 * @param dim : unsqueeze at which dim
 * @return : true if success else false
 */
bool wholememory_unsqueeze_tensor(wholememory_tensor_description_t* p_tensor_description, int dim);

#ifdef __cplusplus
}
#endif
