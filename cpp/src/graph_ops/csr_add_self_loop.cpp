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
#include "csr_add_self_loop_impl.h"
#include "error.hpp"
#include "logger.hpp"
#include <wholememory/graph_op.h>

wholememory_error_code_t csr_add_self_loop(wholememory_tensor_t csr_row_ptr_tensor,
                                           wholememory_tensor_t csr_col_ptr_tensor,
                                           wholememory_tensor_t output_csr_row_ptr_tensor,
                                           wholememory_tensor_t output_csr_col_ptr_tensor,
                                           void* stream)
{
  auto csr_row_ptr_tensor_desc = *wholememory_tensor_get_tensor_description(csr_row_ptr_tensor);
  if (csr_row_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_row_ptr_tensor_desc.dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr_tensor should be int tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto csr_col_ptr_tensor_desc = *wholememory_tensor_get_tensor_description(csr_col_ptr_tensor);
  if (csr_col_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input csr_col_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_desc.dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("Input csr_col_ptr_tensor should be int tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto output_csr_row_ptr_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_csr_row_ptr_tensor);
  if (output_csr_row_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Output output_csr_row_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_csr_row_ptr_tensor_desc.dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("Output output_csr_row_ptr_tensor should be int tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto output_csr_col_ptr_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_csr_col_ptr_tensor);
  if (output_csr_col_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Output output_csr_col_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_csr_col_ptr_tensor_desc.dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("Output output_csr_col_ptr_tensor should be int tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  wholememory_array_description_t csr_row_ptr_array_desc, csr_col_ptr_array_desc,
    output_csr_row_ptr_array_desc, output_csr_col_ptr_array_desc;
  if (!wholememory_convert_tensor_desc_to_array(&csr_row_ptr_array_desc,
                                                &csr_row_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_array(&csr_col_ptr_array_desc,
                                                &csr_col_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_col_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (!wholememory_convert_tensor_desc_to_array(&output_csr_row_ptr_array_desc,
                                                &output_csr_row_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output output_csr_row_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_array(&output_csr_col_ptr_array_desc,
                                                &output_csr_col_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output output_csr_col_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  void* csr_row_ptr        = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* csr_col_ptr        = wholememory_tensor_get_data_pointer(csr_col_ptr_tensor);
  void* output_csr_row_ptr = wholememory_tensor_get_data_pointer(output_csr_row_ptr_tensor);
  void* output_csr_col_ptr = wholememory_tensor_get_data_pointer(output_csr_col_ptr_tensor);
  return graph_ops::csr_add_self_loop_impl(static_cast<int*>(csr_row_ptr),
                                           csr_row_ptr_array_desc,
                                           static_cast<int*>(csr_col_ptr),
                                           csr_col_ptr_array_desc,
                                           static_cast<int*>(output_csr_row_ptr),
                                           output_csr_row_ptr_array_desc,
                                           static_cast<int*>(output_csr_col_ptr),
                                           output_csr_col_ptr_array_desc,
                                           static_cast<cudaStream_t>(stream));
}
