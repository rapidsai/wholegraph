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
#include <wholememory/wholememory_op.h>

#include <wholememory_ops/gather_op_impl.h>

#include "error.hpp"
#include "logger.hpp"

wholememory_error_code_t wholememory_gather(wholememory_tensor_t wholememory_tensor,
                                            wholememory_tensor_t indices_tensor,
                                            wholememory_tensor_t output_tensor,
                                            wholememory_env_func_t* p_env_fns,
                                            void* stream,
                                            int gather_sms)
{
  bool const has_handle                         = wholememory_tensor_has_handle(wholememory_tensor);
  wholememory_memory_type_t memory_type         = WHOLEMEMORY_MT_NONE;
  wholememory_memory_location_t memory_location = WHOLEMEMORY_ML_NONE;
  if (has_handle) {
    auto memory_handle = wholememory_tensor_get_memory_handle(wholememory_tensor);
    memory_type        = wholememory_get_memory_type(memory_handle);
    memory_location    = wholememory_get_memory_location(memory_handle);
  }
  wholememory_matrix_description_t matrix_description;
  auto tensor_description = *wholememory_tensor_get_tensor_description(wholememory_tensor);
  if (tensor_description.dim != 1 && tensor_description.dim != 2) {
    WHOLEMEMORY_ERROR("wholememory_tensor should be 1D or 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description.dim == 1) {
    if (!wholememory_unsqueeze_tensor(&tensor_description, 1)) {
      WHOLEMEMORY_ERROR("Input 1D wholememory_tensor unsqueeze to 2D failed.");
      return WHOLEMEMORY_LOGIC_ERROR;
    }
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&matrix_description, &tensor_description)) {
    WHOLEMEMORY_ERROR("Input wholememory_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (wholememory_tensor_get_tensor_description(indices_tensor)->dim != 1) {
    WHOLEMEMORY_ERROR("indices tensor should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_tensor_description_t output_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_tensor);
  if (output_tensor_desc.dim != tensor_description.dim) {
    WHOLEMEMORY_ERROR("output tensor should be same dim as wholememory_tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_tensor_desc.dim == 1) {
    if (!wholememory_unsqueeze_tensor(&output_tensor_desc, 1)) {
      WHOLEMEMORY_ERROR("Output 1D wholememory_tensor unsqueeze to 2D failed.");
      return WHOLEMEMORY_LOGIC_ERROR;
    }
  }
  void* indices = wholememory_tensor_get_data_pointer(indices_tensor);
  void* output  = wholememory_tensor_get_data_pointer(output_tensor);
  wholememory_array_description_t indices_desc;
  wholememory_matrix_description_t output_desc;
  if (!wholememory_convert_tensor_desc_to_array(
        &indices_desc, wholememory_tensor_get_tensor_description(indices_tensor))) {
    WHOLEMEMORY_ERROR("Convert indices tensor to array failed.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&output_desc, &output_tensor_desc)) {
    WHOLEMEMORY_ERROR("Convert output tensor to matrix failed.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (has_handle && memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
    return wholememory_ops::wholememory_gather_distributed(
      wholememory_tensor_get_memory_handle(wholememory_tensor),
      matrix_description,
      indices,
      indices_desc,
      output,
      output_desc,
      p_env_fns,
      static_cast<cudaStream_t>(stream),
      gather_sms);
  }

  if (has_handle && memory_type == WHOLEMEMORY_MT_HIERARCHY) {
    return wholememory_ops::wholememory_gather_hierarchy(
      wholememory_tensor_get_memory_handle(wholememory_tensor),
      matrix_description,
      indices,
      indices_desc,
      output,
      output_desc,
      p_env_fns,
      static_cast<cudaStream_t>(stream),
      gather_sms);
  }

  WHOLEMEMORY_EXPECTS_NOTHROW(!has_handle || memory_type == WHOLEMEMORY_MT_CHUNKED ||
                                memory_type == WHOLEMEMORY_MT_CONTINUOUS,
                              "Memory type not supported.");

  wholememory_gref_t gref;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_get_global_reference(wholememory_tensor, &gref));

  int64_t entry_size =
    tensor_description.sizes[1] * wholememory_dtype_get_element_size(tensor_description.dtype);
  bool gather_with_sorted_ids =
    (memory_location == WHOLEMEMORY_ML_HOST) && (entry_size <= 512) &&
    (memory_type == WHOLEMEMORY_MT_CHUNKED || memory_type == WHOLEMEMORY_MT_CONTINUOUS);
  return wholememory_ops::wholememory_gather_mapped(gref,
                                                    matrix_description,
                                                    indices,
                                                    indices_desc,
                                                    output,
                                                    output_desc,
                                                    gather_with_sorted_ids,
                                                    p_env_fns,
                                                    static_cast<cudaStream_t>(stream),
                                                    gather_sms);
}
