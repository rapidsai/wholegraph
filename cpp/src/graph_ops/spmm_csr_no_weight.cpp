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
#include "error.hpp"
#include "logger.hpp"
#include <graph_ops/spmm_csr_no_weight_impl.h>
#include <wholememory/graph_op.h>

wholememory_error_code_t spmm_csr_no_weight_forward(wholememory_tensor_t csr_row_ptr_tensor,
                                                    wholememory_tensor_t csr_col_ptr_tensor,
                                                    wholememory_tensor_t feature_tensor,
                                                    int64_t aggregator,
                                                    wholememory_tensor_t output_tensor,
                                                    void* stream)
{
  wholememory_tensor_description_t csr_row_ptr_tensor_desc =
    *wholememory_tensor_get_tensor_description(csr_row_ptr_tensor);
  wholememory_tensor_description_t csr_col_ptr_tensor_desc =
    *wholememory_tensor_get_tensor_description(csr_col_ptr_tensor);
  wholememory_tensor_description_t feature_tensor_desc =
    *wholememory_tensor_get_tensor_description(feature_tensor);
  wholememory_tensor_description_t output_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_tensor);
  if (csr_row_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input csr_col_ptr should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (feature_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Input feature_tensor should be 2D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Output output_tensor should be 2D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  wholememory_array_description_t csr_row_ptr_desc, csr_col_ptr_desc;
  if (!wholememory_convert_tensor_desc_to_array(&csr_row_ptr_desc, &csr_row_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (!wholememory_convert_tensor_desc_to_array(&csr_col_ptr_desc, &csr_col_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_col_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  wholememory_matrix_description_t feature_desc, output_desc;
  if (!wholememory_convert_tensor_desc_to_matrix(&feature_desc, &feature_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input feature_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&output_desc, &output_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output output_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (output_desc.dtype != feature_desc.dtype) {
    WHOLEMEMORY_ERROR("output_tensor dtype should the same with feature tensor dtype.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  void* csr_row_ptr = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* csr_col_ptr = wholememory_tensor_get_data_pointer(csr_col_ptr_tensor);
  void* feature_ptr = wholememory_tensor_get_data_pointer(feature_tensor);
  void* output_ptr  = wholememory_tensor_get_data_pointer(output_tensor);

  return graph_ops::spmm_csr_no_weight_forward_mapped(csr_row_ptr,
                                                      csr_row_ptr_desc,
                                                      csr_col_ptr,
                                                      csr_col_ptr_desc,
                                                      feature_ptr,
                                                      feature_desc,
                                                      aggregator,
                                                      output_ptr,
                                                      output_desc,
                                                      static_cast<cudaStream_t>(stream));
}

wholememory_error_code_t spmm_csr_no_weight_backward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t csr_col_ptr_tensor,
  wholememory_tensor_t input_grad_tensor,
  int64_t aggregator,
  wholememory_tensor_t output_grad_feature_tensor,
  void* stream)
{
  wholememory_tensor_description_t csr_row_ptr_tensor_desc =
    *wholememory_tensor_get_tensor_description(csr_row_ptr_tensor);
  wholememory_tensor_description_t csr_col_ptr_tensor_desc =
    *wholememory_tensor_get_tensor_description(csr_col_ptr_tensor);
  wholememory_tensor_description_t input_grad_tensor_desc =
    *wholememory_tensor_get_tensor_description(input_grad_tensor);
  wholememory_tensor_description_t output_grad_feature_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_grad_feature_tensor);
  if (csr_row_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input csr_col_ptr should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (input_grad_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Input input_grad_tensor should be 2D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_grad_feature_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Output output_grad_feature_tensor should be 2D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_array_description_t csr_row_ptr_desc, csr_col_ptr_desc;
  if (!wholememory_convert_tensor_desc_to_array(&csr_row_ptr_desc, &csr_row_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (!wholememory_convert_tensor_desc_to_array(&csr_col_ptr_desc, &csr_col_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_col_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (input_grad_tensor_desc.dtype != output_grad_feature_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "output_grad_feature_tensor dtype should the same with input_grad_tensor dtype.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (input_grad_tensor_desc.sizes[1] != output_grad_feature_tensor_desc.sizes[1]) {
    WHOLEMEMORY_ERROR("input_grad_tensor size[1] should the same with input_grad_tensor sizes[1].");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  void* csr_row_ptr             = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* csr_col_ptr             = wholememory_tensor_get_data_pointer(csr_col_ptr_tensor);
  void* input_grad_ptr          = wholememory_tensor_get_data_pointer(input_grad_tensor);
  void* output_grad_feature_ptr = wholememory_tensor_get_data_pointer(output_grad_feature_tensor);
  return graph_ops::spmm_csr_no_weight_backward_mapped(csr_row_ptr,
                                                       csr_row_ptr_desc,
                                                       csr_col_ptr,
                                                       csr_col_ptr_desc,
                                                       input_grad_ptr,
                                                       input_grad_tensor_desc,
                                                       aggregator,
                                                       output_grad_feature_ptr,
                                                       output_grad_feature_tensor_desc,
                                                       static_cast<cudaStream_t>(stream));
}
