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
#include <graph_ops/gspmm_csr_weighted_impl.h>
#include <wholememory/graph_op.h>

wholememory_error_code_t gspmm_csr_weighted_forward(wholememory_tensor_t csr_row_ptr_tensor,
                                                    wholememory_tensor_t csr_col_ptr_tensor,
                                                    wholememory_tensor_t edge_weight_tensor,
                                                    wholememory_tensor_t feature_tensor,
                                                    wholememory_tensor_t output_feature_tensor,
                                                    void* stream)
{
  auto csr_row_ptr_tensor_desc = *wholememory_tensor_get_tensor_description(csr_row_ptr_tensor);
  if (csr_row_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("csr_row_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_row_ptr_tensor_desc.dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("csr_row_ptr_tensor should be int tensor.");
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

  auto feature_tensor_desc = *wholememory_tensor_get_tensor_description(feature_tensor);
  if (feature_tensor_desc.dim != 3) {
    WHOLEMEMORY_ERROR("Input feature_tensor should be 3D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (feature_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT &&
      feature_tensor_desc.dtype != WHOLEMEMORY_DT_HALF) {
    WHOLEMEMORY_ERROR("Input feature_tensor should be float or half tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto edge_weight_tensor_desc = *wholememory_tensor_get_tensor_description(edge_weight_tensor);
  if (edge_weight_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Input edge_weight_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (feature_tensor_desc.dtype != edge_weight_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR("Input edge_weight_tensor should be the same with feature_tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  if (edge_weight_tensor_desc.sizes[0] != csr_col_ptr_tensor_desc.sizes[0]) {
    WHOLEMEMORY_ERROR(
      "Input edge_weight_tensor size(0) should be the same with csr_col_ptr_tensor size(0).");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (feature_tensor_desc.sizes[1] != edge_weight_tensor_desc.sizes[1]) {
    WHOLEMEMORY_ERROR(
      "Input edge_weight_tensor size(1) should be the same with feature_tensor size(1).");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto output_feature_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_feature_tensor);
  if (output_feature_tensor_desc.dim != 3) {
    WHOLEMEMORY_ERROR("Output output_feature_tensor should be 3D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_feature_tensor_desc.dtype != feature_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Output output_feature_tensor dtype should be the same with feature_tensor dtype.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_feature_tensor_desc.sizes[0] != csr_row_ptr_tensor_desc.sizes[0] - 1) {
    WHOLEMEMORY_ERROR(
      "Output output_feature_tensor sizes(0) should be the same with csr_row_ptr sizes(0) - 1.");
    fprintf(stderr,
            "output_feature_size[0] = %ld, csr_row_ptr_size[0] = %ld\n",
            output_feature_tensor_desc.sizes[0],
            csr_row_ptr_tensor_desc.sizes[0]);
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_feature_tensor_desc.sizes[1] != feature_tensor_desc.sizes[1]) {
    WHOLEMEMORY_ERROR(
      "Output output_feature_tensor sizes(1) should be the same with feature_tensor sizes(1).");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_feature_tensor_desc.sizes[2] != feature_tensor_desc.sizes[2]) {
    WHOLEMEMORY_ERROR(
      "Output output_feature_tensor sizes(2) should be the same with feature_tensor sizes(2).");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  wholememory_array_description_t csr_row_ptr_array_desc, csr_col_ptr_array_desc;
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
  void* csr_row_ptr        = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* csr_col_ptr        = wholememory_tensor_get_data_pointer(csr_col_ptr_tensor);
  void* feature_ptr        = wholememory_tensor_get_data_pointer(feature_tensor);
  void* edge_weight_ptr    = wholememory_tensor_get_data_pointer(edge_weight_tensor);
  void* output_feature_ptr = wholememory_tensor_get_data_pointer(output_feature_tensor);

  return graph_ops::gspmm_csr_weighted_forward_impl(static_cast<int*>(csr_row_ptr),
                                                    csr_row_ptr_array_desc,
                                                    static_cast<int*>(csr_col_ptr),
                                                    csr_col_ptr_array_desc,
                                                    edge_weight_ptr,
                                                    edge_weight_tensor_desc,
                                                    feature_ptr,
                                                    feature_tensor_desc,
                                                    output_feature_ptr,
                                                    output_feature_tensor_desc,
                                                    static_cast<cudaStream_t>(stream));
}

wholememory_error_code_t gspmm_csr_weighted_backward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t csr_col_ptr_tensor,
  wholememory_tensor_t edge_weight_tensor,
  wholememory_tensor_t feature_tensor,
  wholememory_tensor_t grad_feature_tensor,
  wholememory_tensor_t output_grad_edge_weight_tensor,
  wholememory_tensor_t output_grad_feature_tensor,
  void* stream)
{
  auto csr_row_ptr_tensor_desc = *wholememory_tensor_get_tensor_description(csr_row_ptr_tensor);
  if (csr_row_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("csr_row_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_row_ptr_tensor_desc.dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("csr_row_ptr_tensor should be int tensor.");
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
  auto feature_tensor_desc = *wholememory_tensor_get_tensor_description(feature_tensor);
  if (feature_tensor_desc.dim != 3) {
    WHOLEMEMORY_ERROR("Input feature_tensor should be 3D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (feature_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT &&
      feature_tensor_desc.dtype != WHOLEMEMORY_DT_HALF) {
    WHOLEMEMORY_ERROR("Input feature_tensor should be float or half tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto edge_weight_tensor_desc = *wholememory_tensor_get_tensor_description(edge_weight_tensor);
  if (edge_weight_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Input edge_weight_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  if (feature_tensor_desc.dtype != edge_weight_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR("Input edge_weight_tensor should be the same with feature_tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (edge_weight_tensor_desc.sizes[0] != csr_col_ptr_tensor_desc.sizes[0]) {
    WHOLEMEMORY_ERROR(
      "Input edge_weight_tensor sizes[0] should be the same with csr_col_ptr sizes[0].");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto grad_feature_tensor_desc = *wholememory_tensor_get_tensor_description(grad_feature_tensor);
  if (grad_feature_tensor_desc.dim != feature_tensor_desc.dim) {
    WHOLEMEMORY_ERROR("Input grad_feature_tensor dim should be the same with feature_tensor dim.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  if (grad_feature_tensor_desc.dtype != feature_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Input grad_feature_tensor dtype should be the same with feature_tensor dtype.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto output_grad_feature_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_grad_feature_tensor);
  if (output_grad_feature_tensor_desc.dim != 3 && output_grad_feature_tensor_desc.dim != 0) {
    WHOLEMEMORY_ERROR("Output output_grad_feature_tensor should be 3D tensor or None.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_grad_feature_tensor_desc.dim == 3 &&
      (output_grad_feature_tensor_desc.sizes[0] != feature_tensor_desc.sizes[0] ||
       output_grad_feature_tensor_desc.sizes[1] != feature_tensor_desc.sizes[1] ||
       output_grad_feature_tensor_desc.sizes[2] != feature_tensor_desc.sizes[2])) {
    WHOLEMEMORY_ERROR(
      "Output output_grad_feature_tensor sizes should be the same with feature_tensor sizes.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_grad_feature_tensor_desc.dim == 3 &&
      output_grad_feature_tensor_desc.dtype != feature_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Output output_grad_feature_tensor dtype should be the same with feature tensor dtype.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto output_grad_edge_weight_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_grad_edge_weight_tensor);
  if (output_grad_edge_weight_tensor_desc.dim != 2 &&
      output_grad_edge_weight_tensor_desc.dim != 0) {
    WHOLEMEMORY_ERROR("Output output_grad_edge_weight_tensor should be 2D tensor or None.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_grad_edge_weight_tensor_desc.dim == 2 &&
      output_grad_edge_weight_tensor_desc.dtype != feature_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Output output_grad_edge_weight_tensor_desc dtype should be the same with feature tensor "
      "dtype.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_grad_edge_weight_tensor_desc.dim == 2 &&
      (output_grad_edge_weight_tensor_desc.sizes[0] != edge_weight_tensor_desc.sizes[0] ||
       output_grad_edge_weight_tensor_desc.sizes[1] != edge_weight_tensor_desc.sizes[1])) {
    WHOLEMEMORY_ERROR(
      "Output output_grad_edge_weight_tensor_desc sizes should be the same with edge_weight tensor "
      "sizes.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  wholememory_array_description_t csr_row_ptr_array_desc, csr_col_ptr_array_desc;
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

  void* csr_row_ptr             = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* csr_col_ptr             = wholememory_tensor_get_data_pointer(csr_col_ptr_tensor);
  void* feature_ptr             = wholememory_tensor_get_data_pointer(feature_tensor);
  void* edge_weight_ptr         = wholememory_tensor_get_data_pointer(edge_weight_tensor);
  void* grad_feature_ptr        = wholememory_tensor_get_data_pointer(grad_feature_tensor);
  void* output_grad_feature_ptr = wholememory_tensor_get_data_pointer(output_grad_feature_tensor);
  void* output_grad_edge_weight_ptr =
    wholememory_tensor_get_data_pointer(output_grad_edge_weight_tensor);

  return graph_ops::gspmm_csr_weighted_backward_impl(static_cast<int*>(csr_row_ptr),
                                                     csr_row_ptr_array_desc,
                                                     static_cast<int*>(csr_col_ptr),
                                                     csr_col_ptr_array_desc,
                                                     edge_weight_ptr,
                                                     edge_weight_tensor_desc,
                                                     feature_ptr,
                                                     feature_tensor_desc,
                                                     grad_feature_ptr,
                                                     grad_feature_tensor_desc,
                                                     output_grad_edge_weight_ptr,
                                                     output_grad_edge_weight_tensor_desc,
                                                     output_grad_feature_ptr,
                                                     output_grad_feature_tensor_desc,
                                                     static_cast<cudaStream_t>(stream));
}
