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
#include <graph_ops/spadd_gat_csr_impl.h>
#include <wholememory/graph_op.h>

wholememory_error_code_t spadd_gat_csr_foward(wholememory_tensor_t csr_row_ptr_tensor,
                                              wholememory_tensor_t csr_col_ptr_tensor,
                                              wholememory_tensor_t edge_weight_left_tensor,
                                              wholememory_tensor_t edge_weight_right_tensor,
                                              wholememory_tensor_t output_score_tensor,
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
    WHOLEMEMORY_ERROR("csr_col_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_desc.dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("csr_col_ptr_tensor should be int tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto edge_weight_left_tensor_desc =
    *wholememory_tensor_get_tensor_description(edge_weight_left_tensor);
  if (edge_weight_left_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("edge_weight_left_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (edge_weight_left_tensor_desc.dtype != WHOLEMEMORY_DT_HALF &&
      edge_weight_left_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("edge_weight_left_tensor should be half or float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto edge_weight_right_tensor_desc =
    *wholememory_tensor_get_tensor_description(edge_weight_right_tensor);
  if (edge_weight_right_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("edge_weight_right_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (edge_weight_right_tensor_desc.dtype != WHOLEMEMORY_DT_HALF &&
      edge_weight_right_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("edge_weight_right_tensor should be half or float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto output_score_tensor_desc = *wholememory_tensor_get_tensor_description(output_score_tensor);
  if (output_score_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("output_score_tensor should be 2D tensor.");
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

  if (edge_weight_left_tensor_desc.dtype != edge_weight_right_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Input edge_weight_tensor_left's dtype is not the same with edge_weight_tensor_right's "
      "dtype.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (edge_weight_left_tensor_desc.dtype != output_score_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Input edge_weight_tensor's dtype should be the same with outpu_scaore's dtype.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (edge_weight_left_tensor_desc.sizes[1] != edge_weight_right_tensor_desc.sizes[1]) {
    WHOLEMEMORY_ERROR(
      "Input edge_weight_tensor_left sizes[1] should be the same with edge_weight_tensor_right "
      "sizes[1].");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (csr_row_ptr_array_desc.size - 1 != edge_weight_left_tensor_desc.sizes[0]) {
    WHOLEMEMORY_ERROR(
      "Input edge_weight_tensor_left sizes[0] should be the same with csr_row_ptr size - 1.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  wholememory_matrix_description_t edge_weight_left_matrix_desc, edge_weight_right_matrix_desc,
    output_score_matrix_desc;
  if (!wholememory_convert_tensor_desc_to_matrix(&edge_weight_left_matrix_desc,
                                                 &edge_weight_left_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input edge_weight_left_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&edge_weight_right_matrix_desc,
                                                 &edge_weight_right_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input edge_weight_right_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&output_score_matrix_desc,
                                                 &output_score_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output output_score_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  void* csr_row_ptr           = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* csr_col_ptr           = wholememory_tensor_get_data_pointer(csr_col_ptr_tensor);
  void* edge_weight_left_ptr  = wholememory_tensor_get_data_pointer(edge_weight_left_tensor);
  void* edge_weight_right_ptr = wholememory_tensor_get_data_pointer(edge_weight_right_tensor);
  void* output_score_ptr      = wholememory_tensor_get_data_pointer(output_score_tensor);

  return graph_ops::spadd_gat_csr_forward_impl(static_cast<int*>(csr_row_ptr),
                                               csr_row_ptr_array_desc,
                                               static_cast<int*>(csr_col_ptr),
                                               csr_col_ptr_array_desc,
                                               edge_weight_left_ptr,
                                               edge_weight_left_matrix_desc,
                                               edge_weight_right_ptr,
                                               edge_weight_right_matrix_desc,
                                               output_score_ptr,
                                               output_score_matrix_desc,
                                               static_cast<cudaStream_t>(stream));
}

wholememory_error_code_t spadd_gat_csr_backward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t csr_col_ptr_tensor,
  wholememory_tensor_t grad_score_tensor,
  wholememory_tensor_t output_grad_edge_weight_left_tensor,
  wholememory_tensor_t output_grad_edge_weight_right_tensor,
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
    WHOLEMEMORY_ERROR("csr_col_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_desc.dtype != WHOLEMEMORY_DT_INT) {
    WHOLEMEMORY_ERROR("csr_col_ptr_tensor should be int tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto edge_weight_left_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_grad_edge_weight_left_tensor);
  if (edge_weight_left_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Output grad_edge_weight_left_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (edge_weight_left_tensor_desc.dtype != WHOLEMEMORY_DT_HALF &&
      edge_weight_left_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("Output grad_edge_weight_left_tensor should be half or float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto edge_weight_right_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_grad_edge_weight_right_tensor);
  if (edge_weight_right_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Output grad_edge_weight_right_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (edge_weight_right_tensor_desc.dtype != WHOLEMEMORY_DT_HALF &&
      edge_weight_right_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("Output grad_edge_weight_right_tensor should be half or float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto grad_score_tensor_desc = *wholememory_tensor_get_tensor_description(grad_score_tensor);
  if (grad_score_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("grad_score_tensor should be 2D tensor.");
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

  if (edge_weight_left_tensor_desc.dtype != edge_weight_right_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Output grad_edge_weight_tensor_left's dtype is not the same with "
      "grad_edge_weight_tensor_right's dtype.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (edge_weight_left_tensor_desc.dtype != grad_score_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Output grad_edge_weight_tensor's dtype should be the same with grad_score's dtype.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (edge_weight_left_tensor_desc.sizes[1] != edge_weight_right_tensor_desc.sizes[1]) {
    WHOLEMEMORY_ERROR(
      "Output grad_edge_weight_tensor_left sizes[1] should be the same with "
      "grad_edge_weight_tensor_right sizes[1].");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (csr_row_ptr_array_desc.size - 1 != edge_weight_left_tensor_desc.sizes[0]) {
    WHOLEMEMORY_ERROR(
      "Output grad_edge_weight_tensor_left sizes[0] should be the same with csr_row_ptr size - 1.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  wholememory_matrix_description_t edge_weight_left_matrix_desc, edge_weight_right_matrix_desc,
    grad_score_matrix_desc;
  if (!wholememory_convert_tensor_desc_to_matrix(&edge_weight_left_matrix_desc,
                                                 &edge_weight_left_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output grad_edge_weight_left_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&edge_weight_right_matrix_desc,
                                                 &edge_weight_right_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output grad_edge_weight_right_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&grad_score_matrix_desc,
                                                 &grad_score_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input grad_score_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  void* csr_row_ptr = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* csr_col_ptr = wholememory_tensor_get_data_pointer(csr_col_ptr_tensor);
  void* output_grad_edge_weight_left_ptr =
    wholememory_tensor_get_data_pointer(output_grad_edge_weight_left_tensor);
  void* output_grad_edge_weight_right_ptr =
    wholememory_tensor_get_data_pointer(output_grad_edge_weight_right_tensor);
  void* grad_score_ptr = wholememory_tensor_get_data_pointer(grad_score_tensor);

  return graph_ops::spadd_gat_csr_backward_impl(static_cast<int*>(csr_row_ptr),
                                                csr_row_ptr_array_desc,
                                                static_cast<int*>(csr_col_ptr),
                                                csr_col_ptr_array_desc,
                                                grad_score_ptr,
                                                grad_score_matrix_desc,
                                                output_grad_edge_weight_left_ptr,
                                                edge_weight_left_matrix_desc,
                                                output_grad_edge_weight_right_ptr,
                                                edge_weight_right_matrix_desc,
                                                static_cast<cudaStream_t>(stream));
}
