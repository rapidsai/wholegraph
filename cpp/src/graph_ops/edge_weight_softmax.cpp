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
#include <graph_ops/edge_weight_softmax_impl.h>
#include <wholememory/graph_op.h>

wholememory_error_code_t edge_weight_softmax_csr_forward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t edge_weight_tensor,
  wholememory_tensor_t output_edge_weight_softmax_tensor,
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
  auto edge_weight_tensor_desc = *wholememory_tensor_get_tensor_description(edge_weight_tensor);
  if (edge_weight_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("edge_weight_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  if (edge_weight_tensor_desc.dtype != WHOLEMEMORY_DT_HALF &&
      edge_weight_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("edge_weight_tensor should be half or float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto output_edge_weight_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_edge_weight_softmax_tensor);
  if (output_edge_weight_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Output edge_weight_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  if (output_edge_weight_tensor_desc.dtype != WHOLEMEMORY_DT_HALF &&
      output_edge_weight_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("Output edge_weight_tensor should be half or float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (edge_weight_tensor_desc.dtype != output_edge_weight_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Output edge_weight_tensor's dtype should be the same with input edge_weight_tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  wholememory_array_description_t csr_row_ptr_array_desc;
  if (!wholememory_convert_tensor_desc_to_array(&csr_row_ptr_array_desc,
                                                &csr_row_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  wholememory_matrix_description_t edge_weight_matrix_desc, output_edge_weight_matrix_desc;
  if (!wholememory_convert_tensor_desc_to_matrix(&edge_weight_matrix_desc,
                                                 &edge_weight_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input edge_weight_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (!wholememory_convert_tensor_desc_to_matrix(&output_edge_weight_matrix_desc,
                                                 &output_edge_weight_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output edge_weight_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  void* csr_row_ptr     = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* edge_weight_ptr = wholememory_tensor_get_data_pointer(edge_weight_tensor);
  void* output_edge_weight_ptr =
    wholememory_tensor_get_data_pointer(output_edge_weight_softmax_tensor);

  return graph_ops::edge_weight_softmax_forward_impl(static_cast<int*>(csr_row_ptr),
                                                     csr_row_ptr_array_desc,
                                                     edge_weight_ptr,
                                                     edge_weight_matrix_desc,
                                                     output_edge_weight_ptr,
                                                     output_edge_weight_matrix_desc,
                                                     static_cast<cudaStream_t>(stream));
}

wholememory_error_code_t edge_weight_softmax_csr_backward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t edge_weight_tensor,
  wholememory_tensor_t grad_edge_weight_softmax_tensor,
  wholememory_tensor_t output_grad_edge_weight_tensor,
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
  auto edge_weight_tensor_desc = *wholememory_tensor_get_tensor_description(edge_weight_tensor);
  if (edge_weight_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("edge_weight_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  if (edge_weight_tensor_desc.dtype != WHOLEMEMORY_DT_HALF &&
      edge_weight_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("edge_weight_tensor should be half or float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto grad_edge_weight_softmax_tensor_desc =
    *wholememory_tensor_get_tensor_description(grad_edge_weight_softmax_tensor);
  if (grad_edge_weight_softmax_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Input grad_edge_weight_softmax_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  if (grad_edge_weight_softmax_tensor_desc.dtype != WHOLEMEMORY_DT_HALF &&
      grad_edge_weight_softmax_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("Input grad_edge_weight_softmax_tensor should be half or float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (edge_weight_tensor_desc.dtype != grad_edge_weight_softmax_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Input edge_weight_tensor's dtype should be the same with input "
      "grad_edge_weight_softmax_tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto output_grad_edge_weight_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_grad_edge_weight_tensor);
  if (output_grad_edge_weight_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Output grad_edge_weight_tensor should be 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  if (output_grad_edge_weight_tensor_desc.dtype != WHOLEMEMORY_DT_HALF &&
      output_grad_edge_weight_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("Output grad_edge_weight_tensor should be half or float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_grad_edge_weight_tensor_desc.dtype != edge_weight_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR(
      "Output output_grad_edge_weight_tensor's dtype should be the same with input "
      "edge_weight_tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  wholememory_array_description_t csr_row_ptr_array_desc;
  if (!wholememory_convert_tensor_desc_to_array(&csr_row_ptr_array_desc,
                                                &csr_row_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  wholememory_matrix_description_t edge_weight_matrix_desc, grad_edge_weight_softmax_matrix_desc,
    output_grad_edge_weight_matrix_desc;
  if (!wholememory_convert_tensor_desc_to_matrix(&edge_weight_matrix_desc,
                                                 &edge_weight_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input edge_weight_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&grad_edge_weight_softmax_matrix_desc,
                                                 &grad_edge_weight_softmax_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input grad_edge_weight_softmax_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&output_grad_edge_weight_matrix_desc,
                                                 &output_grad_edge_weight_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output output_grad_edge_weight_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  void* csr_row_ptr     = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* edge_weight_ptr = wholememory_tensor_get_data_pointer(edge_weight_tensor);
  void* grad_edge_weight_softmax_ptr =
    wholememory_tensor_get_data_pointer(grad_edge_weight_softmax_tensor);
  void* output_grad_edge_weight_ptr =
    wholememory_tensor_get_data_pointer(output_grad_edge_weight_tensor);

  return graph_ops::edge_weight_softmax_backward_impl(static_cast<int*>(csr_row_ptr),
                                                      csr_row_ptr_array_desc,
                                                      edge_weight_ptr,
                                                      edge_weight_matrix_desc,
                                                      grad_edge_weight_softmax_ptr,
                                                      grad_edge_weight_softmax_matrix_desc,
                                                      output_grad_edge_weight_ptr,
                                                      output_grad_edge_weight_matrix_desc,
                                                      static_cast<cudaStream_t>(stream));
}
