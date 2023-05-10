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
#include <cuda_runtime_api.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace graph_ops {
wholememory_error_code_t gspmm_csr_weighted_forward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* edge_weight_tensor,
  wholememory_tensor_description_t edge_weight_tensor_desc,
  void* feature_ptr,
  wholememory_tensor_description_t feature_tensor_desc,
  void* output_feature_ptr,
  wholememory_tensor_description_t output_feature_tensor_desc,
  cudaStream_t stream);

wholememory_error_code_t gspmm_csr_weighted_backward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_tensor_description_t edge_weight_tensor_desc,
  void* feature_ptr,
  wholememory_tensor_description_t feature_tensor_desc,
  void* grad_feature_ptr,
  wholememory_tensor_description_t grad_feature_tensor,
  void* output_grad_edge_weight_ptr,
  wholememory_tensor_description_t output_grad_edge_weight_tensor_desc,
  void* output_grad_feature_ptr,
  wholememory_tensor_description_t output_grad_feature_tensor_desc,
  cudaStream_t stream);

}  // namespace graph_ops
