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
wholememory_error_code_t spmm_csr_no_weight_forward_mapped(
  void* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* feature_ptr,
  wholememory_matrix_description_t feature_desc,
  int aggregator,
  void* output_ptr,
  wholememory_matrix_description_t output_desc,
  cudaStream_t stream);

wholememory_error_code_t spmm_csr_no_weight_backward_mapped(
  void* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* input_grad_ptr,
  wholememory_tensor_description_t input_grad_tensor_desc,
  int aggregator,
  void* output_grad_feature_ptr,
  wholememory_tensor_description_t output_grad_feature_tensor_desc,
  cudaStream_t stream);
}  // namespace graph_ops
