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
#include "spadd_gat_csr_func.cuh"
#include <wholememory/wholememory.h>

#include "wholememory_ops/register.hpp"

namespace graph_ops {

REGISTER_DISPATCH_ONE_TYPE(SpADDGATCSRForward, spadd_gat_csr_forward_func, HALF_FLOAT)
REGISTER_DISPATCH_ONE_TYPE(SpADDGATCSRBackward, spadd_gat_csr_backward_func, HALF_FLOAT)
wholememory_error_code_t spadd_gat_csr_forward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* edge_weight_left_ptr,
  wholememory_matrix_description_t edge_weight_left_matrix_desc,
  void* edge_weight_right_ptr,
  wholememory_matrix_description_t edge_weight_right_matrix_desc,
  void* output_score_ptr,
  wholememory_matrix_description_t output_score_matrix_desc,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(edge_weight_left_matrix_desc.dtype,
                      SpADDGATCSRForward,
                      csr_row_ptr,
                      csr_row_ptr_array_desc,
                      csr_col_ptr,
                      csr_col_ptr_array_desc,
                      edge_weight_left_ptr,
                      edge_weight_left_matrix_desc,
                      edge_weight_right_ptr,
                      edge_weight_right_matrix_desc,
                      output_score_ptr,
                      output_score_matrix_desc,
                      stream);

  } catch (const wholememory::cuda_error& rle) {
    // WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t spadd_gat_csr_backward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* grad_score_ptr,
  wholememory_matrix_description_t grad_score_matrix_desc,
  void* output_grad_edge_weight_left_ptr,
  wholememory_matrix_description_t edge_weight_left_matrix_desc,
  void* output_grad_edge_weight_right_ptr,
  wholememory_matrix_description_t edge_weight_right_matrix_desc,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(edge_weight_left_matrix_desc.dtype,
                      SpADDGATCSRBackward,
                      csr_row_ptr,
                      csr_row_ptr_array_desc,
                      csr_col_ptr,
                      csr_col_ptr_array_desc,
                      grad_score_ptr,
                      grad_score_matrix_desc,
                      output_grad_edge_weight_left_ptr,
                      edge_weight_left_matrix_desc,
                      output_grad_edge_weight_right_ptr,
                      edge_weight_right_matrix_desc,
                      stream);

  } catch (const wholememory::cuda_error& rle) {
    // WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace graph_ops
