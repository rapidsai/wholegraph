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
#include <cuda_runtime_api.h>

#include "spmm_csr_no_weight_func.cuh"
#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "wholememory_ops/register.hpp"

namespace graph_ops {

REGISTER_DISPATCH_ONE_TYPE(SpMMCSRNoWeightForward, spmm_csr_no_weight_forward_func, HALF_FLOAT)
REGISTER_DISPATCH_ONE_TYPE(SpMMCSRNoWeightBackward, spmm_csr_no_weight_backward_func, HALF_FLOAT)

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
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(feature_desc.dtype,
                      SpMMCSRNoWeightForward,
                      csr_row_ptr,
                      csr_row_ptr_desc,
                      csr_col_ptr,
                      csr_col_ptr_desc,
                      feature_ptr,
                      feature_desc,
                      aggregator,
                      output_ptr,
                      output_desc,
                      stream);

  } catch (const wholememory::cuda_error& wce) {
    // WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

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
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(input_grad_tensor_desc.dtype,
                      SpMMCSRNoWeightBackward,
                      csr_row_ptr,
                      csr_row_ptr_desc,
                      csr_col_ptr,
                      csr_col_ptr_desc,
                      input_grad_ptr,
                      input_grad_tensor_desc,
                      aggregator,
                      output_grad_feature_ptr,
                      output_grad_feature_tensor_desc,
                      stream);

  } catch (const wholememory::cuda_error& wce) {
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
