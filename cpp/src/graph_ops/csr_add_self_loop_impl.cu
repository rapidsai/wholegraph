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
#include "csr_add_self_loop_func.cuh"
#include <wholememory/wholememory.h>

#include "wholememory_ops/register.hpp"

namespace graph_ops {
wholememory_error_code_t csr_add_self_loop_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  int* output_csr_row_ptr,
  wholememory_array_description_t output_csr_row_ptr_array_desc,
  int* output_csr_col_ptr,
  wholememory_array_description_t output_csr_col_ptr_array_desc,
  cudaStream_t stream)
{
  try {
    csr_add_self_loop_func(csr_row_ptr,
                           csr_row_ptr_array_desc,
                           csr_col_ptr,
                           csr_col_ptr_array_desc,
                           output_csr_row_ptr,
                           output_csr_row_ptr_array_desc,
                           output_csr_col_ptr,
                           output_csr_col_ptr_array_desc,
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
