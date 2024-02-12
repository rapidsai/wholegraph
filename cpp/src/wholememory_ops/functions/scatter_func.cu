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
#include "gather_scatter_func.h"

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"

namespace wholememory_ops {

wholememory_error_code_t scatter_integer_int32_func(const void* input,
                                                    wholememory_matrix_description_t input_desc,
                                                    void* indices,
                                                    wholememory_array_description_t indices_desc,
                                                    wholememory_gref_t embedding_gref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    cudaStream_t stream,
                                                    int scatter_sms);
wholememory_error_code_t scatter_integer_int64_func(const void* input,
                                                    wholememory_matrix_description_t input_desc,
                                                    void* indices,
                                                    wholememory_array_description_t indices_desc,
                                                    wholememory_gref_t embedding_gref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    cudaStream_t stream,
                                                    int scatter_sms);
wholememory_error_code_t scatter_floating_int32_func(
  const void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_gref_t embedding_gref,
  wholememory_matrix_description_t embedding_desc,
  cudaStream_t stream,
  int scatter_sms);
wholememory_error_code_t scatter_floating_int64_func(
  const void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_gref_t embedding_gref,
  wholememory_matrix_description_t embedding_desc,
  cudaStream_t stream,
  int scatter_sms);

wholememory_error_code_t scatter_func(const void* input,
                                      wholememory_matrix_description_t input_desc,
                                      void* indices,
                                      wholememory_array_description_t indices_desc,
                                      wholememory_gref_t embedding_gref,
                                      wholememory_matrix_description_t embedding_desc,
                                      cudaStream_t stream,
                                      int scatter_sms)
{
  try {
    bool embedding_is_float = wholememory_dtype_is_floating_number(embedding_desc.dtype);
    WHOLEMEMORY_CHECK(embedding_is_float ||
                      wholememory_dtype_is_integer_number(embedding_desc.dtype));
    bool input_is_float = wholememory_dtype_is_floating_number(input_desc.dtype);
    WHOLEMEMORY_CHECK(input_is_float || wholememory_dtype_is_integer_number(input_desc.dtype));
    WHOLEMEMORY_EXPECTS(
      embedding_is_float == input_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indices_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
    wholememory_error_code_t (*p_scatter_func)(const void*,
                                               wholememory_matrix_description_t,
                                               void*,
                                               wholememory_array_description_t,
                                               wholememory_gref_t,
                                               wholememory_matrix_description_t,
                                               cudaStream_t,
                                               int) = nullptr;
    if (embedding_is_float) {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_scatter_func = scatter_floating_int32_func;
      } else {
        p_scatter_func = scatter_floating_int64_func;
      }
    } else {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_scatter_func = scatter_integer_int32_func;
      } else {
        p_scatter_func = scatter_integer_int64_func;
      }
    }
    return p_scatter_func(input,
                          input_desc,
                          indices,
                          indices_desc,
                          embedding_gref,
                          embedding_desc,
                          stream,
                          scatter_sms);
  } catch (const wholememory::cuda_error& wle) {
    WHOLEMEMORY_ERROR("scatter CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_ERROR("scatter LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
