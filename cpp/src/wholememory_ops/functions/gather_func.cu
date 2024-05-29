/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

namespace wholememory_ops {

wholememory_error_code_t gather_integer_int32_func(wholememory_gref_t embedding_gref,
                                                   wholememory_matrix_description_t embedding_desc,
                                                   void* indices,
                                                   wholememory_array_description_t indices_desc,
                                                   bool gather_with_sorted_ids,
                                                   void* raw_indices,
                                                   void* output,
                                                   wholememory_matrix_description_t output_desc,
                                                   cudaStream_t stream,
                                                   int gather_sms);
wholememory_error_code_t gather_integer_int64_func(wholememory_gref_t embedding_gref,
                                                   wholememory_matrix_description_t embedding_desc,
                                                   void* indices,
                                                   wholememory_array_description_t indices_desc,
                                                   bool gather_with_sorted_ids,
                                                   void* raw_indices,
                                                   void* output,
                                                   wholememory_matrix_description_t output_desc,
                                                   cudaStream_t stream,
                                                   int gather_sms);
wholememory_error_code_t gather_floating_int32_func(wholememory_gref_t embedding_gref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    void* indices,
                                                    wholememory_array_description_t indices_desc,
                                                    bool gather_with_sorted_ids,
                                                    void* raw_indices,
                                                    void* output,
                                                    wholememory_matrix_description_t output_desc,
                                                    cudaStream_t stream,
                                                    int gather_sms);
wholememory_error_code_t gather_floating_int64_func(wholememory_gref_t embedding_gref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    void* indices,
                                                    wholememory_array_description_t indices_desc,
                                                    bool gather_with_sorted_ids,
                                                    void* raw_indices,
                                                    void* output,
                                                    wholememory_matrix_description_t output_desc,
                                                    cudaStream_t stream,
                                                    int gather_sms);

wholememory_error_code_t gather_func(wholememory_gref_t embedding_gref,
                                     wholememory_matrix_description_t embedding_desc,
                                     void* indices,
                                     wholememory_array_description_t indices_desc,
                                     void* output,
                                     wholememory_matrix_description_t output_desc,
                                     cudaStream_t stream,
                                     int gather_sms)
{
  try {
    bool embedding_is_float = wholememory_dtype_is_floating_number(embedding_desc.dtype);
    WHOLEMEMORY_CHECK(embedding_is_float ||
                      wholememory_dtype_is_integer_number(embedding_desc.dtype));
    bool output_is_float = wholememory_dtype_is_floating_number(output_desc.dtype);
    WHOLEMEMORY_CHECK(output_is_float || wholememory_dtype_is_integer_number(output_desc.dtype));
    WHOLEMEMORY_EXPECTS(
      embedding_is_float == output_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indices_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
    wholememory_error_code_t (*p_gather_func)(wholememory_gref_t,
                                              wholememory_matrix_description_t,
                                              void* indices,
                                              wholememory_array_description_t,
                                              bool,
                                              void*,
                                              void*,
                                              wholememory_matrix_description_t,
                                              cudaStream_t,
                                              int) = nullptr;
    if (embedding_is_float) {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_gather_func = gather_floating_int32_func;
      } else {
        p_gather_func = gather_floating_int64_func;
      }
    } else {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_gather_func = gather_integer_int32_func;
      } else {
        p_gather_func = gather_integer_int64_func;
      }
    }
    return p_gather_func(embedding_gref,
                         embedding_desc,
                         indices,
                         indices_desc,
                         false,
                         nullptr,
                         output,
                         output_desc,
                         stream,
                         gather_sms);
  } catch (const wholememory::cuda_error& rle) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t gather_with_sorted_ids_func(
  wholememory_gref_t embedding_gref,
  wholememory_matrix_description_t embedding_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  void* raw_indices,
  wholememory_array_description_t raw_indices_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  cudaStream_t stream,
  int gather_sms)
{
  try {
    bool embedding_is_float = wholememory_dtype_is_floating_number(embedding_desc.dtype);
    WHOLEMEMORY_CHECK(embedding_is_float ||
                      wholememory_dtype_is_integer_number(embedding_desc.dtype));
    bool output_is_float = wholememory_dtype_is_floating_number(output_desc.dtype);
    WHOLEMEMORY_CHECK(output_is_float || wholememory_dtype_is_integer_number(output_desc.dtype));
    WHOLEMEMORY_EXPECTS(
      embedding_is_float == output_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indices_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
    WHOLEMEMORY_CHECK(indices_desc.size == raw_indices_desc.size);
    WHOLEMEMORY_CHECK(indices_desc.dtype == raw_indices_desc.dtype);
    wholememory_error_code_t (*p_gather_func)(wholememory_gref_t,
                                              wholememory_matrix_description_t,
                                              void* indices,
                                              wholememory_array_description_t,
                                              bool,
                                              void*,
                                              void*,
                                              wholememory_matrix_description_t,
                                              cudaStream_t,
                                              int) = nullptr;
    if (embedding_is_float) {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_gather_func = gather_floating_int32_func;
      } else {
        p_gather_func = gather_floating_int64_func;
      }
    } else {
      if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
        p_gather_func = gather_integer_int32_func;
      } else {
        p_gather_func = gather_integer_int64_func;
      }
    }
    return p_gather_func(embedding_gref,
                         embedding_desc,
                         indices,
                         indices_desc,
                         true,
                         raw_indices,
                         output,
                         output_desc,
                         stream,
                         gather_sms);
  } catch (const wholememory::cuda_error& rle) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
