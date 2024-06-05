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
#pragma once

#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {

wholememory_error_code_t gather_func(wholememory_gref_t embedding_gref,
                                     wholememory_matrix_description_t embedding_desc,
                                     void* indices,
                                     wholememory_array_description_t indices_desc,
                                     void* output,
                                     wholememory_matrix_description_t output_desc,
                                     cudaStream_t stream,
                                     int gather_sms = -1);

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
  int gather_sms);

wholememory_error_code_t scatter_func(const void* input,
                                      wholememory_matrix_description_t input_desc,
                                      void* indices,
                                      wholememory_array_description_t indices_desc,
                                      wholememory_gref_t embedding_gref,
                                      wholememory_matrix_description_t embedding_desc,
                                      cudaStream_t stream,
                                      int scatter_sms = -1);

}  // namespace wholememory_ops
