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
#include <wholememory/wholememory.h>

namespace wholememory_ops {

wholememory_error_code_t wholememory_gather_mapped(
  wholememory_gref_t wholememory_gref,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  bool gather_with_sorted_ids,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms);

wholememory_error_code_t wholememory_gather_nccl(wholememory_handle_t wholememory_handle,
                                                 wholememory_matrix_description_t wholememory_desc,
                                                 void* indices,
                                                 wholememory_array_description_t indice_desc,
                                                 void* output,
                                                 wholememory_matrix_description_t output_desc,
                                                 wholememory_env_func_t* p_env_fns,
                                                 cudaStream_t stream,
                                                 int gather_sms);

wholememory_error_code_t wholememory_gather_hierarchy(
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms);

wholememory_error_code_t wholememory_gather_distributed(
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms);

#ifdef WITH_NVSHMEM_SUPPORT

wholememory_error_code_t wholememory_gather_nvshmem(
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms);
#endif
}  // namespace wholememory_ops
