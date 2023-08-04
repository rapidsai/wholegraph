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

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Append Unique op
 * @param target_nodes_tensor : Wholememory Tensor of graph csr_row_ptr
 * @param neighbor_nodes_tensor : Wholememory Tensor of graph csr_col_ptr
 * @param output_unique_node_memory_context : memory context to output dest nodes
 * @param output_neighbor_raw_to_unique_mapping_tensor : pointer to output sample offset, optional
 * output
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t graph_append_unique(
  wholememory_tensor_t target_nodes_tensor,
  wholememory_tensor_t neighbor_nodes_tensor,
  void* output_unique_node_memory_context,
  wholememory_tensor_t output_neighbor_raw_to_unique_mapping_tensor,
  wholememory_env_func_t* p_env_fns,
  void* stream);

/**
 * Csr Add Self Loop Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ptr_tensor : Wholememory Tensor of csr_col_ptr
 * @param output_csr_row_ptr_tensor : Wholememory Tensor of output_csr_row_ptr
 * @param output_csr_col_ptr_tensor : Wholememory Tensor of output_csr_col_ptr
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t csr_add_self_loop(wholememory_tensor_t csr_row_ptr_tensor,
                                           wholememory_tensor_t csr_col_ptr_tensor,
                                           wholememory_tensor_t output_csr_row_ptr_tensor,
                                           wholememory_tensor_t output_csr_col_ptr_tensor,
                                           void* stream);

#ifdef __cplusplus
}
#endif
