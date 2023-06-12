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

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Unweighted sample without replacement kernel op
 * @param wm_csr_row_ptr_tensor : Wholememory Tensor of graph csr_row_ptr
 * @param wm_csr_col_ptr_tensor : Wholememory Tensor of graph csr_col_ptr
 * @param center_nodes_tensor : None Wholememory Tensor of center node to sample
 * @param max_sample_count : maximum sample count
 * @param output_sample_offset_tensor : pointer to output sample offset
 * @param output_dest_memory_context : memory context to output dest nodes
 * @param output_center_localid_memory_context : memory context to output center local id
 * @param output_edge_gid_memory_context : memory context to output edge global id
 * @param random_seed: random number generator seed
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */

wholememory_error_code_t wholegraph_csr_unweighted_sample_without_replacement(
  wholememory_tensor_t wm_csr_row_ptr_tensor,
  wholememory_tensor_t wm_csr_col_ptr_tensor,
  wholememory_tensor_t center_nodes_tensor,
  int max_sample_count,
  wholememory_tensor_t output_sample_offset_tensor,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  void* stream);

/**
 * Unweighted sample without replacement kernel op
 * @param wm_csr_row_ptr_tensor : Wholememory Tensor of graph csr_row_ptr
 * @param wm_csr_col_ptr_tensor : Wholememory Tensor of graph csr_col_ptr
 * @param wm_csr_weight_ptr_tensor : Wholememory Tensor of graph edge weight
 * @param center_nodes_tensor : None Wholememory Tensor of center node to sample
 * @param max_sample_count : maximum sample count
 * @param output_sample_offset_tensor : pointer to output sample offset
 * @param output_dest_memory_context : memory context to output dest nodes
 * @param output_center_localid_memory_context : memory context to output center local id
 * @param output_edge_gid_memory_context : memory context to output edge global id
 * @param random_seed: random number generator seed
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */

wholememory_error_code_t wholegraph_csr_weighted_sample_without_replacement(
  wholememory_tensor_t wm_csr_row_ptr_tensor,
  wholememory_tensor_t wm_csr_col_ptr_tensor,
  wholememory_tensor_t wm_csr_weight_ptr_tensor,
  wholememory_tensor_t center_nodes_tensor,
  int max_sample_count,
  wholememory_tensor_t output_sample_offset_tensor,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  void* stream);

/**
 * raft_pcg_generator_random_int cpu op
 * @param random_seed : random seed
 * @param subsequence : subsequence for generating random value
 * @param output : Wholememory Tensor of output
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t generate_random_positive_int_cpu(int64_t random_seed,
                                                          int64_t subsequence,
                                                          wholememory_tensor_t output);

/**
 * raft_pcg_generator_random_float cpu op
 * @param random_seed : random seed
 * @param subsequence : subsequence for generating random value
 * @param output : Wholememory Tensor of output
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t generate_exponential_distribution_negative_float_cpu(
  int64_t random_seed, int64_t subsequence, wholememory_tensor_t output);

#ifdef __cplusplus
}
#endif
