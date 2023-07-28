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
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholegraph_ops {

wholememory_error_code_t wholegraph_csr_weighted_sample_without_replacement_mapped(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  wholememory_gref_t wm_csr_weight_ptr,
  wholememory_array_description_t wm_csr_weight_ptr_desc,
  void* center_nodes,
  wholememory_array_description_t center_nodes_desc,
  int max_sample_count,
  void* output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

}  // namespace wholegraph_ops
