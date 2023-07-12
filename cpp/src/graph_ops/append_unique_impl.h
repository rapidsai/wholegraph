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
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace graph_ops {
wholememory_error_code_t graph_append_unique_impl(
  void* target_nodes_ptr,
  wholememory_array_description_t target_nodes_desc,
  void* neighbor_nodes_ptr,
  wholememory_array_description_t neighbor_nodes_desc,
  void* output_unique_node_memory_context,
  int* output_neighbor_raw_to_unique_mapping_ptr,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);
}
