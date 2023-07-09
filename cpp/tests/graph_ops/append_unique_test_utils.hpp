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
#include <wholememory/tensor_description.h>

namespace graph_ops {
namespace testing {
void gen_node_ids(void* host_target_nodes_ptr,
                  wholememory_array_description_t node_desc,
                  int64_t range,
                  bool unique);
void host_append_unique(void* target_nodes_ptr,
                        wholememory_array_description_t target_nodes_desc,
                        void* neighbor_nodes_ptr,
                        wholememory_array_description_t neighbor_nodes_desc,
                        int* host_total_unique_count,
                        void** host_output_unique_nodes_ptr);

void host_gen_append_unique_neighbor_raw_to_unique(
  void* host_output_unique_nodes_ptr,
  wholememory_array_description_t output_unique_nodes_desc,
  void* host_neighbor_nodes_ptr,
  wholememory_array_description_t neighbor_nodes_desc,
  void** ref_host_output_neighbor_raw_to_unique_mapping_ptr,
  wholememory_array_description_t output_neighbor_raw_to_unique_mapping_desc);

}  // namespace testing
}  // namespace graph_ops
