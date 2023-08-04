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
#include <cuda_runtime_api.h>

#include "append_unique_func.cuh"
#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "wholememory_ops/register.hpp"

namespace graph_ops {

REGISTER_DISPATCH_ONE_TYPE(GraphAppendUnique, graph_append_unique_func, SINT3264)
wholememory_error_code_t graph_append_unique_impl(
  void* target_nodes_ptr,
  wholememory_array_description_t target_nodes_desc,
  void* neighbor_nodes_ptr,
  wholememory_array_description_t neighbor_nodes_desc,
  void* output_unique_node_memory_context,
  int* output_neighbor_raw_to_unique_mapping_ptr,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(target_nodes_desc.dtype,
                      GraphAppendUnique,
                      target_nodes_ptr,
                      target_nodes_desc,
                      neighbor_nodes_ptr,
                      neighbor_nodes_desc,
                      output_unique_node_memory_context,
                      output_neighbor_raw_to_unique_mapping_ptr,
                      p_env_fns,
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
