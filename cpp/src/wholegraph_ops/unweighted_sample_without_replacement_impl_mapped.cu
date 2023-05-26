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

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "unweighted_sample_without_replacement_func.cuh"
#include "wholememory_ops/register.hpp"

namespace wholegraph_ops {

REGISTER_DISPATCH_TWO_TYPES(UnweightedSampleWithoutReplacementCSR,
                            wholegraph_csr_unweighted_sample_without_replacement_func,
                            SINT3264,
                            SINT3264)

wholememory_error_code_t wholegraph_csr_unweighted_sample_without_replacement_mapped(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
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
  cudaStream_t stream)
{
  try {
    DISPATCH_TWO_TYPES(center_nodes_desc.dtype,
                       wm_csr_col_ptr_desc.dtype,
                       UnweightedSampleWithoutReplacementCSR,
                       wm_csr_row_ptr,
                       wm_csr_row_ptr_desc,
                       wm_csr_col_ptr,
                       wm_csr_col_ptr_desc,
                       center_nodes,
                       center_nodes_desc,
                       max_sample_count,
                       output_sample_offset,
                       output_sample_offset_desc,
                       output_dest_memory_context,
                       output_center_localid_memory_context,
                       output_edge_gid_memory_context,
                       random_seed,
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

}  // namespace wholegraph_ops
