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
#include <wholememory/wholegraph_op.h>

#include <wholegraph_ops/unweighted_sample_without_replacement_impl.h>

#include "error.hpp"
#include "logger.hpp"

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
  void* stream)
{
  bool const csr_row_ptr_has_handle = wholememory_tensor_has_handle(wm_csr_row_ptr_tensor);
  wholememory_memory_type_t csr_row_ptr_memory_type = WHOLEMEMORY_MT_NONE;
  if (csr_row_ptr_has_handle) {
    csr_row_ptr_memory_type =
      wholememory_get_memory_type(wholememory_tensor_get_memory_handle(wm_csr_row_ptr_tensor));
  }
  WHOLEMEMORY_EXPECTS_NOTHROW(!csr_row_ptr_has_handle ||
                                csr_row_ptr_memory_type == WHOLEMEMORY_MT_CHUNKED ||
                                csr_row_ptr_memory_type == WHOLEMEMORY_MT_CONTINUOUS ||
                                csr_row_ptr_memory_type == WHOLEMEMORY_MT_DISTRIBUTED,
                              "Memory type not supported.");
  bool const csr_col_ptr_has_handle = wholememory_tensor_has_handle(wm_csr_col_ptr_tensor);
  wholememory_memory_type_t csr_col_ptr_memory_type = WHOLEMEMORY_MT_NONE;
  if (csr_col_ptr_has_handle) {
    csr_col_ptr_memory_type =
      wholememory_get_memory_type(wholememory_tensor_get_memory_handle(wm_csr_col_ptr_tensor));
  }
  WHOLEMEMORY_EXPECTS_NOTHROW(!csr_col_ptr_has_handle ||
                                csr_col_ptr_memory_type == WHOLEMEMORY_MT_CHUNKED ||
                                csr_col_ptr_memory_type == WHOLEMEMORY_MT_CONTINUOUS ||
                                csr_row_ptr_memory_type == WHOLEMEMORY_MT_DISTRIBUTED,
                              "Memory type not supported.");

  auto csr_row_ptr_tensor_description =
    *wholememory_tensor_get_tensor_description(wm_csr_row_ptr_tensor);
  auto csr_col_ptr_tensor_description =
    *wholememory_tensor_get_tensor_description(wm_csr_col_ptr_tensor);

  if (csr_row_ptr_tensor_description.dim != 1) {
    WHOLEMEMORY_ERROR("wm_csr_row_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_description.dim != 1) {
    WHOLEMEMORY_ERROR("wm_csr_col_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  wholememory_array_description_t wm_csr_row_ptr_desc, wm_csr_col_ptr_desc;
  if (!wholememory_convert_tensor_desc_to_array(&wm_csr_row_ptr_desc,
                                                &csr_row_ptr_tensor_description)) {
    WHOLEMEMORY_ERROR("Input wm_csr_row_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_array(&wm_csr_col_ptr_desc,
                                                &csr_col_ptr_tensor_description)) {
    WHOLEMEMORY_ERROR("Input wm_csr_col_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  wholememory_tensor_description_t center_nodes_tensor_desc =
    *wholememory_tensor_get_tensor_description(center_nodes_tensor);
  if (center_nodes_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input center_nodes_tensor should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_array_description_t center_nodes_desc;
  if (!wholememory_convert_tensor_desc_to_array(&center_nodes_desc, &center_nodes_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input center_nodes_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  wholememory_tensor_description_t output_sample_offset_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_sample_offset_tensor);
  if (output_sample_offset_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Output output_sample_offset_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_array_description_t output_sample_offset_desc;
  if (!wholememory_convert_tensor_desc_to_array(&output_sample_offset_desc,
                                                &output_sample_offset_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output output_sample_offset_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  void* center_nodes         = wholememory_tensor_get_data_pointer(center_nodes_tensor);
  void* output_sample_offset = wholememory_tensor_get_data_pointer(output_sample_offset_tensor);

  if (csr_col_ptr_memory_type == WHOLEMEMORY_MT_DISTRIBUTED &&
      csr_row_ptr_memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
    wholememory_distributed_backend_t distributed_backend_row = wholememory_get_distributed_backend(
      wholememory_tensor_get_memory_handle(wm_csr_row_ptr_tensor));
    wholememory_distributed_backend_t distributed_backend_col = wholememory_get_distributed_backend(
      wholememory_tensor_get_memory_handle(wm_csr_col_ptr_tensor));
    if (distributed_backend_col == WHOLEMEMORY_DB_NCCL &&
        distributed_backend_row == WHOLEMEMORY_DB_NCCL) {
      wholememory_handle_t wm_csr_row_ptr_handle =
        wholememory_tensor_get_memory_handle(wm_csr_row_ptr_tensor);
      wholememory_handle_t wm_csr_col_ptr_handle =
        wholememory_tensor_get_memory_handle(wm_csr_col_ptr_tensor);
      return wholegraph_ops::wholegraph_csr_unweighted_sample_without_replacement_nccl(
        wm_csr_row_ptr_handle,
        wm_csr_col_ptr_handle,
        csr_row_ptr_tensor_description,
        csr_col_ptr_tensor_description,
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
        static_cast<cudaStream_t>(stream));
    } else {
      WHOLEMEMORY_ERROR("Only NCCL communication backend is supported for sampling.");
      return WHOLEMEMORY_INVALID_INPUT;
    }
  }

  wholememory_gref_t wm_csr_row_ptr_gref, wm_csr_col_ptr_gref;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_tensor_get_global_reference(wm_csr_row_ptr_tensor, &wm_csr_row_ptr_gref));
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_tensor_get_global_reference(wm_csr_col_ptr_tensor, &wm_csr_col_ptr_gref));

  return wholegraph_ops::wholegraph_csr_unweighted_sample_without_replacement_mapped(
    wm_csr_row_ptr_gref,
    wm_csr_row_ptr_desc,
    wm_csr_col_ptr_gref,
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
    static_cast<cudaStream_t>(stream));
}
