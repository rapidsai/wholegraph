/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#ifdef WITH_NVSHMEM_SUPPORT

#include "nvshmem_gather_scatter_func.cuh"
#include <wholememory/wholememory.h>

#include "logger.hpp"
#include "wholememory_ops/register.hpp"
namespace wholememory_ops {

template <typename EmbeddingT, typename OutputT>
void nvshmem_gather_integer_int64_temp_func(wholememory_comm_t wm_comm,
                                            wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                                            wholememory_matrix_description_t embedding_desc,
                                            const void* indices,
                                            int64_t indice_count,
                                            void* output,
                                            void* temp_output,
                                            wholememory_matrix_description_t output_desc,
                                            size_t* embedding_entry_offsets,
                                            wholememory_env_func_t* p_env_fns,
                                            cudaStream_t stream,
                                            int gather_sms)
{
  nvshmem_gather_temp_get_mem_sort_idx_func<EmbeddingT, int64_t, OutputT>(wm_comm,
                                                                          embeding_nvshmem_ptr,
                                                                          embedding_desc,
                                                                          indices,
                                                                          indice_count,
                                                                          output,
                                                                          temp_output,
                                                                          output_desc,
                                                                          embedding_entry_offsets,
                                                                          p_env_fns,
                                                                          stream,
                                                                          gather_sms);
}

REGISTER_DISPATCH_TWO_TYPES(NvshmemGatherFuncIntegerInt64,
                            nvshmem_gather_integer_int64_temp_func,
                            ALLSINT,
                            ALLSINT)

wholememory_error_code_t nvshmem_gather_integer_int64_func(
  wholememory_comm_t wm_comm,
  wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
  wholememory_matrix_description_t embedding_desc,
  const void* indices,
  wholememory_array_description_t indices_desc,
  void* output,
  void* temp_output,
  wholememory_matrix_description_t output_desc,
  size_t* embedding_entry_offsets,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms)
{
  try {
    WHOLEMEMORY_CHECK(wholememory_dtype_is_integer_number(embedding_desc.dtype));
    WHOLEMEMORY_CHECK(wholememory_dtype_is_integer_number(output_desc.dtype));
    WHOLEMEMORY_CHECK(indices_desc.dtype == WHOLEMEMORY_DT_INT64);

    DISPATCH_TWO_TYPES(embedding_desc.dtype,
                       output_desc.dtype,
                       NvshmemGatherFuncIntegerInt64,
                       wm_comm,
                       embeding_nvshmem_ptr,
                       embedding_desc,
                       indices,
                       indices_desc.size,
                       output,
                       temp_output,
                       output_desc,
                       embedding_entry_offsets,
                       p_env_fns,
                       stream,
                       gather_sms);
  } catch (const wholememory::cuda_error& wle) {
    WHOLEMEMORY_ERROR("gather CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_ERROR("gather CUDA LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}
template <>
__global__ void scatter_func_with_nvshmem_sort_idxs_kernel<float, int>(
  float* __restrict__ temp_input,
  wholememory_matrix_description_t temp_input_desc,
  wholememory_nvshmem_ref_t embeding_nvshmem_ref,
  wholememory_matrix_description_t embedding_desc,
  const int* __restrict__ sorted_index,  //
  const int* __restrict__ input_index,   //
  int64_t indice_count,
  const int max_blocks_for_local,
  const int intra_node_ranks,
  const int node_rank,
  size_t* embedding_entry_offsets,
  const int threads_per_group);
};  // namespace wholememory_ops

#endif
