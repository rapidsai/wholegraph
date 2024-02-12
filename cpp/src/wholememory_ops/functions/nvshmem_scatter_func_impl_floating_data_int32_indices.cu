/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

template <typename InputT, typename EmbeddingT>
void nvshmem_scatter_floating_int32_temp_func(wholememory_comm_t wm_comm,
                                              void* input,
                                              void* temp_input,
                                              wholememory_matrix_description_t input_desc,
                                              const void* indices,
                                              int64_t indice_count,
                                              wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                                              wholememory_matrix_description_t embedding_desc,
                                              size_t embedding_entry_count_per_rank,
                                              wholememory_env_func_t* p_env_fns,
                                              cudaStream_t stream,
                                              int scatter_sms)
{
  nvshmem_scatter_temp_put_mem_sort_idx_func<InputT, int32_t, EmbeddingT>(
    wm_comm,
    input,
    temp_input,
    input_desc,
    indices,
    indice_count,
    embeding_nvshmem_ptr,
    embedding_desc,
    embedding_entry_count_per_rank,
    p_env_fns,
    stream,
    scatter_sms);
}

REGISTER_DISPATCH_TWO_TYPES(NvshmemScatterFuncFloatingInt32,
                            nvshmem_scatter_floating_int32_temp_func,
                            HALF_FLOAT_DOUBLE,
                            HALF_FLOAT_DOUBLE);

wholememory_error_code_t nvshmem_scatter_floating_int32_func(
  wholememory_comm_t wm_comm,
  void* input,
  void* temp_input,
  wholememory_matrix_description_t input_desc,
  const void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
  wholememory_matrix_description_t embedding_desc,
  size_t embedding_entry_count_per_rank,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms)
{
  try {
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(embedding_desc.dtype));
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(input_desc.dtype));
    WHOLEMEMORY_CHECK(indices_desc.dtype == WHOLEMEMORY_DT_INT);

    DISPATCH_TWO_TYPES(input_desc.dtype,
                       embedding_desc.dtype,
                       NvshmemScatterFuncFloatingInt32,
                       wm_comm,
                       input,
                       temp_input,
                       input_desc,
                       indices,
                       indices_desc.size,
                       embeding_nvshmem_ptr,
                       embedding_desc,
                       embedding_entry_count_per_rank,
                       p_env_fns,
                       stream,
                       scatter_sms);
  } catch (const wholememory::cuda_error& wle) {
    WHOLEMEMORY_ERROR("scatter CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_ERROR("scatter CUDA LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

};  // namespace wholememory_ops

#endif
