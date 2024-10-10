/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "sort_indices_func.h"
#include "sort_unique_indices_func.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/sequence.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "wholememory_ops/register.hpp"

namespace wholememory_ops {

template <typename IndexT>
void SortUniqueIndicesTempFunc(const void* indices,
                               wholememory_array_description_t indice_desc,
                               void* sort_raw_indices,
                               int* num_runs,
                               temp_memory_handle* unique_indices_handle,
                               temp_memory_handle* unique_count_handle,
                               wm_thrust_allocator* p_thrust_allocator,
                               wholememory_env_func_t* p_env_fns,
                               cudaStream_t stream)
{
  if (indice_desc.size == 0) return;
  wm_thrust_allocator& allocator = *p_thrust_allocator;
  WHOLEMEMORY_CHECK_NOTHROW(indice_desc.storage_offset == 0);
  temp_memory_handle sorted_indices_handle(p_env_fns);
  sorted_indices_handle.device_malloc(indice_desc.size, indice_desc.dtype);
  IndexT* sorted_indices = static_cast<IndexT*>(sorted_indices_handle.pointer());

  sort_indices_func(
    indices, indice_desc, sorted_indices, sort_raw_indices, p_thrust_allocator, p_env_fns, stream);

  unique_indices_handle->device_malloc(indice_desc.size, indice_desc.dtype);
  unique_count_handle->device_malloc(indice_desc.size, WHOLEMEMORY_DT_INT);
  IndexT* unique_indices = static_cast<IndexT*>(unique_indices_handle->pointer());
  int* unique_counts     = static_cast<int*>(unique_count_handle->pointer());
  temp_memory_handle number_runs_handle(p_env_fns);
  number_runs_handle.device_malloc(1, WHOLEMEMORY_DT_INT);
  int* number_runs          = static_cast<int*>(number_runs_handle.pointer());
  void* cub_temp_storage    = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(cub_temp_storage,
                                     temp_storage_bytes,
                                     sorted_indices,
                                     unique_indices,
                                     unique_counts,
                                     number_runs,
                                     indice_desc.size,
                                     stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceRunLengthEncode::Encode(cub_temp_storage,
                                     temp_storage_bytes,
                                     sorted_indices,
                                     unique_indices,
                                     unique_counts,
                                     number_runs,
                                     indice_desc.size,
                                     stream);
  WM_CUDA_CHECK_NO_THROW(
    cudaMemcpyAsync(num_runs, number_runs, sizeof(int), cudaMemcpyDeviceToHost, stream));
}

REGISTER_DISPATCH_ONE_TYPE(SortUniqueIndicesTempFunc, SortUniqueIndicesTempFunc, SINT3264)

wholememory_error_code_t sort_unique_indices_func(const void* indices,
                                                  wholememory_array_description_t indice_desc,
                                                  void* sort_raw_indices,
                                                  int* num_runs,
                                                  temp_memory_handle* unique_indices_handle,
                                                  temp_memory_handle* unique_count_handle,
                                                  wm_thrust_allocator* p_thrust_allocator,
                                                  wholememory_env_func_t* p_env_fns,
                                                  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      SortUniqueIndicesTempFunc,
                      indices,
                      indice_desc,
                      sort_raw_indices,
                      num_runs,
                      unique_indices_handle,
                      unique_count_handle,
                      p_thrust_allocator,
                      p_env_fns,
                      stream);
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("sort_unique_indices_func CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("sort_unique_indices_func LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
