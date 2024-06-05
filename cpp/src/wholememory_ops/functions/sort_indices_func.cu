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
#include "sort_indices_func.h"

#include <cub/device/device_radix_sort.cuh>
#include <thrust/sequence.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "wholememory_ops/register.hpp"

namespace wholememory_ops {

template <typename IndexT>
struct UnsignedType {};

template <>
struct UnsignedType<int> {
  using UType = unsigned int;
};

template <>
struct UnsignedType<int64_t> {
  using UType = uint64_t;
};

template <typename IndexT>
void sort_indices_temp_func(const void* indices_before_sort,
                            wholememory_array_description_t indices_desc,
                            void* indices_after_sort,
                            void* raw_indices,
                            wm_thrust_allocator* p_thrust_allocator,
                            wholememory_env_func_t* p_env_fns,
                            cudaStream_t stream)
{
  auto index_type = indices_desc.dtype;
  WHOLEMEMORY_CHECK(indices_desc.storage_offset == 0);
  WHOLEMEMORY_CHECK(index_type == WHOLEMEMORY_DT_INT || index_type == WHOLEMEMORY_DT_INT64);
  wm_thrust_allocator& allocator = *p_thrust_allocator;

  IndexT* seq_indices = reinterpret_cast<IndexT*>(allocator.allocate(
    wholememory_get_memory_element_count_from_array(&indices_desc) * sizeof(IndexT)));
  thrust::sequence(thrust::cuda::par_nosync(allocator).on(stream),
                   seq_indices,
                   seq_indices + indices_desc.size,
                   0);
  // use UTypeT to put minus indices at last.
  using UTypeT                  = typename UnsignedType<IndexT>::UType;
  const UTypeT* indices_to_sort = static_cast<const UTypeT*>(indices_before_sort);
  UTypeT* sorted_indice         = static_cast<UTypeT*>(indices_after_sort);
  void* cub_temp_storage        = nullptr;
  size_t temp_storage_bytes     = 0;
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indices_to_sort,
                                  sorted_indice,
                                  seq_indices,
                                  static_cast<IndexT*>(raw_indices),
                                  indices_desc.size,
                                  0,
                                  sizeof(UTypeT) * 8,
                                  stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indices_to_sort,
                                  sorted_indice,
                                  seq_indices,
                                  static_cast<IndexT*>(raw_indices),
                                  indices_desc.size,
                                  0,
                                  sizeof(UTypeT) * 8,
                                  stream);
  allocator.deallocate(reinterpret_cast<char*>(seq_indices),
                       wholememory_get_memory_size_from_array(&indices_desc));
  allocator.deallocate(static_cast<char*>(cub_temp_storage), temp_storage_bytes);
}

REGISTER_DISPATCH_ONE_TYPE(SortIndices, sort_indices_temp_func, SINT3264)

wholememory_error_code_t sort_indices_func(const void* indices_before_sort,
                                           wholememory_array_description_t indice_desc,
                                           void* indices_after_sort,
                                           void* raw_indices,
                                           wm_thrust_allocator* p_thrust_allocator,
                                           wholememory_env_func_t* p_env_fns,
                                           cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      SortIndices,
                      indices_before_sort,
                      indice_desc,
                      indices_after_sort,
                      raw_indices,
                      p_thrust_allocator,
                      p_env_fns,
                      stream);
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("sort_indices_func CUDA LOGIC Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("sort_indices_func LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
