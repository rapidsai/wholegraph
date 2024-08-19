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

#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include <wholememory_ops/temp_memory_handle.hpp>
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

wholememory_error_code_t sort_unique_indices_func(const void* indices,
                                                  wholememory_array_description_t indice_desc,
                                                  void* sort_raw_indices,
                                                  int* num_runs,
                                                  temp_memory_handle* unique_indices_handle,
                                                  temp_memory_handle* unique_count_handle,
                                                  wm_thrust_allocator* p_thrust_allocator,
                                                  wholememory_env_func_t* p_env_fns,
                                                  cudaStream_t stream);

}  // namespace wholememory_ops
