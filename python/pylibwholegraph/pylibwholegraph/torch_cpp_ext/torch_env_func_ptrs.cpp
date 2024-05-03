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
#include "torch_env_func_ptrs.h"

#include <c10/cuda/CUDAStream.h>

#include "torch_utils.h"

namespace wholegraph_torch {

void* torch_malloc_func(wholememory_tensor_description_t* tensor_description,
                        wholememory_memory_allocation_type_t memory_allocation_type,
                        void* memory_context,
                        void* /*global_context*/)
{
  bool gpu_memory    = memory_allocation_type == WHOLEMEMORY_MA_DEVICE;
  bool pinned_memory = memory_allocation_type == WHOLEMEMORY_MA_PINNED;
  return torch_common_malloc_func(tensor_description, memory_context, gpu_memory, pinned_memory);
}

static wholememory_env_func_t pytorch_env_func = {
  .temporary_fns =
    {
      .create_memory_context_fn  = create_torch_memory_context_func,
      .destroy_memory_context_fn = destroy_torch_memory_context_func,
      .malloc_fn                 = torch_malloc_func,
      .free_fn                   = torch_common_free_func,
      .global_context            = nullptr,
    },
  .output_fns = {
    .malloc_fn      = torch_malloc_func,
    .free_fn        = torch_common_free_func,
    .global_context = nullptr,
  }};

wholememory_env_func_t* get_pytorch_env_func() { return &pytorch_env_func; }

cudaStream_t get_current_stream() { return at::cuda::getCurrentCUDAStream(); }

void* create_output_context() {
  void* output_context = nullptr;
  create_torch_memory_context_func(&output_context, nullptr);
  return output_context;
}

void destroy_output_context(void* output_context) {
  destroy_torch_memory_context_func(output_context, nullptr);
}

void free_context_data(void* output_context) {
  torch_common_free_func(output_context, nullptr);
}

}  // namespace wholegraph_torch
