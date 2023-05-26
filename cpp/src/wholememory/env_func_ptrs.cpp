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
#include <wholememory/env_func_ptrs.hpp>

#include <mutex>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "initialize.hpp"

namespace wholememory {

void default_create_memory_context_func(void** memory_context, void* /*global_context*/)
{
  auto* default_memory_context = new default_memory_context_t;
  wholememory_initialize_tensor_desc(&default_memory_context->desc);
  default_memory_context->ptr             = nullptr;
  default_memory_context->allocation_type = WHOLEMEMORY_MA_NONE;
  *memory_context                         = default_memory_context;
}

void default_destroy_memory_context_func(void* memory_context, void* /*global_context*/)
{
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  delete default_memory_context;
}

void* default_malloc_func(wholememory_tensor_description_t* tensor_description,
                          wholememory_memory_allocation_type_t memory_allocation_type,
                          void* memory_context,
                          void* /*global_context*/)
{
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  void* ptr                    = nullptr;
  try {
    if (memory_allocation_type == WHOLEMEMORY_MA_HOST) {
      ptr = malloc(wholememory_get_memory_size_from_tensor(tensor_description));
      if (ptr == nullptr) { WHOLEMEMORY_FAIL_NOTHROW("malloc returned nullptr.\n"); }
    } else if (memory_allocation_type == WHOLEMEMORY_MA_PINNED) {
      WM_CUDA_CHECK(
        cudaMallocHost(&ptr, wholememory_get_memory_size_from_tensor(tensor_description)));
    } else if (memory_allocation_type == WHOLEMEMORY_MA_DEVICE) {
      WM_CUDA_CHECK(cudaMalloc(&ptr, wholememory_get_memory_size_from_tensor(tensor_description)));
    } else {
      WHOLEMEMORY_FAIL_NOTHROW("memory_allocation_type incorrect.\n");
    }
  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("cudaMalloc failed, %s.\n", wce.what());
  }
  default_memory_context->desc            = *tensor_description;
  default_memory_context->ptr             = ptr;
  default_memory_context->allocation_type = memory_allocation_type;
  return ptr;
}

void default_free_func(void* memory_context, void* /*global_context*/)
{
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  auto memory_allocation_type  = default_memory_context->allocation_type;
  if (memory_allocation_type == WHOLEMEMORY_MA_HOST) {
    free(default_memory_context->ptr);
  } else if (memory_allocation_type == WHOLEMEMORY_MA_PINNED) {
    WM_CUDA_CHECK(cudaFreeHost(default_memory_context->ptr));
  } else if (memory_allocation_type == WHOLEMEMORY_MA_DEVICE) {
    WM_CUDA_CHECK(cudaFree(default_memory_context->ptr));
  } else {
    WHOLEMEMORY_FAIL_NOTHROW("memory_allocation_type incorrect.\n");
  }
  wholememory_initialize_tensor_desc(&default_memory_context->desc);
  default_memory_context->ptr             = nullptr;
  default_memory_context->allocation_type = WHOLEMEMORY_MA_NONE;
}

static wholememory_env_func_t default_env_func = {
  .temporary_fns =
    {
      .create_memory_context_fn  = default_create_memory_context_func,
      .destroy_memory_context_fn = default_destroy_memory_context_func,
      .malloc_fn                 = default_malloc_func,
      .free_fn                   = default_free_func,
      .global_context            = nullptr,
    },
  .output_fns = {
    .malloc_fn      = default_malloc_func,
    .free_fn        = default_free_func,
    .global_context = nullptr,
  }};

wholememory_env_func_t* get_default_env_func() { return &default_env_func; }

class CachedAllocator {
 public:
  CachedAllocator() { WHOLEMEMORY_FAIL_NOTHROW("Not implemented."); };
  ~CachedAllocator() { DropCaches(); }
  void* MallocHost() { return nullptr; }
  void* MallocDevice() { return nullptr; }
  void* MallocPinned() { return nullptr; }
  void FreeHost() {}
  void FreeDevice() {}
  void FreePinned() {}
  void DropCaches() {}

 private:
  std::mutex mu_;
};

#define K_MAX_DEVICE_COUNT (16)

static CachedAllocator* p_cached_allocators[K_MAX_DEVICE_COUNT] = {nullptr};

wholememory_env_func_t* get_cached_env_func() { WHOLEMEMORY_FAIL_NOTHROW("Not implemented."); }

void drop_env_func_cache() { WHOLEMEMORY_FAIL_NOTHROW("Not implemented."); }

}  // namespace wholememory

#ifdef __cplusplus
extern "C" {
#endif

cudaDeviceProp* get_device_prop(int dev_id) { return wholememory::get_device_prop(dev_id); }

#ifdef __cplusplus
}
#endif
