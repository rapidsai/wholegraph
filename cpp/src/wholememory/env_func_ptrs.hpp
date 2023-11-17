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
#pragma once

#include <wholememory/env_func_ptrs.h>

namespace wholememory {

struct default_memory_context_t {
  wholememory_tensor_description_t desc;
  wholememory_memory_allocation_type_t allocation_type;
  void* ptr;
};

/**
 * @brief : Default environment functions for memory allocation.
 * Will use cudaMalloc/cudaFree, cudaMallocHost/cudaFreeHost, malloc/free.
 * Useful for function tests, NOT designed for performance tests.
 *
 * @return : pointers to the functions of current CUDA device
 */
wholememory_env_func_t* get_default_env_func();

/**
 * @brief : Environment functions for memory allocation with caches.
 * Will cache allocated memory blocks, and reuse if possible.
 * Minimal block size is 256 bytes, block with size < 1G bytes is aligned to power of 2,
 * block with size >= 1G bytes is aligned to 1G bytes.
 * Useful for performance tests. Need warm up to fill caches.
 *
 * @return : pointers to the functions of current CUDA device
 */
wholememory_env_func_t* get_cached_env_func();

/**
 * @brief : drop all caches of inside cached allocator of current CUDA device
 */
void drop_cached_env_func_cache();

}  // namespace wholememory
