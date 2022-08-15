/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cuda_runtime_api.h>

#include <functional>
#include <map>

namespace whole_graph {

typedef struct {
  // pointer of this temporary memory
  void *ptr = nullptr;
  // size of this temporary memory
  size_t size = 0;
  // private of this allocation
  void *private_data = nullptr;
} TempMemoryHandle;

struct CUDAEnvFns {
  // stream sync function
  std::function<void(cudaStream_t, const char *file, int line)> sync_fn;
  // temporary device memory allocation function
  std::function<void *(size_t, TempMemoryHandle *)> allocate_temp_fn;
  // temporary device memory free function
  std::function<void(TempMemoryHandle *)> free_temp_fn;
  // temporary host memory allocation function
  std::function<void *(size_t, TempMemoryHandle *)> allocate_host_temp_fn;
  // temporary host memory free function
  std::function<void(TempMemoryHandle *)> free_host_temp_fn;
};

#define CUDA_STREAM_SYNC(CUDA_ENV_FNS, STREAM) \
  (CUDA_ENV_FNS).sync_fn((STREAM), __FILE__, __LINE__)

class WMThrustAllocator {
 public:
  typedef char value_type;
  explicit WMThrustAllocator(CUDAEnvFns fns) : fns_(fns) {
  }
  ~WMThrustAllocator() = default;

  char *allocate(std::ptrdiff_t size);
  void deallocate(char *p, size_t size);
  void deallocate_all();

  CUDAEnvFns fns_;
  std::map<void *, TempMemoryHandle> tmhs_;
};

}// namespace whole_graph