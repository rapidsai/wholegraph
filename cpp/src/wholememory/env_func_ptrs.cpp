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

#include <memory>
#include <mutex>
#include <queue>
#include <vector>

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

class ChunkedMemoryPool {
 public:
  ChunkedMemoryPool();
  ~ChunkedMemoryPool();
  void* CachedMalloc(size_t size);
  void CachedFree(void* ptr, size_t size);
  void EmptyCache();
  virtual void* MallocFnImpl(size_t size) = 0;
  virtual void FreeFnImpl(void* ptr)      = 0;

 private:
  static constexpr int kBucketCount = 64;
  std::vector<std::unique_ptr<std::mutex>> mutexes_;
  std::vector<std::queue<void*>> sized_pool_;
};
static size_t GetChunkIndex(size_t size)
{
  if (size == 0) return 0;
  int power           = 0;
  size_t shifted_size = size;
  while (shifted_size) {
    shifted_size >>= 1;
    power++;
  }
  if ((size & (size - 1)) == 0) {
    return power - 1;
  } else {
    return power;
  }
}
ChunkedMemoryPool::ChunkedMemoryPool()
{
  sized_pool_.resize(kBucketCount);
  mutexes_.resize(kBucketCount);
  for (int i = 0; i < kBucketCount; i++) {
    mutexes_[i] = std::make_unique<std::mutex>();
  }
}
ChunkedMemoryPool::~ChunkedMemoryPool() {}
void* ChunkedMemoryPool::CachedMalloc(size_t size)
{
  size_t chunked_index = GetChunkIndex(size);
  std::unique_lock<std::mutex> mlock(*mutexes_[chunked_index]);
  if (!sized_pool_[chunked_index].empty()) {
    void* ptr = sized_pool_[chunked_index].front();
    sized_pool_[chunked_index].pop();
    return ptr;
  } else {
    return MallocFnImpl(1ULL << chunked_index);
  }
  return nullptr;
}
void ChunkedMemoryPool::CachedFree(void* ptr, size_t size)
{
  size_t chunked_index = GetChunkIndex(size);
  std::unique_lock<std::mutex> mlock(*mutexes_[chunked_index]);
  sized_pool_[chunked_index].push(ptr);
}
void ChunkedMemoryPool::EmptyCache()
{
  for (int i = 0; i < kBucketCount; i++) {
    std::unique_lock<std::mutex> mlock(*mutexes_[i]);
    while (!sized_pool_[i].empty()) {
      FreeFnImpl(sized_pool_[i].front());
      sized_pool_[i].pop();
    }
  }
}
class DeviceChunkedMemoryPool : public ChunkedMemoryPool {
 public:
  explicit DeviceChunkedMemoryPool(int device_id);
  ~DeviceChunkedMemoryPool();
  void* MallocFnImpl(size_t size) override;
  void FreeFnImpl(void* ptr) override;

 protected:
  int device_id_ = -1;
};
DeviceChunkedMemoryPool::DeviceChunkedMemoryPool(int device_id) : device_id_(device_id) {}
DeviceChunkedMemoryPool::~DeviceChunkedMemoryPool() {}
void* DeviceChunkedMemoryPool::MallocFnImpl(size_t size)
{
  int old_dev;
  void* ptr;
  WM_CUDA_CHECK(cudaGetDevice(&old_dev));
  WM_CUDA_CHECK(cudaSetDevice(device_id_));
  WM_CUDA_CHECK(cudaMalloc(&ptr, size));
  WM_CUDA_CHECK(cudaSetDevice(old_dev));
  return ptr;
}
void DeviceChunkedMemoryPool::FreeFnImpl(void* ptr)
{
  int old_dev;
  WM_CUDA_CHECK(cudaGetDevice(&old_dev));
  WM_CUDA_CHECK(cudaSetDevice(device_id_));
  WM_CUDA_CHECK(cudaFree(ptr));
  WM_CUDA_CHECK(cudaSetDevice(old_dev));
}

class PinnedChunkedMemoryPool : public ChunkedMemoryPool {
 public:
  PinnedChunkedMemoryPool()  = default;
  ~PinnedChunkedMemoryPool() = default;
  void* MallocFnImpl(size_t size) override;
  void FreeFnImpl(void* ptr) override;
};
void* PinnedChunkedMemoryPool::MallocFnImpl(size_t size)
{
  void* ptr;
  WM_CUDA_CHECK(cudaMallocHost(&ptr, size));
  return ptr;
}
void PinnedChunkedMemoryPool::FreeFnImpl(void* ptr) { WM_CUDA_CHECK(cudaFreeHost(ptr)); }

class HostChunkedMemoryPool : public ChunkedMemoryPool {
 public:
  HostChunkedMemoryPool()  = default;
  ~HostChunkedMemoryPool() = default;
  void* MallocFnImpl(size_t size) override;
  void FreeFnImpl(void* ptr) override;
};
void* HostChunkedMemoryPool::MallocFnImpl(size_t size) { return malloc(size); }
void HostChunkedMemoryPool::FreeFnImpl(void* ptr) { free(ptr); }
class CachedAllocator {
 public:
  void* MallocHost(size_t size);
  void* MallocDevice(size_t size);
  void* MallocPinned(size_t size);
  void FreeHost(void* ptr, size_t size);
  void FreeDevice(void* ptr, size_t size);
  void FreePinned(void* ptr, size_t size);
  void DropCaches();
  static CachedAllocator* GetInst();

 private:
  CachedAllocator()
  {
    device_chunked_mem_pools_.resize(kMaxSupportedDeviceCount);
    for (int i = 0; i < kMaxSupportedDeviceCount; i++) {
      device_chunked_mem_pools_[i] = std::make_unique<DeviceChunkedMemoryPool>(i);
    }
    pinned_chunked_mem_pool_ = std::make_unique<PinnedChunkedMemoryPool>();
    host_chunked_mem_pool_   = std::make_unique<HostChunkedMemoryPool>();
  }
  ~CachedAllocator() {}
  CachedAllocator(const CachedAllocator& ca)                  = delete;
  const CachedAllocator& operator=(const CachedAllocator& ca) = delete;

  static CachedAllocator ca_inst_;
  std::vector<std::unique_ptr<DeviceChunkedMemoryPool>> device_chunked_mem_pools_;
  std::unique_ptr<PinnedChunkedMemoryPool> pinned_chunked_mem_pool_;
  std::unique_ptr<HostChunkedMemoryPool> host_chunked_mem_pool_;
  static constexpr int kMaxSupportedDeviceCount = 16;
};

CachedAllocator CachedAllocator::ca_inst_;
CachedAllocator* CachedAllocator::GetInst() { return &ca_inst_; }

void* CachedAllocator::MallocHost(size_t size)
{
  return host_chunked_mem_pool_->CachedMalloc(size);
}
void CachedAllocator::FreeHost(void* ptr, size_t size)
{
  host_chunked_mem_pool_->CachedFree(ptr, size);
}
void* CachedAllocator::MallocDevice(size_t size)
{
  int dev_id;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  return device_chunked_mem_pools_[dev_id]->CachedMalloc(size);
}
void CachedAllocator::FreeDevice(void* ptr, size_t size)
{
  int dev_id;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  device_chunked_mem_pools_[dev_id]->CachedFree(ptr, size);
}
void* CachedAllocator::MallocPinned(size_t size)
{
  return pinned_chunked_mem_pool_->CachedMalloc(size);
}
void CachedAllocator::FreePinned(void* ptr, size_t size)
{
  pinned_chunked_mem_pool_->CachedFree(ptr, size);
}
void CachedAllocator::DropCaches()
{
  for (int i = 0; i < kMaxSupportedDeviceCount; i++) {
    device_chunked_mem_pools_[i]->EmptyCache();
  }
  pinned_chunked_mem_pool_->EmptyCache();
  host_chunked_mem_pool_->EmptyCache();
}

void* cached_malloc_func(wholememory_tensor_description_t* tensor_description,
                         wholememory_memory_allocation_type_t memory_allocation_type,
                         void* memory_context,
                         void* /*global_context*/)
{
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  void* ptr                    = nullptr;
  CachedAllocator* cached_inst = CachedAllocator::GetInst();
  int devid;
  WM_CUDA_CHECK(cudaGetDevice((&devid)));
  try {
    if (memory_allocation_type == WHOLEMEMORY_MA_HOST) {
      ptr = cached_inst->MallocHost(wholememory_get_memory_size_from_tensor(tensor_description));
      if (ptr == nullptr) { WHOLEMEMORY_FAIL_NOTHROW("cached malloc host returned nullptr.\n"); }
    } else if (memory_allocation_type == WHOLEMEMORY_MA_PINNED) {
      ptr = cached_inst->MallocPinned(wholememory_get_memory_size_from_tensor(tensor_description));
      if (ptr == nullptr) { WHOLEMEMORY_FAIL_NOTHROW("cached malloc pinned returned nullptr.\n"); }
    } else if (memory_allocation_type == WHOLEMEMORY_MA_DEVICE) {
      ptr = cached_inst->MallocDevice(wholememory_get_memory_size_from_tensor(tensor_description));
      if (ptr == nullptr) { WHOLEMEMORY_FAIL_NOTHROW("cached malloc device returned nullptr.\n"); }
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

void cached_free_func(void* memory_context, void* /*global_context*/)
{
  CachedAllocator* cached_inst = CachedAllocator::GetInst();
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  auto memory_allocation_type  = default_memory_context->allocation_type;
  if (memory_allocation_type == WHOLEMEMORY_MA_HOST) {
    cached_inst->FreeHost(default_memory_context->ptr,
                          wholememory_get_memory_size_from_tensor(&default_memory_context->desc));
  } else if (memory_allocation_type == WHOLEMEMORY_MA_PINNED) {
    cached_inst->FreePinned(default_memory_context->ptr,
                            wholememory_get_memory_size_from_tensor(&default_memory_context->desc));
  } else if (memory_allocation_type == WHOLEMEMORY_MA_DEVICE) {
    cached_inst->FreeDevice(default_memory_context->ptr,
                            wholememory_get_memory_size_from_tensor(&default_memory_context->desc));
  } else {
    WHOLEMEMORY_FAIL_NOTHROW("memory_allocation_type incorrect.\n");
  }
  wholememory_initialize_tensor_desc(&default_memory_context->desc);
  default_memory_context->ptr             = nullptr;
  default_memory_context->allocation_type = WHOLEMEMORY_MA_NONE;
}

static wholememory_env_func_t cached_env_func = {
  .temporary_fns =
    {
      .create_memory_context_fn  = default_create_memory_context_func,
      .destroy_memory_context_fn = default_destroy_memory_context_func,
      .malloc_fn                 = cached_malloc_func,
      .free_fn                   = cached_free_func,
      .global_context            = nullptr,
    },
  .output_fns = {
    .malloc_fn      = cached_malloc_func,
    .free_fn        = cached_free_func,
    .global_context = nullptr,
  }};

wholememory_env_func_t* get_cached_env_func() { return &cached_env_func; }

void drop_cached_env_func_cache() { CachedAllocator::GetInst()->DropCaches(); }

}  // namespace wholememory

#ifdef __cplusplus
extern "C" {
#endif

cudaDeviceProp* get_device_prop(int dev_id) { return wholememory::get_device_prop(dev_id); }

#ifdef __cplusplus
}
#endif
