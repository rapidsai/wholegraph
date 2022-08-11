#include "cuda_env_fns.h"

#include <assert.h>
#include <cuda_runtime_api.h>

#include "macros.h"

namespace whole_graph {

static void StreamSyncFunc(cudaStream_t s, const char* file, int line) {
  auto result = cudaStreamSynchronize(s);
  if (result != cudaSuccess) {
    const char* p_err_str = cudaGetErrorName(result);
    fprintf(stderr, "File %s Line %d %s returned %s.\n",
        __FILE__, __LINE__, "cudaStreamSynchronize", p_err_str);
    abort();
  }
}

static void* DefaultAllocateFunc(size_t size, TempMemoryHandle* t) {
  t->size = size;
  if (t->size < 128) t->size = 128;
  WM_CUDA_CHECK(cudaMalloc(&t->ptr, t->size));
  t->private_data = nullptr;
  return t->ptr;
}

static void DefaultFreeFunc(TempMemoryHandle* t) {
  WM_CUDA_CHECK(cudaFree(t->ptr));
}

static void* DefaultAllocateHostFunc(size_t size, TempMemoryHandle* t) {
  t->size = size;
  if (t->size < 128) t->size = 128;
  WM_CUDA_CHECK(cudaMallocHost(&t->ptr, t->size));
  t->private_data = nullptr;
  return t->ptr;
}

static void DefaultFreeHostFunc(TempMemoryHandle* t) {
  WM_CUDA_CHECK(cudaFreeHost(t->ptr));
}


char* WMThrustAllocator::allocate(std::ptrdiff_t size) {
  TempMemoryHandle tmh;
  fns_.allocate_temp_fn(size, &tmh);
  tmhs_.emplace(tmh.ptr, tmh);
  return (char*)tmh.ptr;
}

void WMThrustAllocator::deallocate(char *p, size_t size) {
  auto it = tmhs_.find(p);
  assert(it != tmhs_.end());
  fns_.free_temp_fn(&it->second);
  tmhs_.erase(p);
}

void WMThrustAllocator::deallocate_all() {
  while(!tmhs_.empty()) {
    auto it = tmhs_.begin();
    fns_.free_temp_fn(&it->second);
    tmhs_.erase(it->first);
  }
}

CUDAEnvFns default_cuda_env_fns {
  .sync_fn = StreamSyncFunc,
  .allocate_temp_fn = DefaultAllocateFunc,
  .free_temp_fn = DefaultFreeFunc,
  .allocate_host_temp_fn = DefaultAllocateHostFunc,
  .free_host_temp_fn = DefaultFreeHostFunc
};

}