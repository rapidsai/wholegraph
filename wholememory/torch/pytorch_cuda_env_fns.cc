#include "pytorch_cuda_env_fns.h"

#include <cuda_runtime_api.h>

#include "macros.h"

namespace whole_memory {

namespace pytorch {

static void StreamSyncFunc(cudaStream_t s, const char* file, int line) {
  auto result = cudaStreamSynchronize(s);
  if (result != cudaSuccess) {
    const char* p_err_str = cudaGetErrorName(result);
    fprintf(stderr, "File %s Line %d %s returned %s.\n",
            __FILE__, __LINE__, "cudaStreamSynchronize", p_err_str);
    abort();
  }
}

static size_t AlignedInt64Count(size_t size) {
  return (size + sizeof(int64_t) - 1) / sizeof(int64_t);
}

static void* PytorchAllocateFunc(size_t size, whole_memory::TempMemoryHandle* tmh, torch::Device d) {
  size = AlignUp(size, 256);
  auto to = torch::TensorOptions().device(d).dtype(torch::kInt64).requires_grad(false);
  size_t aligned_int64_count = AlignedInt64Count(size);
  if (aligned_int64_count < 16) aligned_int64_count = 16;
  tmh->size = aligned_int64_count * sizeof(int64_t);
  auto* t = new torch::Tensor;
  *t = torch::empty({(long)aligned_int64_count}, to);
  tmh->ptr = t->data_ptr();
  tmh->private_data = t;
  return tmh->ptr;
}

static void PytorchFreeFunc(whole_memory::TempMemoryHandle* tmh) {
  auto* t = (torch::Tensor*)tmh->private_data;
  delete t;
  tmh->ptr = nullptr;
  tmh->size = 0;
  tmh->private_data = nullptr;
}

whole_memory::CUDAEnvFns GetCUDAEnvFns(torch::Device d) {
  whole_memory::CUDAEnvFns cuda_env_fns;
  cuda_env_fns.sync_fn = StreamSyncFunc;
  cuda_env_fns.allocate_temp_fn = std::bind(PytorchAllocateFunc, std::placeholders::_1, std::placeholders::_2, d);
  cuda_env_fns.free_temp_fn = PytorchFreeFunc;
  cuda_env_fns.allocate_host_temp_fn = std::bind(PytorchAllocateFunc,
                                                 std::placeholders::_1,
                                                 std::placeholders::_2,
                                                 torch::Device(torch::Device::Type::CPU));
  cuda_env_fns.free_host_temp_fn = PytorchFreeFunc;
  return cuda_env_fns;
}

}

}