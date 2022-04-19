#pragma once

#include <assert.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <wait.h>
#include <sys/time.h>

#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "whole_chunked_memory.cuh"
#include "random.cuh"

#define TIME_DIFF_US(TVS, TVE) ((TVE.tv_sec - TVS.tv_sec)*1000ULL*1000ULL + (TVE.tv_usec - TVS.tv_usec))

#define DIV_UP(X, Y) (((X) + (Y) - 1) / (Y))

void WriteDataCPU(int* data, int count, int offset) {
  for (int i = 0; i < count; i++) {
    data[i] = i + offset;
  }
}

void CheckDataCPU(const int* data, int count, int offset) {
  for (int i = 0; i < count; i++) {
    assert(data[i] == i + offset);
  }
}

__global__ void WriteDataKernel(int* data, int count, int offset) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= count) return;
  data[idx] = idx + offset;
}

__global__ void CheckDataKernel(const int* data, int count, int offset, int dev_id) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= count) return;
  if (data[idx] != idx + offset) {
    //printf("dev_id=%d, data[%d]=%d, offset=%d, should be %d, diff=%d\n",
    //       dev_id, idx, data[idx], offset, idx + offset, idx + offset - data[idx]);
    assert(0);
  }
}

__global__ void GatherFloat(float* output, const float* param, int* indice, size_t embedding_dim) {
  int bidx = blockIdx.x;
  int idx = indice[bidx];
  float* output_ptr = (float*)(output + (size_t)bidx * embedding_dim);
  const float* param_ptr = (const float*)(param + (size_t)idx * embedding_dim);
  for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
    output_ptr[i] = param_ptr[i];
  }
}

__global__ void GatherChunkedFloat(float* output, const whole_memory::WholeChunkedMemoryHandle* h, int* indice, size_t embedding_dim) {
  int bidx = blockIdx.x;
  int idx = indice[bidx];
  whole_memory::PtrGen<const whole_memory::WholeChunkedMemoryHandle, float> ptr_gen(h);
  float* output_ptr = (float*)(output + (size_t)bidx * embedding_dim);
  const float* param_ptr = (const float*)(ptr_gen.At((size_t)idx * embedding_dim));
  for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
    output_ptr[i] = param_ptr[i];
  }
}

__global__ void GatherFloat4(float* output, const float* param, int* indice, size_t embedding_dim) {
  int bidx = blockIdx.x;
  int idx = indice[bidx];
  float4* output_ptr = (float4*)(output + (size_t)bidx * embedding_dim);
  const float4* param_ptr = (const float4*)(param + (size_t)idx * embedding_dim);
  for (int i = threadIdx.x; i < embedding_dim / 4; i += blockDim.x) {
    output_ptr[i] = param_ptr[i];
  }
}

__global__ void GatherChunkedFloat4(float* output, const whole_memory::WholeChunkedMemoryHandle* h, int* indice, size_t embedding_dim) {
  int bidx = blockIdx.x;
  int idx = indice[bidx];
  whole_memory::PtrGen<const whole_memory::WholeChunkedMemoryHandle, float> ptr_gen(h);
  float4* output_ptr = (float4*)(output + (size_t)bidx * embedding_dim);
  const float4* param_ptr = (const float4*)(ptr_gen.At((size_t)idx * embedding_dim));
  for (int i = threadIdx.x; i < embedding_dim / 4; i += blockDim.x) {
    output_ptr[i] = param_ptr[i];
  }
}

void WriteData(int* data, int count, int offset) {
  int block_count = DIV_UP(count, 256);
  WriteDataKernel<<<block_count, 256>>>(data, count, offset);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  std::cout << "Done writing " << count << " int values." << std::endl;
}
void CheckData(int* data, int count, int offset) {
  int dev_id = -1;
  assert(cudaGetDevice(&dev_id) == cudaSuccess);
  int block_count = DIV_UP(count, 256);
  CheckDataKernel<<<block_count, 256>>>(data, count, offset, dev_id);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  std::cout << "Done checking " << count << " int values." << std::endl;
}

template <typename T, int OpType>
class GroupWholeMemoryTestOperator {
};

template <typename T>
class GroupWholeMemoryTestOperator<T, 0> {
 public:
  __device__ __forceinline__ void Op(T* local_mem, T* whole_mem) {
    *local_mem = *whole_mem;
  }
};

template <typename T>
class GroupWholeMemoryTestOperator<T, 1> {
 public:
  __device__ __forceinline__ void Op(T* local_mem, T* whole_mem) {
    *whole_mem = *local_mem;
  }
};

template <typename T>
class GroupWholeMemoryTestOperator<T, 2> {
 public:
  __device__ __forceinline__ void Op(T* local_mem, T* whole_mem) {
    atomicAdd_system(whole_mem, *local_mem);
  }
};

template <typename EmbHandle, typename EmbDataType, int GroupCount, int OpType>
__global__ void GroupWholeMemoryTestKernel(EmbDataType* local_mem, EmbHandle* src, const int64_t* indice, int indice_count, int64_t entry_count) {
  int64_t thread_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx = thread_idx / GroupCount;
  if (idx >= indice_count) return;
  int64_t src_idx = indice[idx] % entry_count;
  int lane_idx = thread_idx % GroupCount;
  whole_memory::PtrGen<EmbHandle, EmbDataType> ptr_gen(src);
  GroupWholeMemoryTestOperator<EmbDataType, OpType> gwmtop;
  gwmtop.Op(&local_mem[thread_idx], ptr_gen.At(src_idx * GroupCount + lane_idx));
}

template <typename EmbHandle, typename EmbDataType, int GroupCount, int OpType>
inline void GroupWholeMemoryTest(EmbDataType* local_mem, EmbHandle* src, const int64_t* indice, int indice_count, int64_t entry_count, cudaStream_t stream = nullptr) {
  static_assert(GroupCount <= 1024, "GroupCount should be less than 1024");
  static_assert(GroupCount > 0, "GroupCount should be larger than 0.");
  static_assert((GroupCount & (GroupCount - 1)) == 0, "GroupCount should be power of 2");
  int thread_count = GroupCount;
  if (thread_count < 64) thread_count = 64;
  int group_per_block = thread_count / GroupCount;
  int block_count = (indice_count + group_per_block - 1) / group_per_block;
  GroupWholeMemoryTestKernel<EmbHandle, EmbDataType, GroupCount, OpType><<<block_count, thread_count, 0, stream>>>(
      local_mem, src, indice, indice_count, entry_count
  );
  assert(cudaGetLastError() == cudaSuccess);
}
