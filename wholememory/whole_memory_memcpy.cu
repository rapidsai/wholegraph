#include "whole_chunked_memory.h"
#include "whole_chunked_memory.cuh"

#include <stdint.h>

#include <iostream>

namespace whole_memory {

static inline size_t GetAlignment(size_t offset, const void* ptr, size_t copy_bytes) {
  size_t alignment = 16;
  while (alignment > 1) {
    if (offset % alignment == 0 && ((size_t)ptr) % alignment == 0 && copy_bytes % alignment == 0) break;
    alignment /= 2;
  }
  return alignment;
}

template <typename T>
__global__ void MemcpyToWholeChunkedMemoryKernel(WholeChunkedMemoryHandle* wcmh, size_t elt_offset, const T* src, size_t copy_elt_count) {
  PtrGen<const WholeChunkedMemoryHandle, T> ptr_gen(wcmh, elt_offset);
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < copy_elt_count; idx += (size_t)gridDim.x * blockDim.x) {
    *(ptr_gen.At(idx)) = src[idx];
  }
}

template <typename T>
void DoMemcpyToWholeChunkedMemory(WholeChunkedMemoryHandle* wcmh, size_t offset_in_bytes, const void* src, size_t copy_bytes, cudaStream_t stream) {
  const auto* src_ptr = (const T*)src;
  size_t copy_elt_count = copy_bytes / sizeof(T);
  size_t base_elt_offset = offset_in_bytes / sizeof(T);
  static constexpr int kBlockSize = 256;
  size_t all_block_count = (copy_elt_count + kBlockSize - 1) / kBlockSize;
  int block_count;
  if (all_block_count < INT32_MAX) {
    block_count = (int)all_block_count;
  } else {
    block_count = INT32_MAX;
  }
  MemcpyToWholeChunkedMemoryKernel<<<block_count, kBlockSize, 0, stream>>>(wcmh, base_elt_offset, src_ptr, copy_elt_count);
}

void WcmmpMemcpyToWholeChunkedMemory(WholeChunkedMemoryHandle* wcmh, size_t offset_in_bytes, const void* src, size_t copy_bytes, cudaStream_t stream) {
  auto alignment = (int8_t)GetAlignment(offset_in_bytes, src, copy_bytes);
  switch (alignment) {
    case 16: {
      DoMemcpyToWholeChunkedMemory<int4>(wcmh, offset_in_bytes, src, copy_bytes, stream);
      break;
    }
    case 8: {
      DoMemcpyToWholeChunkedMemory<int2>(wcmh, offset_in_bytes, src, copy_bytes, stream);
      break;
    }
    case 4: {
      DoMemcpyToWholeChunkedMemory<int>(wcmh, offset_in_bytes, src, copy_bytes, stream);
      break;
    }
    case 2: {
      DoMemcpyToWholeChunkedMemory<int16_t>(wcmh, offset_in_bytes, src, copy_bytes, stream);
      break;
    }
    case 1: {
      DoMemcpyToWholeChunkedMemory<int8_t>(wcmh, offset_in_bytes, src, copy_bytes, stream);
      break;
    }
    default: {
      std::cerr << "WcmmpMemcpyToWholeChunkedMemory alignment=" << alignment << std::endl;
      abort();
    }
  }
}

template <typename T>
__global__ void MemcpyFromWholeChunkedMemoryKernel(T* dst, WholeChunkedMemoryHandle* wcmh, size_t elt_offset, size_t copy_elt_count) {
  PtrGen<const WholeChunkedMemoryHandle, T> ptr_gen(wcmh, elt_offset);
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < copy_elt_count; idx += (size_t)gridDim.x * blockDim.x) {
    dst[idx] = *(ptr_gen.At(idx));
  }
}

template <typename T>
void DoMemcpyFromWholeChunkedMemory(void* dst, WholeChunkedMemoryHandle* wcmh, size_t offset_in_bytes, size_t copy_bytes, cudaStream_t stream) {
  auto* dst_ptr = (T*)dst;
  size_t copy_elt_count = copy_bytes / sizeof(T);
  size_t base_elt_offset = offset_in_bytes / sizeof(T);
  static constexpr int kBlockSize = 256;
  size_t all_block_count = copy_elt_count / kBlockSize;
  int block_count;
  if (all_block_count < INT32_MAX) {
    block_count = (int)all_block_count;
  } else {
    block_count = INT32_MAX;
  }
  MemcpyFromWholeChunkedMemoryKernel<<<block_count, kBlockSize, 0, stream>>>(dst_ptr, wcmh, base_elt_offset, copy_elt_count);
}

void WcmmpMemcpyFromWholeChunkedMemory(void* dst, WholeChunkedMemoryHandle* wcmh, size_t offset_in_bytes, size_t copy_bytes, cudaStream_t stream) {
  auto alignment = (int8_t)GetAlignment(offset_in_bytes, dst, copy_bytes);
  switch (alignment) {
    case 16: {
      DoMemcpyFromWholeChunkedMemory<int4>(dst, wcmh, offset_in_bytes, copy_bytes, stream);
      break;
    }
    case 8: {
      DoMemcpyFromWholeChunkedMemory<int2>(dst, wcmh, offset_in_bytes, copy_bytes, stream);
      break;
    }
    case 4: {
      DoMemcpyFromWholeChunkedMemory<int>(dst, wcmh, offset_in_bytes, copy_bytes, stream);
      break;
    }
    case 2: {
      DoMemcpyFromWholeChunkedMemory<int16_t>(dst, wcmh, offset_in_bytes, copy_bytes, stream);
      break;
    }
    case 1: {
      DoMemcpyFromWholeChunkedMemory<int8_t>(dst, wcmh, offset_in_bytes, copy_bytes, stream);
      break;
    }
    default: {
      std::cerr << "WcmmpMemcpyToWholeChunkedMemory alignment=" << alignment << std::endl;
      abort();
    }
  }
}

}