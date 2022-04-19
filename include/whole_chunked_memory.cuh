#pragma once

#include "whole_chunked_memory.h"

namespace whole_memory {

template <typename HandleType, typename DataType>
class PtrGen {
 public:
  __device__ __host__ __forceinline__ explicit PtrGen(HandleType* h, size_t base_elt_offset = 0) {
    h_ = (DataType*)h + base_elt_offset;
  }
  __device__ __host__ __forceinline__ DataType* At(size_t elt_offset) const {
    return ((DataType*)h_) + elt_offset;
  }
 private:
  DataType* h_;
};

template <typename DataType>
class PtrGen<const WholeChunkedMemoryHandle, DataType> {
 public:
  __device__ __host__ __forceinline__ explicit PtrGen(const WholeChunkedMemoryHandle* h, size_t base_elt_offset = 0) {
    base_elt_offset_ = base_elt_offset;
    h_ = h;
  }
  __device__ __host__ __forceinline__ DataType* At(size_t elt_offset) const {
    elt_offset += base_elt_offset_;
    int chunk_idx = elt_offset * sizeof(DataType) / h_->chunk_size;
    size_t offset_in_chunk = elt_offset * sizeof(DataType) - chunk_idx * h_->chunk_size;
    return (DataType*)((char*)h_->chunked_ptrs[chunk_idx] + offset_in_chunk);
  }
 protected:
  const WholeChunkedMemoryHandle* h_;
  size_t base_elt_offset_ = 0;
};

template <typename DataType>
class PtrGen<WholeChunkedMemoryHandle, DataType> : public PtrGen<const WholeChunkedMemoryHandle, DataType> {
 public:
  __device__ __host__ __forceinline__ explicit PtrGen(WholeChunkedMemoryHandle *h, size_t base_elt_offset = 0) : PtrGen<
      const WholeChunkedMemoryHandle,
      DataType>(h, base_elt_offset) {}
};

}