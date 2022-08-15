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
#include "whole_memory_graph.h"

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <thrust/scan.h>

#include <random>

#include "data_type.h"
#include "macros.h"
#include "random.cuh"
#include "whole_chunked_memory.cuh"
#include "whole_graph_mixed_graph.cuh"
#include "whole_memory.h"

namespace whole_graph {

template<typename IdType, typename WMIdType, typename WMOffsetType>
__global__ void GetSampleCountWithoutReplacementKernel(int *sample_offset,
                                                       const IdType *input_nodes,
                                                       int input_node_count,
                                                       WMOffsetType *wm_csr_row_ptr,
                                                       WMIdType *wm_csr_col_ptr,
                                                       int max_sample_count) {
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  int input_idx = gidx;
  if (input_idx >= input_node_count) return;
  IdType nid = input_nodes[input_idx];
  whole_graph::PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);
  // sample_count <= 0 means sample all.
  if (max_sample_count > 0) {
    neighbor_count = min(neighbor_count, max_sample_count);
  }
  sample_offset[input_idx] = neighbor_count;
}

__device__ __forceinline__ int Log2UpCUDA(int x) {
  if (x <= 2) return x - 1;
  return 32 - __clz(x - 1);
}

template<typename IdType, typename LocalIdType, typename WMIdType, typename WMOffsetType, int BLOCK_DIM = 32, int ITEMS_PER_THREAD = 1>
__global__ void UnWeightedSampleWithOutReplacementKernel(IdType *output,
                                                         LocalIdType *src_lid,
                                                         const int *sample_offset,
                                                         const IdType *input_nodes,
                                                         int input_node_count,
                                                         WMOffsetType *wm_csr_row_ptr,
                                                         WMIdType *wm_csr_col_ptr,
                                                         int max_sample_count,
                                                         unsigned long long random_seed) {
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  RandomNumGen rng(gidx, random_seed);
  rng.NextValue();
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  whole_graph::PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  whole_graph::PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  IdType nid = input_nodes[input_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);
  if (neighbor_count <= 0) return;
  int offset = sample_offset[input_idx];
  // use all neighbors if neighbors less than max_sample_count
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += blockDim.x) {
      IdType gid = *csr_col_ptr_gen.At(start + sample_id);
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = input_idx;
    }
    return;
  }
  uint64_t sa_p[ITEMS_PER_THREAD];
  int M = max_sample_count;
  int N = neighbor_count;
  //UnWeightedIndexSampleWithOutReplacement<BLOCK_DIM, ITEMS_PER_THREAD>(M, N, sa_p, rng);
  typedef cub::BlockRadixSort<uint64_t, BLOCK_DIM, ITEMS_PER_THREAD> BlockRadixSort;
  struct IntArray {
    int value[BLOCK_DIM * ITEMS_PER_THREAD];
  };
  struct SampleSharedData {
    IntArray s;
    IntArray p;
    IntArray q;
    IntArray chain;
    IntArray last_chain_tmp;
  };
  __shared__ union {
    typename BlockRadixSort::TempStorage temp_storage;
    SampleSharedData sample_shared_data;
  } shared_data;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    uint32_t idx = i * BLOCK_DIM + threadIdx.x;
    uint32_t r = idx < M ? rng.RandomMod(N - idx) : N;
    sa_p[i] = ((uint64_t) r << 32UL) | idx;
  }
  __syncthreads();
  BlockRadixSort(shared_data.temp_storage).SortBlockedToStriped(sa_p);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_DIM + threadIdx.x;
    int s = (int) (sa_p[i] >> 32UL);
    shared_data.sample_shared_data.s.value[idx] = s;
    int p = sa_p[i] & 0xFFFFFFFF;
    shared_data.sample_shared_data.p.value[idx] = p;
    if (idx < M) shared_data.sample_shared_data.q.value[p] = idx;
    shared_data.sample_shared_data.chain.value[idx] = idx;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_DIM + threadIdx.x;
    int si = shared_data.sample_shared_data.s.value[idx];
    int si1 = shared_data.sample_shared_data.s.value[idx + 1];
    if (idx < M && (idx == M - 1 || si != si1) && si >= N - M) {
      shared_data.sample_shared_data.chain.value[N - si - 1] = shared_data.sample_shared_data.p.value[idx];
    }
  }
  __syncthreads();
  for (int step = 0; step < Log2UpCUDA(M); ++step) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int idx = i * BLOCK_DIM + threadIdx.x;
      shared_data.sample_shared_data.last_chain_tmp.value[idx] = shared_data.sample_shared_data.chain.value[idx];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int idx = i * BLOCK_DIM + threadIdx.x;
      if (idx < M) {
        shared_data.sample_shared_data.chain.value[idx] =
            shared_data.sample_shared_data.last_chain_tmp.value[shared_data.sample_shared_data.last_chain_tmp.value[idx]];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_DIM + threadIdx.x;
    shared_data.sample_shared_data.last_chain_tmp.value[idx] = N - shared_data.sample_shared_data.chain.value[idx] - 1;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_DIM + threadIdx.x;
    int ai;
    if (idx < M) {
      int qi = shared_data.sample_shared_data.q.value[idx];
      if (idx == 0 || qi == 0
          || shared_data.sample_shared_data.s.value[qi] != shared_data.sample_shared_data.s.value[qi - 1]) {
        ai = shared_data.sample_shared_data.s.value[qi];
      } else {
        int prev_i = shared_data.sample_shared_data.p.value[qi - 1];
        ai = shared_data.sample_shared_data.last_chain_tmp.value[prev_i];
      }
      sa_p[i] = ai;
    }
  }
  // Output
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_DIM + threadIdx.x;
    int ai = sa_p[i];
    if (idx < M) {
      IdType gid = *csr_col_ptr_gen.At(start + ai);
      output[offset + idx] = gid;
      if (src_lid) src_lid[offset + idx] = (LocalIdType) input_idx;
    }
  }
}

template<typename IdType, typename LocalIdType, typename WMIdType, typename WMOffsetType>
__global__ void SampleAllKernel(IdType *output,
                                LocalIdType *src_lid,
                                const int *sample_offset,
                                const IdType *input_nodes,
                                int input_node_count,
                                WMOffsetType *wm_csr_row_ptr,
                                WMIdType *wm_csr_col_ptr) {
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  whole_graph::PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  whole_graph::PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  IdType nid = input_nodes[input_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);
  if (neighbor_count <= 0) return;
  int offset = sample_offset[input_idx];
  for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += blockDim.x) {
    int neighbor_idx = sample_id;
    IdType gid = *csr_col_ptr_gen.At(start + neighbor_idx);
    output[offset + sample_id] = gid;
    if (src_lid) src_lid[offset + sample_id] = (LocalIdType) input_idx;
  }
}

template<typename IdType, typename LocalIdType, typename WMIdType, typename WMOffsetType>
__global__ void LargeSampleKernel(IdType *output,
                                  LocalIdType *src_lid,
                                  const int *sample_offset,
                                  const IdType *input_nodes,
                                  int input_node_count,
                                  WMOffsetType *wm_csr_row_ptr,
                                  WMIdType *wm_csr_col_ptr,
                                  int max_sample_count,
                                  unsigned long long random_seed) {
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  RandomNumGen rng(gidx, random_seed);
  rng.NextValue();
  whole_graph::PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  whole_graph::PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  IdType nid = input_nodes[input_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);
  int offset = sample_offset[input_idx];
  // sample all
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += blockDim.x) {
      int neighbor_idx = sample_id;
      IdType gid = *csr_col_ptr_gen.At(start + neighbor_idx);
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = (LocalIdType) input_idx;
    }
    return;
  }
  for (int sample_id = threadIdx.x; sample_id < max_sample_count; sample_id += blockDim.x) {
    output[offset + sample_id] = (IdType) sample_id;
    if (src_lid) src_lid[offset + sample_id] = (LocalIdType) input_idx;
  }
  __syncthreads();
  for (int idx = max_sample_count + threadIdx.x; idx < neighbor_count; idx += blockDim.x) {
    const int rand_num = rng.RandomMod(idx + 1);
    if (rand_num < max_sample_count) {
      atomicMax((int *) (output + offset + rand_num), idx);
    }
  }
  __syncthreads();
  for (int sample_id = threadIdx.x; sample_id < max_sample_count; sample_id += blockDim.x) {
    int neighbor_idx = *(int *) (output + offset + sample_id);
    output[offset + sample_id] = *csr_col_ptr_gen.At(start + neighbor_idx);
  }
}

template<typename IdType, typename WMIdType, typename WMOffsetType>
void UnweightedSampleWithoutReplacementCommon(const std::function<void *(size_t)> &sample_output_allocator,
                                              const std::function<void *(size_t)> &center_localid_allocator,
                                              int *sample_offset,
                                              void *wm_csr_row_ptr,
                                              void *wm_csr_col_ptr,
                                              const void *center_nodes,
                                              int center_node_count,
                                              int max_sample_count,
                                              const CUDAEnvFns &cuda_env_fns,
                                              cudaStream_t stream) {
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  thread_local std::uniform_int_distribution<unsigned long long> distrib;
  unsigned long long random_seed = distrib(gen);
  whole_graph::TempMemoryHandle tmh;
  cuda_env_fns.allocate_temp_fn(sizeof(int) * (center_node_count + 1), &tmh);
  int *sample_count = (int *) tmh.ptr;
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  GetSampleCountWithoutReplacementKernel<IdType, WMIdType, WMOffsetType><<<DivUp(center_node_count,
                                                                                 32),
                                                                           32, 0, stream>>>(sample_count,
                                                                                            (const IdType *) center_nodes,
                                                                                            center_node_count,
                                                                                            (WMOffsetType *) wm_csr_row_ptr,
                                                                                            (WMIdType *) wm_csr_col_ptr,
                                                                                            max_sample_count);
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  WMThrustAllocator allocator(cuda_env_fns);
  thrust::exclusive_scan(thrust::cuda::par(allocator).on(stream),
                         sample_count,
                         sample_count + center_node_count + 1,
                         sample_offset);
  int count;
  WM_CUDA_CHECK(cudaMemcpyAsync(&count,
                                sample_offset + center_node_count,
                                sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  cuda_env_fns.free_temp_fn(&tmh);
  allocator.deallocate_all();
  auto *sample_output = (IdType *) sample_output_allocator(count);
  auto *src_lid = (int *) center_localid_allocator(count);
  if (max_sample_count <= 0) {
    SampleAllKernel<IdType, int, WMIdType, WMOffsetType><<<center_node_count, 64, 0, stream>>>(sample_output,
                                                                                               src_lid,
                                                                                               sample_offset,
                                                                                               (const IdType *) center_nodes,
                                                                                               center_node_count,
                                                                                               (WMOffsetType *) wm_csr_row_ptr,
                                                                                               (WMIdType *) wm_csr_col_ptr);
    WM_CUDA_CHECK(cudaGetLastError());
    CUDA_STREAM_SYNC(cuda_env_fns, stream);
    return;
  }
  if (max_sample_count > 1024) {
    LargeSampleKernel<IdType, int, WMIdType, WMOffsetType><<<center_node_count, 32, 0, stream>>>(sample_output,
                                                                                                 src_lid,
                                                                                                 sample_offset,
                                                                                                 (const IdType *) center_nodes,
                                                                                                 center_node_count,
                                                                                                 (WMOffsetType *) wm_csr_row_ptr,
                                                                                                 (WMIdType *) wm_csr_col_ptr,
                                                                                                 max_sample_count,
                                                                                                 random_seed);
    WM_CUDA_CHECK(cudaGetLastError());
    CUDA_STREAM_SYNC(cuda_env_fns, stream);
    return;
  }
  typedef void (*UnWeightedSampleFuncType)(IdType *,
                                           int *,
                                           const int *,
                                           const IdType *,
                                           int,
                                           WMOffsetType *,
                                           WMIdType *,
                                           int,
                                           unsigned long long);
  static const UnWeightedSampleFuncType func_array[32] = {
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 32, 1>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 32, 2>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 32, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 64, 2>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 64, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 64, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 128, 2>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 128, 2>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 128, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 128, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 128, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 128, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 2>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 2>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 2>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 2>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 3>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 4>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 4>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 4>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 4>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 4>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 4>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 4>,
      UnWeightedSampleWithOutReplacementKernel<IdType, int, WMIdType, WMOffsetType, 256, 4>};
  static const int warp_count_array[32] = {1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8,
                                           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
  int func_idx = (max_sample_count - 1) / 32;
  func_array[func_idx]<<<center_node_count, warp_count_array[func_idx] * 32, 0, stream>>>(sample_output,
                                                                                          src_lid,
                                                                                          sample_offset,
                                                                                          (const IdType *) center_nodes,
                                                                                          center_node_count,
                                                                                          (WMOffsetType *) wm_csr_row_ptr,
                                                                                          (WMIdType *) wm_csr_col_ptr,
                                                                                          max_sample_count,
                                                                                          random_seed);
  WM_CUDA_CHECK(cudaGetLastError());
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
}

template<typename IdType>
void UnweightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                        const std::function<void *(size_t)> &center_localid_allocator,
                                        int *sample_offset,
                                        void *wm_csr_row_ptr,
                                        void *wm_csr_col_ptr,
                                        const void *center_nodes,
                                        int center_node_count,
                                        int max_sample_count,
                                        const CUDAEnvFns &cuda_env_fns,
                                        cudaStream_t stream) {
  UnweightedSampleWithoutReplacementCommon<IdType, IdType, int64_t>(sample_output_allocator,
                                                                    center_localid_allocator,
                                                                    sample_offset,
                                                                    wm_csr_row_ptr,
                                                                    wm_csr_col_ptr,
                                                                    center_nodes,
                                                                    center_node_count,
                                                                    max_sample_count,
                                                                    cuda_env_fns,
                                                                    stream);
}

REGISTER_DISPATCH_ONE_TYPE(UnweightedSampleWithoutReplacement, UnweightedSampleWithoutReplacement, SINT3264)

void WmmpUnweightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                            const std::function<void *(size_t)> &center_localid_allocator,
                                            int *sample_offset,
                                            void *wm_csr_row_ptr,
                                            void *wm_csr_col_ptr,
                                            WMType id_type,
                                            const void *center_nodes,
                                            int center_node_count,
                                            int max_sample_count,
                                            const CUDAEnvFns &cuda_env_fns,
                                            cudaStream_t stream) {
  DISPATCH_ONE_TYPE(id_type,
                    UnweightedSampleWithoutReplacement,
                    sample_output_allocator,
                    center_localid_allocator,
                    sample_offset,
                    wm_csr_row_ptr,
                    wm_csr_col_ptr,
                    center_nodes,
                    center_node_count,
                    max_sample_count,
                    cuda_env_fns,
                    stream);
}

template<typename IdType>
void ChunkedUnweightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                               const std::function<void *(size_t)> &center_localid_allocator,
                                               int *sample_offset,
                                               void *wm_csr_row_ptr,
                                               void *wm_csr_col_ptr,
                                               const void *center_nodes,
                                               int center_node_count,
                                               int max_sample_count,
                                               const CUDAEnvFns &cuda_env_fns,
                                               cudaStream_t stream) {
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle *wm_csr_row_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_row_ptr, dev_id);
  WholeChunkedMemoryHandle *wm_csr_col_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_col_ptr, dev_id);
  UnweightedSampleWithoutReplacementCommon<IdType,
                                           const whole_graph::WholeChunkedMemoryHandle,
                                           const whole_graph::WholeChunkedMemoryHandle>(
      sample_output_allocator,
      center_localid_allocator,
      sample_offset,
      wm_csr_row_handle,
      wm_csr_col_handle,
      center_nodes,
      center_node_count,
      max_sample_count,
      cuda_env_fns,
      stream);
}

REGISTER_DISPATCH_ONE_TYPE(ChunkedUnweightedSampleWithoutReplacement,
                           ChunkedUnweightedSampleWithoutReplacement,
                           SINT3264)

void WmmpChunkedUnweightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                                   const std::function<void *(size_t)> &center_localid_allocator,
                                                   int *sample_offset,
                                                   void *wm_csr_row_ptr,
                                                   void *wm_csr_col_ptr,
                                                   WMType id_type,
                                                   const void *center_nodes,
                                                   int center_node_count,
                                                   int max_sample_count,
                                                   const CUDAEnvFns &cuda_env_fns,
                                                   cudaStream_t stream) {
  DISPATCH_ONE_TYPE(id_type,
                    ChunkedUnweightedSampleWithoutReplacement,
                    sample_output_allocator,
                    center_localid_allocator,
                    sample_offset,
                    wm_csr_row_ptr,
                    wm_csr_col_ptr,
                    center_nodes,
                    center_node_count,
                    max_sample_count,
                    cuda_env_fns,
                    stream);
}

template<typename KeyT, int BucketSize>
class AppendUniqueHash;

template<typename KeyT, int BucketSize, bool IsTarget = true>
__global__ void InsertKeysKernel(AppendUniqueHash<KeyT, BucketSize> auh);

template<typename KeyT, int BucketSize, bool IsTarget = false, bool NeedValueID = true>
__global__ void RetrieveKeysKernel(AppendUniqueHash<KeyT, BucketSize> auh, int *output);

struct HashTempMemory {
  TempMemoryHandle table_keys_tm;
  TempMemoryHandle value_id_tm;
  TempMemoryHandle value_count_tm;
};

template<typename T>
__device__ __forceinline__ T atomicCASSigned(T *ptr, T cmp, T val) {
  return atomicCAS(ptr, cmp, val);
}

template<>
__device__ __forceinline__ int64_t atomicCASSigned<int64_t>(int64_t *ptr, int64_t cmp, int64_t val) {
  return (int64_t) atomicCAS((unsigned long long *) ptr, cmp, val);
}

static constexpr int kAssignBucketSize = 32;
static constexpr int kAssignThreadBlockSize = 8 * 32;
template<typename KeyT, int BucketSize = kAssignBucketSize / sizeof(KeyT)>
class AppendUniqueHash {
 public:
  AppendUniqueHash(int target_count, int neighbor_count, const KeyT *targets, const KeyT *neighbors)
      : target_count_(target_count), neighbor_count_(neighbor_count), targets_(targets), neighbors_(neighbors) {
    int total_slots_needed = (target_count + neighbor_count) * 2;
    total_slots_needed = AlignUp(total_slots_needed, kAssignBucketSize);
    bucket_count_ = DivUp(total_slots_needed, BucketSize) + 1;
  }
  ~AppendUniqueHash() {
  }
  void AllocateMemoryAndInit(const CUDAEnvFns &fns, HashTempMemory &htm, cudaStream_t stream) {
    // compute bucket_count_ and allocate memory.
    size_t total_alloc_slots = AlignUp(bucket_count_ * BucketSize, kAssignThreadBlockSize);
    fns.allocate_temp_fn(total_alloc_slots * sizeof(KeyT), &htm.table_keys_tm);
    fns.allocate_temp_fn(total_alloc_slots * sizeof(int), &htm.value_id_tm);
    fns.allocate_temp_fn(total_alloc_slots * sizeof(int), &htm.value_count_tm);
    table_keys_ = (KeyT *) htm.table_keys_tm.ptr;
    value_id_ = (int *) htm.value_id_tm.ptr;
    value_count_ = (int *) htm.value_count_tm.ptr;

    // init key to -1
    WM_CUDA_CHECK(cudaMemsetAsync(table_keys_, -1, total_alloc_slots * sizeof(KeyT), stream));
    // init value_id to -1
    WM_CUDA_CHECK(cudaMemsetAsync(value_id_, -1, total_alloc_slots * sizeof(int), stream));
    // init value_count to 0
    WM_CUDA_CHECK(cudaMemsetAsync(value_count_, 0, total_alloc_slots * sizeof(int), stream));
  }
  void DeAllocateMemory(const CUDAEnvFns &fns, HashTempMemory &htm) {
    // deallocate memory.
    fns.free_temp_fn(&htm.table_keys_tm);
    fns.free_temp_fn(&htm.value_id_tm);
    fns.free_temp_fn(&htm.value_count_tm);
  }
  void InsertKeys(cudaStream_t stream) {
    const int thread_count = 512;
    int target_block_count = DivUp(target_count_ * BucketSize, thread_count);
    InsertKeysKernel<KeyT, BucketSize, true><<<target_block_count, thread_count, 0, stream>>>(*this);
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    int neighbor_block_count = DivUp(neighbor_count_ * BucketSize, thread_count);
    InsertKeysKernel<KeyT, BucketSize, false><<<neighbor_block_count, thread_count, 0, stream>>>(*this);
  }
  void RetrieveNeighborKeysForValueIDs(cudaStream_t stream, int *value_ids) {
    const int thread_count = 512;
    int target_block_count = DivUp(neighbor_count_ * BucketSize, thread_count);
    RetrieveKeysKernel<KeyT, BucketSize><<<target_block_count, thread_count, 0, stream>>>(*this, value_ids);
  }
  __host__ __device__ __forceinline__ int TargetCount() { return target_count_; }
  __host__ __device__ __forceinline__ int NeighborCount() { return neighbor_count_; }
  __host__ __device__ __forceinline__ const KeyT *Targets() { return targets_; }
  __host__ __device__ __forceinline__ const KeyT *Neighbors() { return neighbors_; }
  __host__ __device__ __forceinline__ KeyT *TableKeys() { return table_keys_; }
  __host__ __device__ __forceinline__ int32_t *ValueID() { return value_id_; }
  __host__ __device__ __forceinline__ int32_t *ValueCount() { return value_count_; }
  size_t SlotCount() {
    return bucket_count_ * BucketSize;
  }
  void GetBucketLayout(int *bucket_count, int *bucket_size) {
    *bucket_count = bucket_count_;
    *bucket_size = BucketSize;
  }
  static constexpr KeyT kInvalidKey = -1LL;
  static constexpr int kInvalidValueID = -1;
  static constexpr int kNeedAssignValueID = -2;

  __device__ __forceinline__ int retrieve_key(const KeyT &key,
                                              cooperative_groups::thread_block_tile<BucketSize> &group) {
    // On find, return global slot offset
    // On not find, return new slot and set key. Not find and don't need new slot case should not happen.
    int base_bucket_id = bucket_for_key(key);
    int bucket_id;
    int local_slot_offset = -1;
    int try_idx = 0;
    do {
      bucket_id = bucket_id_on_conflict(base_bucket_id, try_idx);
      local_slot_offset = key_in_bucket(key, bucket_id, group);
      try_idx++;
    } while (local_slot_offset < 0);
    return bucket_id * BucketSize + local_slot_offset;
  }
  __device__ __forceinline__ void insert_key(const KeyT &key,
                                             const int id,
                                             cooperative_groups::thread_block_tile<BucketSize> &group) {
    int slot_offset = retrieve_key(key, group);
    int *value_id_ptr = value_id_ + slot_offset;
    if (group.thread_rank() == 0) {
      if (id == kNeedAssignValueID) {
        // neighbor
        atomicCAS(value_id_ptr, kInvalidValueID, id);
        int count = atomicAdd(value_count_ + slot_offset, 1);
      } else {
        // target
        *value_id_ptr = id;
      }
    }
  }

 private:
  __device__ __forceinline__ int bucket_for_key(const KeyT &key) {
    const uint32_t
        hash_value = ((uint32_t)((uint64_t) key >> 32ULL)) * 0x85ebca6b + (uint32_t)((uint64_t) key & 0xFFFFFFFFULL);
    return hash_value % bucket_count_;
  }
  __device__ __forceinline__ int bucket_id_on_conflict(int base_bucket_id, int try_idx) {
    return (base_bucket_id + try_idx * try_idx) % bucket_count_;
  }
  __device__ __forceinline__ int key_in_bucket(const KeyT &key,
                                               int bucket_id,
                                               cooperative_groups::thread_block_tile<BucketSize> &group) {
    // On find or inserted(no work thread should not do insertion), return local slot offset.
    // On not find and bucket is full, return -1.
    // Should do CAS loop
    // cooperative_groups::thread_block_tile<BucketSize> g = cooperative_groups::tiled_partition<BucketSize>(cooperative_groups::this_thread_block());
    KeyT *key_ptr = table_keys_ + bucket_id * BucketSize + group.thread_rank();
    KeyT old_key = *key_ptr;
    unsigned int match_key = group.ballot(old_key == key);
    int match_lane_id = __ffs(match_key) - 1;
    if (match_lane_id >= 0) {
      return match_lane_id;
    }
    unsigned int empty_key = group.ballot(old_key == AppendUniqueHash<KeyT>::kInvalidKey);
    while (empty_key != 0) {
      int leader = __ffs((int) empty_key) - 1;
      KeyT old;
      if (group.thread_rank() == leader) {
        old = atomicCASSigned(key_ptr, old_key, key);
      }
      old = group.shfl(old, leader);
      old_key = group.shfl(old_key, leader);
      if (old == old_key || old == key) {
        // success and duplicate.
        return leader;
      }
      empty_key ^= (1UL << (unsigned) leader);
    }
    return -1;
  }

  int bucket_count_;

  KeyT *table_keys_ = nullptr;    // -1 invalid
  int32_t *value_id_ = nullptr;   // -1 invalid, -2 need assign final neighbor id
  int32_t *value_count_ = nullptr;// 0 initialized

  const KeyT *targets_ = nullptr;
  const KeyT *neighbors_ = nullptr;
  int target_count_;
  int neighbor_count_;
};

template<typename KeyT, int BucketSize, bool IsTarget>
__global__ void InsertKeysKernel(AppendUniqueHash<KeyT, BucketSize> auh) {
  int input_key_count = IsTarget ? auh.TargetCount() : auh.NeighborCount();
  const KeyT *input_key_ptr = IsTarget ? auh.Targets() : auh.Neighbors();
  int key_idx = (blockIdx.x * blockDim.x + threadIdx.x) / BucketSize;
  cooperative_groups::thread_block_tile<BucketSize> group =
      cooperative_groups::tiled_partition<BucketSize>(cooperative_groups::this_thread_block());
  if (key_idx >= input_key_count) return;
  KeyT key = input_key_ptr[key_idx];
  int id = IsTarget ? key_idx : AppendUniqueHash<KeyT, BucketSize>::kNeedAssignValueID;
  auh.insert_key(key, id, group);
}

template<typename KeyT, int BucketSize, bool IsTarget, bool NeedValueID>
__global__ void RetrieveKeysKernel(AppendUniqueHash<KeyT, BucketSize> auh, int *output) {
  int input_key_count = IsTarget ? auh.TargetCount() : auh.NeighborCount();
  const KeyT *input_key_ptr = IsTarget ? auh.Targets() : auh.Neighbors();
  const int *output_value_ptr = NeedValueID ? auh.ValueID() : auh.ValueCount();
  int key_idx = (blockIdx.x * blockDim.x + threadIdx.x) / BucketSize;
  cooperative_groups::thread_block_tile<BucketSize> group =
      cooperative_groups::tiled_partition<BucketSize>(cooperative_groups::this_thread_block());
  if (key_idx >= input_key_count) return;
  KeyT key = input_key_ptr[key_idx];
  int offset = auh.retrieve_key(key, group);
  if (group.thread_rank() == 0) {
    output[key_idx] = output_value_ptr[offset];
  }
}

template<typename KeyT>
__global__ void CountBucketKernel(const int *value_id, int *bucket_count_ptr) {
  __shared__ int count_buffer[kAssignThreadBlockSize / kAssignBucketSize];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int value = value_id[idx];
  unsigned int assign_mask = __ballot_sync(0xffffffff, value == AppendUniqueHash<KeyT>::kNeedAssignValueID);
  if (threadIdx.x % 32 == 0) {
    int assign_count = __popc((int) assign_mask);
    count_buffer[threadIdx.x / 32] = assign_count;
  }
  __syncthreads();
  if (threadIdx.x < kAssignThreadBlockSize / kAssignBucketSize) {
    bucket_count_ptr[kAssignThreadBlockSize / kAssignBucketSize * blockIdx.x + threadIdx.x] = count_buffer[threadIdx.x];
  }
}

template<typename KeyT>
__global__ void AssignValueKernel(int *value_id,
                                  const int *value_counts,
                                  const int *bucket_prefix_sum_ptr,
                                  int target_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_start = bucket_prefix_sum_ptr[idx / 32];
  int value = value_id[idx];
  //int value_count = value_counts[idx];
  unsigned int thread_mask = (1UL << (threadIdx.x % 32)) - 1;
  unsigned int assign_mask = __ballot_sync(0xffffffff, value == AppendUniqueHash<KeyT>::kNeedAssignValueID);
  assign_mask &= thread_mask;
  int idx_in_warp = __popc((int) assign_mask);
  int assigned_idx = idx_in_warp + warp_start;
  //printf("##Raw value_id[%d]=%d, value_count[%d]=%d\n", idx, value, idx, value_count);
  if (/*value_count > 0 && */ value == AppendUniqueHash<KeyT>::kNeedAssignValueID) {
    value_id[idx] = assigned_idx + target_count;
  }
}

template<typename KeyT>
__global__ void ComputeOutputUniqueNeighborAndCountKernel(
    const KeyT *table_keys, const int *value_ids, const int *value_counts, int target_count,
    KeyT *unique_total_output, int *unique_output_neighbor_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  KeyT key = table_keys[idx];
  int value_id = value_ids[idx];
  int value_count = value_counts[idx];
  if (value_id >= target_count) {
    unique_total_output[value_id] = key;
  }
  if (value_id >= 0) {
    unique_output_neighbor_count[value_id] = value_count;
  }
}

template<typename KeyT>
void AppendUniqueCommon(const void *target,
                        int target_count,
                        const void *neighbor,
                        int neighbor_count,
                        const std::function<void *(size_t)> &unique_total_output_allocator,
                        const std::function<int32_t *(size_t)> &neighbor_raw_to_unique_mapping_allocator,
                        const std::function<int32_t *(size_t)> &unique_output_neighbor_count_allocator,
                        const CUDAEnvFns &fns,
                        cudaStream_t stream) {
  AppendUniqueHash<KeyT> auh(target_count, neighbor_count, (const KeyT *) target, (const KeyT *) neighbor);
  HashTempMemory htm;
  auh.AllocateMemoryAndInit(fns, htm, stream);
  auh.InsertKeys(stream);

  TempMemoryHandle bucket_count_tm, bucket_prefix_sum_tm;
  fns.allocate_temp_fn((DivUp(auh.SlotCount(), kAssignBucketSize) + 1) * sizeof(int), &bucket_count_tm);
  fns.allocate_temp_fn((DivUp(auh.SlotCount(), kAssignBucketSize) + 1) * sizeof(int), &bucket_prefix_sum_tm);
  int *bucket_count_ptr = (int *) bucket_count_tm.ptr;
  int *bucket_prefix_sum_ptr = (int *) bucket_prefix_sum_tm.ptr;

  KeyT *table_keys = auh.TableKeys();
  int *value_id = auh.ValueID();
  int *value_count = auh.ValueCount();

  CountBucketKernel<KeyT><<<DivUp(auh.SlotCount(), kAssignThreadBlockSize), kAssignThreadBlockSize, 0, stream>>>(
      value_id,
      bucket_count_ptr);

  WMThrustAllocator allocator(fns);
  thrust::exclusive_scan(thrust::cuda::par(allocator).on(stream),
                         bucket_count_ptr,
                         bucket_count_ptr + DivUp(auh.SlotCount(), kAssignBucketSize) + 1,
                         bucket_prefix_sum_ptr);
  //ExclusiveSum(bucket_prefix_sum_ptr, bucket_count_ptr, DIV_UP(auh.SlotCount(), kAssignBucketSize) + 1, ops, stream);
  int unique_neighbor_count = 0;
  WM_CUDA_CHECK(cudaMemcpyAsync(&unique_neighbor_count,
                                bucket_prefix_sum_ptr + DivUp(auh.SlotCount(), kAssignBucketSize),
                                sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));

  CUDA_STREAM_SYNC(fns, stream);

  AssignValueKernel<KeyT><<<DivUp(auh.SlotCount(), kAssignThreadBlockSize), kAssignThreadBlockSize, 0, stream>>>(value_id, value_count, bucket_prefix_sum_ptr, target_count);

  allocator.deallocate_all();

  KeyT *unique_total_output = (KeyT *) unique_total_output_allocator(target_count + unique_neighbor_count);
  int32_t *neighbor_raw_to_unique_mapping = neighbor_raw_to_unique_mapping_allocator(neighbor_count);
  int32_t *unique_output_neighbor_count = unique_output_neighbor_count_allocator(target_count + unique_neighbor_count);

  WM_CUDA_CHECK(cudaMemcpyAsync(unique_total_output,
                                target,
                                target_count * sizeof(KeyT),
                                cudaMemcpyDeviceToDevice,
                                stream));
  // scan hash table for neighbors in unique_total_output
  // scan hash table for unique_output_neighbor_count
  ComputeOutputUniqueNeighborAndCountKernel<KeyT>
      <<<DivUp(auh.SlotCount(), kAssignThreadBlockSize), kAssignThreadBlockSize, 0, stream>>>(table_keys, value_id, value_count, target_count, unique_total_output, unique_output_neighbor_count);

  // scan neighbors array and use hash table to generate neighbor_raw_to_unique_mapping
  auh.RetrieveNeighborKeysForValueIDs(stream, neighbor_raw_to_unique_mapping);
  CUDA_STREAM_SYNC(fns, stream);

  fns.free_temp_fn(&bucket_count_tm);
  fns.free_temp_fn(&bucket_prefix_sum_tm);
  auh.DeAllocateMemory(fns, htm);
}

REGISTER_DISPATCH_ONE_TYPE(AppendUniqueCommon, AppendUniqueCommon, SINT3264)

void AppendUnique(const void *target,
                  int target_count,
                  const void *neighbor,
                  int neighbor_count,
                  WMType id_type,
                  const std::function<void *(size_t)> &unique_total_output_allocator,
                  const std::function<int32_t *(size_t)> &neighbor_raw_to_unique_mapping_allocator,
                  const std::function<int32_t *(size_t)> &unique_output_neighbor_count_allocator,
                  const CUDAEnvFns &cuda_env_fns,
                  cudaStream_t stream) {
  DISPATCH_ONE_TYPE(id_type,
                    AppendUniqueCommon,
                    target,
                    target_count,
                    neighbor,
                    neighbor_count,
                    unique_total_output_allocator,
                    neighbor_raw_to_unique_mapping_allocator,
                    unique_output_neighbor_count_allocator,
                    cuda_env_fns,
                    stream);
}

static constexpr int kEdgeHashSetBucketSize = 8;
static constexpr double kEdgeHashSetLoadFactor = 0.5;

template<typename EdgeIDType>
class EdgeHashSet;

template<typename EdgeIDType>
__global__ void InsertEdgesKernel(EdgeHashSet<EdgeIDType> ehs,
                                  const EdgeIDType *src_ids,
                                  const EdgeIDType *dst_ids,
                                  int edge_count);

template<typename EdgeIDType>
__global__ void RetrieveCOOEdgesKernel(EdgeHashSet<EdgeIDType> ehs,
                                       const EdgeIDType *src_ids,
                                       const EdgeIDType *dst_ids,
                                       int edge_count,
                                       int *output);

template<typename T>
__device__ __host__ __forceinline__ T AcquireLoadGlobal(const T *ptr) {
  abort();
  return *ptr;
}

template<>
__device__ __host__ __forceinline__ int AcquireLoadGlobal(const int *ptr) {
  int value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [ %1 ];"
               : "=r"(value)
               : "l"(ptr)
               : "memory");
  return value;
}

template<>
__device__ __host__ __forceinline__ int64_t AcquireLoadGlobal(const int64_t *ptr) {
  int64_t value;
  asm volatile("ld.acquire.gpu.global.u64 %0, [ %1 ];"
               : "=l"(value)
               : "l"(ptr)
               : "memory");
  return value;
}

template<typename T>
__device__ __host__ __forceinline__ void ReleaseStoreGlobal(T *ptr, T value) {
  abort();
  return;
}

template<>
__device__ __host__ __forceinline__ void ReleaseStoreGlobal(int *ptr, int value) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;"
               :
               : "l"(ptr), "r"(value)
               : "memory");
}

template<>
__device__ __host__ __forceinline__ void ReleaseStoreGlobal(int64_t *ptr, int64_t value) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;"
               :
               : "l"(ptr), "l"(value)
               : "memory");
}

template<typename EdgeIDType>
class EdgeHashSet {
 public:
  EdgeHashSet(int mem_elt_count, void *src_ptr, void *dst_ptr) {
    WM_CHECK(mem_elt_count % (2 * kEdgeHashSetBucketSize) == 0);
    bucket_count_ = mem_elt_count / (2 * kEdgeHashSetBucketSize);
    keys_mem_ptr_ = (EdgeIDType *) src_ptr;
    values_mem_ptr_ = (EdgeIDType *) dst_ptr;
  }
  ~EdgeHashSet() = default;
  void InitMemory(cudaStream_t stream) {
    WM_CUDA_CHECK(cudaMemsetAsync(keys_mem_ptr_,
                                  -1,
                                  bucket_count_ * kEdgeHashSetBucketSize * sizeof(EdgeIDType),
                                  stream));
    WM_CUDA_CHECK(cudaMemsetAsync(values_mem_ptr_,
                                  -1,
                                  bucket_count_ * kEdgeHashSetBucketSize * sizeof(EdgeIDType),
                                  stream));
  }
  void InsertEdges(const EdgeIDType *src_ids, const EdgeIDType *dst_ids, int edge_count, cudaStream_t stream) {
    int block_size = 32;
    int edges_per_block = block_size / kEdgeHashSetBucketSize;
    int block_count = DivUp(edge_count, edges_per_block);
    InsertEdgesKernel<EdgeIDType><<<block_count, block_size, 0, stream>>>(*this, src_ids, dst_ids, edge_count);
  }
  void RetriveCOOEdges(const EdgeIDType *src_ids,
                       const EdgeIDType *dst_ids,
                       int edge_count,
                       int *output,
                       cudaStream_t stream) {
    int block_size = 32;
    int edges_per_block = block_size / kEdgeHashSetBucketSize;
    int block_count = DivUp(edge_count, edges_per_block);
    RetrieveCOOEdgesKernel<EdgeIDType><<<block_count, block_size, 0, stream>>>(*this,
                                                                               src_ids,
                                                                               dst_ids,
                                                                               edge_count,
                                                                               output);
  }
  static constexpr EdgeIDType kInvalidKey = -1LL;
  static constexpr EdgeIDType kInvalidValueID = -1LL;
  __device__ __forceinline__ int insert_edge(const EdgeIDType &src,
                                             const EdgeIDType &dst,
                                             cooperative_groups::thread_block_tile<kEdgeHashSetBucketSize> &group) {
    // return position of the insert edge.
    int base_bucket_id = bucket_for_key(src);
    int bucket_id;
    int local_slot_offset = -1;
    int try_idx = 0;
    do {
      bucket_id = bucket_id_on_conflict(base_bucket_id, try_idx, src);
      local_slot_offset = try_insert_key_value_to_bucket(src, dst, bucket_id, group);
      try_idx++;
    } while (local_slot_offset < 0);
    return bucket_id * kEdgeHashSetBucketSize + local_slot_offset;
  }
  __device__ __forceinline__ bool has_edge_on_built_hash(const EdgeIDType &src,
                                                         const EdgeIDType &dst,
                                                         cooperative_groups::thread_block_tile<kEdgeHashSetBucketSize> &group) {
    int base_bucket_id = bucket_for_key(src);
    int bucket_id;
    int local_slot_offset = -1;
    int try_idx = 0;
    do {
      bucket_id = bucket_id_on_conflict(base_bucket_id, try_idx, src);
      local_slot_offset = has_key_value_in_bucket_on_built_hash(src, dst, bucket_id, group);
      if (local_slot_offset >= 0) break;
      try_idx++;
    } while (local_slot_offset == -1);

    return local_slot_offset != -2;
  }

 private:
  __device__ __forceinline__ int bucket_for_key(const EdgeIDType &key) {
    const uint32_t
        hash_value = ((uint32_t)((uint64_t) key >> 32ULL)) * 0x85ebca6b + (uint32_t)((uint64_t) key & 0xFFFFFFFFULL);
    return hash_value % bucket_count_;
  }
  __device__ __forceinline__ int bucket_id_on_conflict(int base_bucket_id, int try_idx, const EdgeIDType &key) {
    int bucket_step = key % (bucket_count_ - 1) + 1;
    return (base_bucket_id + bucket_step * try_idx) % bucket_count_;
  }
  __device__ __forceinline__ int try_insert_key_value_to_bucket(const EdgeIDType &key,
                                                                const EdgeIDType &value,
                                                                int bucket_id,
                                                                cooperative_groups::thread_block_tile<
                                                                    kEdgeHashSetBucketSize> &group) {
    // On find or inserted(no work thread should not do insertion), return local slot offset.
    // On bucket is full, return -1.
    // Should do CAS loop
    // cooperative_groups::thread_block_tile<BucketSize> g = cooperative_groups::tiled_partition<BucketSize>(cooperative_groups::this_thread_block());
    EdgeIDType *key_ptr = keys_mem_ptr_ + bucket_id * kEdgeHashSetBucketSize + group.thread_rank();
    EdgeIDType *value_ptr = values_mem_ptr_ + bucket_id * kEdgeHashSetBucketSize + group.thread_rank();
    EdgeIDType old_key = AcquireLoadGlobal(key_ptr);
    EdgeIDType old_value = AcquireLoadGlobal(value_ptr);
    if (old_key == key) {
      while ((old_value = AcquireLoadGlobal(value_ptr)) == kInvalidValueID) __nanosleep(50);
    }
    group.sync();
    unsigned int match_key_value = group.ballot(old_key == key && old_value == value);
    int match_lane_id = __ffs(match_key_value) - 1;
    if (match_lane_id >= 0) {
      return match_lane_id;
    }
    unsigned int empty_key = group.ballot(old_key == kInvalidKey);
    while (empty_key != 0) {
      int leader = __ffs((int) empty_key) - 1;
      EdgeIDType old;
      if (group.thread_rank() == leader) {
        old = atomicCASSigned(key_ptr, old_key, key);
      }
      old = group.shfl(old, leader);
      old_key = group.shfl(old_key, leader);
      if (old == old_key) {
        // success.
        if (group.thread_rank() == leader) {
          ReleaseStoreGlobal(value_ptr, value);
        }
        return leader;
      } else if (old == key) {
        if (group.thread_rank() == leader) {
          while ((old_value = AcquireLoadGlobal(value_ptr)) == kInvalidValueID) __nanosleep(50);
        }
        group.sync();
        old_value = group.shfl(old_value, leader);
        if (old_value == value) {
          // find duplicated.
          return leader;
        }
      }
      empty_key ^= (1UL << (unsigned) leader);
    }
    return -1;
  }
  __device__ __forceinline__ int has_key_value_in_bucket_on_built_hash(const EdgeIDType &key,
                                                                       const EdgeIDType &value,
                                                                       int bucket_id,
                                                                       cooperative_groups::thread_block_tile<
                                                                           kEdgeHashSetBucketSize> &group) {
    // On find, return local slot offset.
    // On not find and bucket is not full, return -2;
    // On not find and bucket is full, return -1.
    // Should do CAS loop
    // cooperative_groups::thread_block_tile<BucketSize> g = cooperative_groups::tiled_partition<BucketSize>(cooperative_groups::this_thread_block());
    EdgeIDType *key_ptr = keys_mem_ptr_ + bucket_id * kEdgeHashSetBucketSize + group.thread_rank();
    EdgeIDType *value_ptr = values_mem_ptr_ + bucket_id * kEdgeHashSetBucketSize + group.thread_rank();
    EdgeIDType old_key = *key_ptr;
    EdgeIDType old_value;
    if (old_key == key) {
      old_value = AcquireLoadGlobal(value_ptr);
    }
    unsigned int match_key_value = group.ballot(old_key == key && old_value == value);
    int match_lane_id = __ffs(match_key_value) - 1;
    if (match_lane_id >= 0) {
      return match_lane_id;
    }
    unsigned int empty_key = group.ballot(old_key == kInvalidKey);
    if (empty_key == 0) {
      return -1;
    } else {
      return -2;
    }
  }

  int bucket_count_;
  EdgeIDType *keys_mem_ptr_;
  EdgeIDType *values_mem_ptr_;
};

template<typename EdgeIDType>
__global__ void InsertEdgesKernel(EdgeHashSet<EdgeIDType> ehs,
                                  const EdgeIDType *src_ids,
                                  const EdgeIDType *dst_ids,
                                  int edge_count) {
  int key_idx = (blockIdx.x * blockDim.x + threadIdx.x) / kEdgeHashSetBucketSize;
  cooperative_groups::thread_block_tile<kEdgeHashSetBucketSize> group =
      cooperative_groups::tiled_partition<kEdgeHashSetBucketSize>(cooperative_groups::this_thread_block());
  if (key_idx >= edge_count) return;
  ehs.insert_edge(src_ids[key_idx], dst_ids[key_idx], group);
}

template<typename EdgeIDType>
__global__ void RetrieveCOOEdgesKernel(EdgeHashSet<EdgeIDType> ehs,
                                       const EdgeIDType *src_ids,
                                       const EdgeIDType *dst_ids,
                                       int edge_count,
                                       int *output) {
  int key_idx = (blockIdx.x * blockDim.x + threadIdx.x) / kEdgeHashSetBucketSize;
  cooperative_groups::thread_block_tile<kEdgeHashSetBucketSize> group =
      cooperative_groups::tiled_partition<kEdgeHashSetBucketSize>(cooperative_groups::this_thread_block());
  if (key_idx >= edge_count) return;
  bool has_edge = ehs.has_edge_on_built_hash(src_ids[key_idx], dst_ids[key_idx], group);
  if (group.thread_rank() == 0) {
    if (has_edge) {
      output[key_idx] = 1;
    } else {
      output[key_idx] = 0;
    }
  }
}

int64_t GetEdgeHashSetEltCount(int edge_count) {
  int bucket_count = DivUp(ceil((double) edge_count / kEdgeHashSetLoadFactor), kEdgeHashSetBucketSize);
  bucket_count = whole_graph::GetPrimeValue(bucket_count);
  int entry_count = bucket_count * kEdgeHashSetBucketSize;
  return entry_count * 2;
}

template<typename EdgeIDType>
void CreateEdgeHashSetCommon(const void *src_ids,
                             const void *dst_ids,
                             int edge_count,
                             void *hash_set_mem,
                             int hash_set_elt_count,
                             cudaStream_t stream) {
  EdgeHashSet<EdgeIDType> edge_hash_set(hash_set_elt_count, hash_set_mem, (char *) hash_set_mem + hash_set_elt_count / 2 * sizeof(EdgeIDType));
  edge_hash_set.InitMemory(stream);
  edge_hash_set.InsertEdges((EdgeIDType *) src_ids, (EdgeIDType *) dst_ids, edge_count, stream);
}

REGISTER_DISPATCH_ONE_TYPE(CreateEdgeHashSetCommon, CreateEdgeHashSetCommon, SINT3264)

void CreateEdgeHashSet(WMType id_type,
                       const void *src_ids,
                       const void *dst_ids,
                       int edge_count,
                       void *hash_set_mem,
                       int hash_set_elt_count,
                       cudaStream_t stream) {
  WM_CHECK(hash_set_elt_count >= GetEdgeHashSetEltCount(edge_count));
  DISPATCH_ONE_TYPE(id_type, CreateEdgeHashSetCommon,
                    src_ids,
                    dst_ids,
                    edge_count,
                    hash_set_mem,
                    hash_set_elt_count,
                    stream);
}

template<typename EdgeIDType>
void RetrieveCOOEdgesCommon(const void *src_ids,
                            const void *dst_ids,
                            int edge_count,
                            void *hash_set_mem,
                            int hash_set_elt_count,
                            int *output,
                            cudaStream_t stream) {
  EdgeHashSet<EdgeIDType> edge_hash_set(hash_set_elt_count, hash_set_mem, (char *) hash_set_mem + hash_set_elt_count / 2 * sizeof(EdgeIDType));
  edge_hash_set.RetriveCOOEdges((EdgeIDType *) src_ids, (EdgeIDType *) dst_ids, edge_count, output, stream);
}

REGISTER_DISPATCH_ONE_TYPE(RetrieveCOOEdgesCommon, RetrieveCOOEdgesCommon, SINT3264)

void RetrieveCOOEdges(WMType id_type,
                      const void *src_ids,
                      const void *dst_ids,
                      int edge_count,
                      void *hash_set_mem,
                      int hash_set_elt_count,
                      int *output,
                      cudaStream_t stream) {
  DISPATCH_ONE_TYPE(id_type, RetrieveCOOEdgesCommon,
                    src_ids,
                    dst_ids,
                    edge_count,
                    hash_set_mem,
                    hash_set_elt_count,
                    output,
                    stream);
}

template<typename EdgeIDType>
__global__ void FilterCSREdgesKernel(EdgeHashSet<EdgeIDType> ehs,
                                     const EdgeIDType *src_ids,
                                     const EdgeIDType *dst_ids,
                                     const int *offset,
                                     int node_count,
                                     int edge_count,
                                     int *output,
                                     int *new_count) {
  static constexpr int TileCountPerBlock = 32 / kEdgeHashSetBucketSize;
  assert(blockDim.x == 32);
  __shared__ int count_array[TileCountPerBlock];
  int row_idx = blockIdx.x;
  int col_idx = threadIdx.x / kEdgeHashSetBucketSize;
  cooperative_groups::thread_block_tile<kEdgeHashSetBucketSize> group =
      cooperative_groups::tiled_partition<kEdgeHashSetBucketSize>(cooperative_groups::this_thread_block());
  int start_idx = offset[row_idx];
  int col_count = offset[row_idx + 1] - start_idx;
  int count = 0;
  for (; col_idx < col_count; col_idx += TileCountPerBlock) {
    bool has_edge = ehs.has_edge_on_built_hash(src_ids[row_idx], dst_ids[col_idx + start_idx], group);
    if (group.thread_rank() == 0) {
      output[col_idx + start_idx] = has_edge ? 1 : 0;
    }
    if (!has_edge) count++;
    group.sync();
  }
  if (group.thread_rank() == 0) {
    count_array[threadIdx.x / kEdgeHashSetBucketSize] = count;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    count = 0;
    for (int i = 0; i < TileCountPerBlock; i++) {
      count += count_array[i];
    }
    new_count[row_idx] = count;
  }
}

template<typename EdgeIDType>
__global__ void RemoveFilteredEdgesKernel(const EdgeIDType *src_ids,
                                          const EdgeIDType *dst_ids,
                                          const int *old_offset,
                                          const int *flags,
                                          const int *new_offset,
                                          int node_count,
                                          EdgeIDType *new_dst_ids,
                                          int *new_src_lids) {
  int row_idx = blockIdx.x;
  int tidx = threadIdx.x;
  unsigned tmask = (1U << (unsigned) tidx) - 1;
  int old_col_start = old_offset[row_idx];
  int old_col_count = old_offset[row_idx + 1] - old_col_start;
  int new_col_start = new_offset[row_idx];
  //int new_col_count = new_offset[row_idx + 1] - new_col_start;
  int old_col_idx = tidx;
  int keep_count = 0;
  unsigned warp_keep_mask;
  for (; old_col_idx < AlignUp(old_col_count, 32); old_col_idx += 32) {
    int overall_old_col_idx = old_col_idx + old_col_start;
    bool keep = false;
    if (old_col_idx < old_col_count) {
      keep = flags[overall_old_col_idx] == 0;
    }
    __syncwarp();
    warp_keep_mask = __ballot_sync(0xffffffff, keep);
    auto thread_lower_keep_mask = warp_keep_mask & tmask;
    int new_col_idx = __popc(thread_lower_keep_mask) + keep_count;
    int overall_new_col_idx = new_col_idx + new_col_start;
    keep_count += __popc(warp_keep_mask);
    if (keep) {
      new_dst_ids[overall_new_col_idx] = dst_ids[overall_old_col_idx];
      new_src_lids[overall_new_col_idx] = row_idx;
    }
    __syncwarp();
  }
  assert(new_offset[row_idx + 1] - new_col_start == keep_count);
}

template<typename EdgeIDType>
void FilterCSREdgesCommon(const void *src_ids,
                          int node_count,
                          const int *gids_offset,
                          const void *dst_ids_vdata,
                          int edge_count,
                          void *hash_set_mem,
                          int hash_memory_elt_count,
                          int *new_gids_offset,
                          const std::function<void *(size_t)> &new_dst_ids_allocator,
                          const std::function<int *(size_t)> &new_src_lids_allocator,
                          const CUDAEnvFns &env_fns,
                          cudaStream_t stream) {
  TempMemoryHandle tmh_flags;
  int *flags = (int *) env_fns.allocate_temp_fn(sizeof(int) * edge_count, &tmh_flags);
  EdgeHashSet<EdgeIDType> edge_hash_set(hash_memory_elt_count, hash_set_mem, (char *) hash_set_mem + hash_memory_elt_count / 2 * sizeof(EdgeIDType));
  FilterCSREdgesKernel<EdgeIDType><<<node_count, 32, 0, stream>>>(edge_hash_set,
                                                                  (const EdgeIDType *) src_ids,
                                                                  (const EdgeIDType *) dst_ids_vdata,
                                                                  gids_offset,
                                                                  node_count,
                                                                  edge_count,
                                                                  flags,
                                                                  new_gids_offset);
  WM_CUDA_CHECK(cudaGetLastError());
  WMThrustAllocator allocator(env_fns);
  thrust::exclusive_scan(thrust::cuda::par(allocator).on(stream),
                         new_gids_offset,
                         new_gids_offset + node_count + 1,
                         new_gids_offset);
  int new_edge_count;
  WM_CUDA_CHECK(cudaMemcpyAsync(&new_edge_count,
                                new_gids_offset + node_count,
                                sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  EdgeIDType *new_dst_ids = (EdgeIDType *) new_dst_ids_allocator(new_edge_count);
  int *new_src_lids = new_src_lids_allocator(new_edge_count);
  RemoveFilteredEdgesKernel<EdgeIDType><<<node_count, 32, 0, stream>>>((const EdgeIDType *) src_ids,
                                                                       (const EdgeIDType *) dst_ids_vdata,
                                                                       gids_offset,
                                                                       flags,
                                                                       new_gids_offset,
                                                                       node_count,
                                                                       new_dst_ids,
                                                                       new_src_lids);
  allocator.deallocate_all();
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  env_fns.free_temp_fn(&tmh_flags);
}

REGISTER_DISPATCH_ONE_TYPE(FilterCSREdgesCommon, FilterCSREdgesCommon, SINT3264)

void FilterCSREdges(WMType id_type,
                    const void *src_ids,
                    int node_count,
                    const int *gids_offset,
                    void *dst_ids_vdata,
                    int edge_count,
                    void *hash_set_mem_tensor,
                    int hash_memory_elt_count,
                    int *new_gids_offset,
                    const std::function<void *(size_t)> &new_dst_ids_allocator,
                    const std::function<int *(size_t)> &new_src_lids_allocator,
                    const CUDAEnvFns &env_fns,
                    cudaStream_t stream) {
  DISPATCH_ONE_TYPE(id_type,
                    FilterCSREdgesCommon,
                    src_ids,
                    node_count,
                    gids_offset,
                    dst_ids_vdata,
                    edge_count,
                    hash_set_mem_tensor,
                    hash_memory_elt_count,
                    new_gids_offset,
                    new_dst_ids_allocator,
                    new_src_lids_allocator,
                    env_fns,
                    stream);
}

template<typename MixedIDType, typename ToTypeIDHandle>
__global__ void GetCSRMixedSubGraphEdgeTypesKernel(int8_t *output,
                                                   const MixedIDType *src_mixid,
                                                   const int *sub_graph_csr_row_ptr,
                                                   const MixedIDType *sub_graph_csr_col_mixid,
                                                   const int8_t *edge_type_dict,
                                                   const ToTypeIDHandle *to_typed_id,
                                                   int64_t src_node_count,
                                                   int64_t dst_node_count,
                                                   int64_t node_type_count,
                                                   int64_t edge_type_count) {
  int row_idx = blockIdx.x;
  MixedIDType src_mid = src_mixid[row_idx];
  PtrGen<ToTypeIDHandle, int64_t> to_typed_id_gen(to_typed_id);
  int64_t src_typed_id = *to_typed_id_gen.At(src_mid);
  int src_type = TypeOfTypedID(src_typed_id);
  assert(src_type < node_type_count);
  int row_start = sub_graph_csr_row_ptr[row_idx];
  int row_end = sub_graph_csr_row_ptr[row_idx + 1];
  assert(row_start <= dst_node_count && row_end <= dst_node_count);
  for (int col_idx = threadIdx.x; col_idx < row_end - row_start; col_idx += blockDim.x) {
    MixedIDType dst_mid = sub_graph_csr_col_mixid[col_idx + row_start];
    int64_t dst_typed_id = *to_typed_id_gen.At(dst_mid);
    int dst_type = TypeOfTypedID(dst_typed_id);
    assert(dst_type < node_type_count);
    int8_t edge_type = edge_type_dict[src_type * node_type_count + dst_type];
    assert(edge_type < edge_type_count && edge_type >= 0);
    output[col_idx + row_start] = edge_type;
  }
}

template<typename MixedIDType>
void GetCSRMixedSubGraphEdgeTypesFunc(int8_t *output,
                                      const void *src_mixid,
                                      const int *sub_graph_csr_row_ptr,
                                      const void *sub_graph_csr_col_mixid,
                                      const int8_t *edge_type_dict,
                                      const int64_t *to_typed_id,
                                      int64_t src_node_count,
                                      int64_t dst_node_count,
                                      int64_t node_type_count,
                                      int64_t edge_type_count,
                                      cudaStream_t stream) {
  GetCSRMixedSubGraphEdgeTypesKernel<MixedIDType, const int64_t><<<src_node_count, 64, 0, stream>>>(
      output,
      (const MixedIDType *) src_mixid,
      sub_graph_csr_row_ptr,
      (const MixedIDType *) sub_graph_csr_col_mixid,
      edge_type_dict,
      to_typed_id,
      src_node_count,
      dst_node_count,
      node_type_count,
      edge_type_count);
  WM_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH_ONE_TYPE(GetCSRMixedSubGraphEdgeTypesFunc, GetCSRMixedSubGraphEdgeTypesFunc, SINT3264)

void WmmpGetCSRMixedSubGraphEdgeTypes(WMType mixid_type,
                                      int8_t *output,
                                      const void *src_mixid,
                                      const int *sub_graph_csr_row_ptr,
                                      const void *sub_graph_csr_col_mixid,
                                      const int8_t *edge_type_dict,
                                      const int64_t *to_typed_id,
                                      int64_t src_node_count,
                                      int64_t dst_node_count,
                                      int64_t node_type_count,
                                      int64_t edge_type_count,
                                      cudaStream_t stream) {
  DISPATCH_ONE_TYPE(mixid_type, GetCSRMixedSubGraphEdgeTypesFunc,
                    output,
                    src_mixid,
                    sub_graph_csr_row_ptr,
                    sub_graph_csr_col_mixid,
                    edge_type_dict,
                    to_typed_id,
                    src_node_count,
                    dst_node_count,
                    node_type_count,
                    edge_type_count,
                    stream);
}

template<typename MixedIDType>
void GetCSRMixedSubGraphEdgeTypesChunkedFunc(int8_t *output,
                                             const void *src_mixid,
                                             const int *sub_graph_csr_row_ptr,
                                             const void *sub_graph_csr_col_mixid,
                                             const int8_t *edge_type_dict,
                                             WholeChunkedMemory_t to_typed_id_wcmt,
                                             int64_t src_node_count,
                                             int64_t dst_node_count,
                                             int64_t node_type_count,
                                             int64_t edge_type_count,
                                             cudaStream_t stream) {
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle *to_typed_id_handle = GetDeviceChunkedHandle(to_typed_id_wcmt, dev_id);
  GetCSRMixedSubGraphEdgeTypesKernel<MixedIDType, const WholeChunkedMemoryHandle><<<src_node_count, 64, 0, stream>>>(
      output,
      (const MixedIDType *) src_mixid,
      sub_graph_csr_row_ptr,
      (const MixedIDType *) sub_graph_csr_col_mixid,
      edge_type_dict,
      to_typed_id_handle,
      src_node_count,
      dst_node_count,
      node_type_count,
      edge_type_count);
  WM_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH_ONE_TYPE(GetCSRMixedSubGraphEdgeTypesChunkedFunc, GetCSRMixedSubGraphEdgeTypesChunkedFunc, SINT3264)

void WmmpGetCSRMixedSubGraphEdgeTypesChunked(WMType mixid_type,
                                             int8_t *output,
                                             const void *src_mixid,
                                             const int *sub_graph_csr_row_ptr,
                                             const void *sub_graph_csr_col_mixid,
                                             const int8_t *edge_type_dict,
                                             WholeChunkedMemory_t to_typed_id_wcmt,
                                             int64_t src_node_count,
                                             int64_t dst_node_count,
                                             int64_t node_type_count,
                                             int64_t edge_type_count,
                                             cudaStream_t stream) {
  DISPATCH_ONE_TYPE(mixid_type, GetCSRMixedSubGraphEdgeTypesChunkedFunc,
                    output,
                    src_mixid,
                    sub_graph_csr_row_ptr,
                    sub_graph_csr_col_mixid,
                    edge_type_dict,
                    to_typed_id_wcmt,
                    src_node_count,
                    dst_node_count,
                    node_type_count,
                    edge_type_count,
                    stream);
}

__global__ void GetBucketedCSRFromSortedTypedIDsKernel(int *output_csr,
                                                       const int64_t *typed_ids,
                                                       int id_count,
                                                       int node_type_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= id_count) return;
  int last_type = -1;
  if (idx > 0) {
    int64_t last_typed_id = typed_ids[idx - 1];
    last_type = TypeOfTypedID(last_typed_id);
    assert(last_type >= 0 && last_type < node_type_count);
  }
  int current_type = TypeOfTypedID(typed_ids[idx]);
  assert(current_type >= 0 && current_type < node_type_count);
  for (int type = last_type + 1; type <= current_type; type++) {
    output_csr[type] = idx;
  }
  if (idx == id_count - 1) {
    output_csr[node_type_count] = id_count;
  }
}

void WmmpGetBucketedCSRFromSortedTypedIDs(int *output_csr,
                                          const int64_t *typed_ids,
                                          int64_t id_count,
                                          int64_t node_type_count,
                                          cudaStream_t stream) {
  int block_count = DivUp(id_count, 32);
  GetBucketedCSRFromSortedTypedIDsKernel<<<block_count, 32, 0, stream>>>(
      output_csr, typed_ids, id_count, node_type_count);
}

__global__ void PackToTypedIDsKernel(int64_t *output,
                                     const int8_t *type_ids,
                                     const int *ids,
                                     int64_t id_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= id_count) return;
  output[idx] = MakeTypedID(type_ids[idx], ids[idx]);
}

void WmmpPackToTypedIDs(int64_t *output,
                        const int8_t *type_ids,
                        const int *ids,
                        int64_t id_count,
                        cudaStream_t stream) {
  int block_count = DivUp(id_count, 32);
  PackToTypedIDsKernel<<<block_count, 32, 0, stream>>>(output, type_ids, ids, id_count);
}

__global__ void UnpackTypedIDsKernel(const int64_t *typed_ids,
                                     int8_t *type_ids,
                                     int *ids,
                                     int64_t id_count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= id_count) return;
  int64_t typed_id = typed_ids[idx];
  type_ids[idx] = TypeOfTypedID(typed_id);
  ids[idx] = IDOfTypedID(typed_id);
}

void WmmpUnpackTypedIDs(const int64_t *typed_ids,
                        int8_t *type_ids,
                        int *ids,
                        int64_t id_count,
                        cudaStream_t stream) {
  int block_count = DivUp(id_count, 32);
  UnpackTypedIDsKernel<<<block_count, 32, 0, stream>>>(typed_ids, type_ids, ids, id_count);
}

}// namespace whole_graph
