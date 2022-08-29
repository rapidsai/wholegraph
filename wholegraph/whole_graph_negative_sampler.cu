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

namespace whole_graph {

template<typename IdType, typename WMIdType, typename WMOffsetType>
__global__ void PerNodeUniformNegativeSampleSimpleKernel(IdType *output,
                                                         const IdType *input_nodes,
                                                         int input_node_count,
                                                         int graph_dst_node_count,
                                                         int negative_sample_count,
                                                         WMOffsetType *wm_csr_row_ptr,
                                                         WMIdType *wm_csr_col_ptr,
                                                         unsigned long long random_seed,
                                                         int num_try_loop = 10) {

  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  int input_idx = gidx;
  if (input_idx >= input_node_count)
    return;
  RandomNumGen rng(gidx, random_seed);
  rng.NextValue();
  whole_graph::PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  whole_graph::PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  IdType nid = input_nodes[input_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);

  for (int negative_id = 0; negative_id < negative_sample_count; negative_id++) {
    IdType r;
    int num_try = num_try_loop;
    while (num_try--) {
      bool positive_edge = false;
      if (sizeof(IdType) == 4) {
        r = (uint32_t) rng.RandomMod(graph_dst_node_count);
      } else if (sizeof(IdType) == 8) {
        r = (uint64_t) rng.RandomMod64(graph_dst_node_count);
      }

      for (int i = 0; i < neighbor_count; i++) {
        IdType global_neighbor_id = *csr_col_ptr_gen.At(start + i);
        if (global_neighbor_id == r) {
          positive_edge = true;
          break;
        }
      }
      if (!positive_edge) {
        break;
      }
    }
    output[input_idx * negative_sample_count + negative_id] = r;
  }
}

template<typename IdType, typename WMIdType, typename WMOffsetType, int LoopRandNum = 16>
__global__ void PerNodeUniformNegativeSampleSmallNodeKernel(IdType *output,
                                                            const IdType *input_nodes,
                                                            int input_node_count,
                                                            int graph_dst_node_count,
                                                            int negative_sample_count,
                                                            WMOffsetType *wm_csr_row_ptr,
                                                            WMIdType *wm_csr_col_ptr,
                                                            unsigned long long random_seed,
                                                            int num_try_loop = 10) {
  static const int warp_size = 32;
  static const int LoopEdgeNum = warp_size - LoopRandNum;
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  int input_idx = blockIdx.x;
  int lane_id = threadIdx.x % warp_size;
  if (input_idx >= input_node_count)
    return;
  RandomNumGen rng(gidx, random_seed);
  rng.NextValue();
  whole_graph::PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  whole_graph::PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  IdType nid = input_nodes[input_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);
  for (int negative_id = 0; negative_id < negative_sample_count; negative_id++) {
    IdType r;
    unsigned int mask;
    int num_try = num_try_loop;
    while (num_try--) {
      bool positive_edge = false;
      if (lane_id < LoopRandNum) {
        positive_edge = true;
      }
      if (sizeof(IdType) == sizeof(uint32_t)) {
        r = (uint32_t) rng.RandomMod(graph_dst_node_count);
      } else if (sizeof(IdType) == sizeof(uint64_t)) {
        r = (uint64_t) rng.RandomMod64(graph_dst_node_count);
      }

      if (r == nid) {
        positive_edge = true;
      }

      //post-half warp thread random num compared to pre-half thread value from graph neighbor nodes
      for (int i = 0; i < neighbor_count; i += LoopEdgeNum) {
        int global_neighbor_id_index = start + i + lane_id;
        IdType global_neighbor_id = (global_neighbor_id_index < end && lane_id < LoopEdgeNum) ? (*csr_col_ptr_gen.At(global_neighbor_id_index)) : (IdType)(-1);
        IdType value = (lane_id >= LoopRandNum) ? r : global_neighbor_id;
        __syncwarp();
        unsigned int return_mask = __match_any_sync(0xffffffff, value);
        return_mask <<= LoopRandNum;
        if (return_mask != 0) {
          positive_edge = true;
        }
      }
      __syncwarp();
      mask = __ballot_sync(0xffffffff, !positive_edge);
      if (mask) {
        break;
      }
    }
    if (!mask) {
      if (lane_id == 0) {
        output[input_idx * negative_sample_count + negative_id] = r;
      }
      continue;
    }
    int leader = __ffs(mask) - 1;
    if (lane_id == leader) {
      output[input_idx * negative_sample_count + negative_id] = r;
    }
  }
}

template<typename IdType, typename WMIdType, typename WMOffsetType, int LoopRandNum = 16>
__global__ void PerNodeUniformNegativeSampleSmallNodeOptKernel(IdType *output,
                                                               const IdType *input_nodes,
                                                               int input_node_count,
                                                               int graph_dst_node_count,
                                                               int negative_sample_count,
                                                               WMOffsetType *wm_csr_row_ptr,
                                                               WMIdType *wm_csr_col_ptr,
                                                               unsigned long long random_seed,
                                                               int num_try_loop = 10) {
  static const int warp_size = 32;
  static const int LoopEdgeNum = warp_size - LoopRandNum;
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  int input_idx = blockIdx.x;
  int lane_id = threadIdx.x % warp_size;
  if (input_idx >= input_node_count)
    return;
  RandomNumGen rng(gidx, random_seed);
  rng.NextValue();
  whole_graph::PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  whole_graph::PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  IdType nid = input_nodes[input_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);
  for (int negative_id = 0; negative_id < negative_sample_count;) {
    IdType r;
    unsigned int mask;
    int num_try = num_try_loop;
    while (num_try--) {
      bool positive_edge = false;
      if (lane_id < LoopRandNum) {
        positive_edge = true;
      }
      if (sizeof(IdType) == sizeof(uint32_t)) {
        r = (uint32_t) rng.RandomMod(graph_dst_node_count);
      } else if (sizeof(IdType) == sizeof(uint64_t)) {
        r = (uint64_t) rng.RandomMod64(graph_dst_node_count);
      }

      if (r == nid) {
        positive_edge = true;
      }

      // post-half warp thread random num compared to pre-half thread value from graph neighbor nodes
      for (int i = 0; i < neighbor_count; i += LoopEdgeNum) {
        int global_neighbor_id_index = start + i + lane_id;
        IdType global_neighbor_id =
            (global_neighbor_id_index < end && lane_id < LoopEdgeNum) ? (*csr_col_ptr_gen.At(global_neighbor_id_index))
                                                                      : (IdType)(-1);
        IdType value = (lane_id >= LoopRandNum) ? r : global_neighbor_id;
        __syncwarp();
        unsigned int return_mask = __match_any_sync(0xffffffff, value);
        return_mask <<= LoopRandNum;
        if (return_mask != 0) {
          positive_edge = true;
        }
      }
      __syncwarp();
      mask = __ballot_sync(0xffffffff, !positive_edge);
      if (mask) {
        break;
      }
    }
    int leave_count = negative_sample_count - negative_id;
    if (!mask) {
      if (leave_count > warp_size) {
        output[input_idx * negative_sample_count + negative_id + lane_id] = r;
        negative_id += warp_size;
        continue;
      } else {
        if (lane_id < leave_count) {
          output[input_idx * negative_sample_count + negative_id + lane_id] = r;
          return;
        }
      }
    }

    int random_count = __popc(mask);
    if (random_count >= leave_count) {
      int output_local_index = __popc(mask & ((1 << lane_id) - 1));
      if (mask & (1 << lane_id) && output_local_index < leave_count) {
        output[input_idx * negative_sample_count + negative_id + output_local_index] = r;
      }
      return;
    } else {
      int output_local_index = __popc(mask & ((1 << lane_id) - 1));
      if (mask & (1 << lane_id)) {
        output[input_idx * negative_sample_count + negative_id + output_local_index] = r;
      }
      negative_id += random_count;
    }
    //__syncwarp();
  }
}

template<typename IdType, typename WMIdType, typename WMOffsetType>
void PerNodeUniformNegativeSampleComm(const std::function<void *(size_t)> &sample_output_allocator,
                                      void *wm_csr_row_ptr,
                                      void *wm_csr_col_ptr,
                                      const void *input_nodes,
                                      int input_node_count,
                                      int graph_dst_node_count,
                                      int negative_sample_count,
                                      const CUDAEnvFns &cuda_env_fns,
                                      cudaStream_t stream) {
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  thread_local std::uniform_int_distribution<unsigned long long> distrib;
  unsigned long long random_seed = distrib(gen);
  int count = input_node_count * negative_sample_count;
  auto *sample_output = (IdType *) sample_output_allocator(count);
  PerNodeUniformNegativeSampleSmallNodeOptKernel<IdType, WMIdType, WMOffsetType><<<input_node_count, 32, 0, stream>>>(
      sample_output,
      (const IdType *) input_nodes,
      input_node_count,
      graph_dst_node_count,
      negative_sample_count,
      (WMOffsetType *) wm_csr_row_ptr,
      (WMIdType *) wm_csr_col_ptr,
      random_seed);
  WM_CUDA_CHECK(cudaGetLastError());
}

template<typename IdType>
void PerNodeUniformNegativeSample(const std::function<void *(size_t)> &sample_output_allocator,
                                  void *wm_csr_row_ptr,
                                  void *wm_csr_col_ptr,
                                  const void *input_nodes,
                                  int input_node_count,
                                  int graph_dst_node_count,
                                  int negative_sample_count,
                                  const CUDAEnvFns &cuda_env_fns,
                                  cudaStream_t stream) {

  PerNodeUniformNegativeSampleComm<IdType, IdType, int64_t>(sample_output_allocator,
                                                            wm_csr_row_ptr,
                                                            wm_csr_col_ptr,
                                                            input_nodes,
                                                            input_node_count,
                                                            graph_dst_node_count,
                                                            negative_sample_count,
                                                            cuda_env_fns,
                                                            stream);
}

REGISTER_DISPATCH_ONE_TYPE(PerNodeUniformNegativeSample, PerNodeUniformNegativeSample, SINT3264)

void WmmpPerNodeUniformNegativeSample(const std::function<void *(size_t)> &sample_output_allocator,
                                      void *wm_csr_row_ptr,
                                      void *wm_csr_col_ptr,
                                      WMType id_type,
                                      const void *target_nodes,
                                      int target_node_count,
                                      int graph_dst_node_count,
                                      int negative_sample_count,
                                      const CUDAEnvFns &cuda_env_fns,
                                      cudaStream_t stream) {
  DISPATCH_ONE_TYPE(id_type,
                    PerNodeUniformNegativeSample,
                    sample_output_allocator,
                    wm_csr_row_ptr,
                    wm_csr_col_ptr,
                    target_nodes,
                    target_node_count,
                    graph_dst_node_count,
                    negative_sample_count,
                    cuda_env_fns,
                    stream);
}

template<typename IdType>
void ChunkedPerNodeUniformNegativeSample(const std::function<void *(size_t)> &sample_output_allocator,
                                         void *wm_csr_row_ptr,
                                         void *wm_csr_col_ptr,
                                         const void *input_nodes,
                                         int input_node_count,
                                         int graph_dst_node_count,
                                         int negative_sample_count,
                                         const CUDAEnvFns &cuda_env_fns,
                                         cudaStream_t stream) {

  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle *wm_csr_row_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_row_ptr, dev_id);
  WholeChunkedMemoryHandle *wm_csr_col_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_col_ptr, dev_id);
  PerNodeUniformNegativeSampleComm<IdType,
                                   const whole_graph::WholeChunkedMemoryHandle,
                                   const whole_graph::WholeChunkedMemoryHandle>(sample_output_allocator,
                                                                                wm_csr_row_handle,
                                                                                wm_csr_col_handle,
                                                                                input_nodes,
                                                                                input_node_count,
                                                                                graph_dst_node_count,
                                                                                negative_sample_count,
                                                                                cuda_env_fns,
                                                                                stream);
}

REGISTER_DISPATCH_ONE_TYPE(ChunkedPerNodeUniformNegativeSample,
                           ChunkedPerNodeUniformNegativeSample,
                           SINT3264)

void WmmpChunkedPerNodeUniformNegativeSample(const std::function<void *(size_t)> &sample_output_allocator,
                                             void *wm_csr_row_ptr,
                                             void *wm_csr_col_ptr,
                                             WMType id_type,
                                             const void *target_nodes,
                                             int target_node_count,
                                             int graph_dst_node_count,
                                             int negative_sample_count,
                                             const CUDAEnvFns &cuda_env_fns,
                                             cudaStream_t stream) {
  DISPATCH_ONE_TYPE(id_type,
                    ChunkedPerNodeUniformNegativeSample,
                    sample_output_allocator,
                    wm_csr_row_ptr,
                    wm_csr_col_ptr,
                    target_nodes,
                    target_node_count,
                    graph_dst_node_count,
                    negative_sample_count,
                    cuda_env_fns,
                    stream);
}

}// namespace whole_graph
