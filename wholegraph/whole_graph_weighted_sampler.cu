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

#include <cub/cub.cuh>
#include <thrust/scan.h>

#include <random>

#include "block_radix_topk.cuh"
#include "data_type.h"
#include "macros.h"
#include "random.cuh"
#include "whole_chunked_memory.cuh"
#include "whole_graph_mixed_graph.cuh"
#include "whole_memory.h"

namespace whole_graph {

template<typename WeightType>
__device__ __forceinline__ WeightType GenKeyFromWeight(const WeightType weight, RandomNumGen &rng) {
  rng.NextValue();
  float u = -rng.RandomUniformFloat(1.0f, 0.5f);
  long long random_num2 = 0;
  int seed_count = -1;
  do {
    random_num2 = rng.Random64();
    seed_count++;
  } while (!random_num2);
  int one_bit = __clzll(random_num2) + seed_count * 64;
  u *= exp2f(-one_bit);
  WeightType logk = (log1pf(u) / logf(2.0)) * (1 / weight);
  // u = random_uniform(0,1), logk = 1/weight *logf(u)
  return logk;
}

// A-RES algorithmn
// https://en.wikipedia.org/wiki/Reservoir_sampling#Algorithm_A-Res
// max_sample_count should <=(BLOCK_SIZE*ITEMS_PER_THREAD*/4)  otherwise,need to change the template parameters of BlockRadixTopK.
template<typename IdType, typename LocalIdType, typename WeightType,
         typename WMIdType, typename WMOffsetType, typename WMWeightType, typename WMLocalIdType, unsigned int ITEMS_PER_THREAD,
         unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void WeightSampleWithoutReplacementKernel(IdType *output,
                                                                                   LocalIdType *src_lid,
                                                                                   const int *sample_offset,
                                                                                   const IdType *input_nodes,
                                                                                   int input_node_count,
                                                                                   WMOffsetType *wm_csr_row_ptr,
                                                                                   WMIdType *wm_csr_col_ptr,
                                                                                   WMWeightType *wm_csr_weight_ptr,
                                                                                   WMLocalIdType *wm_csr_local_sorted_map_indices_ptr,
                                                                                   int max_sample_count,
                                                                                   unsigned long long random_seed) {
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  PtrGen<WMWeightType, WeightType> csr_weight_ptr_gen(
      wm_csr_weight_ptr);
  PtrGen<WMLocalIdType, LocalIdType> wm_csr_local_sorted_map_indices_ptr_gen(wm_csr_local_sorted_map_indices_ptr);
  IdType nid = input_nodes[input_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);
  int offset = sample_offset[input_idx];
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count;
         sample_id += BLOCK_SIZE) {
      int neighbor_idx = sample_id;
      IdType gid = *csr_col_ptr_gen.At(start + neighbor_idx);
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = (LocalIdType) input_idx;
    }
    return;
  } else {
    RandomNumGen rng(gidx, random_seed);
    // rng.NextValue();
    WeightType weight_keys[ITEMS_PER_THREAD];
    int neighbor_idxs[ITEMS_PER_THREAD];

    using BlockRadixTopKT = BlockRadixTopKRegister<WeightType, BLOCK_SIZE, ITEMS_PER_THREAD, true, int>;

    __shared__ typename BlockRadixTopKT::TempStorage sort_tmp_storage;

    const int tx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int idx = BLOCK_SIZE * i + tx;
      if (idx < neighbor_count) {
        WeightType thread_weight = *(csr_weight_ptr_gen.At(start + idx));
        weight_keys[i] = GenKeyFromWeight(thread_weight, rng);
        neighbor_idxs[i] = idx;
      }
    }
    const int valid_count = (neighbor_count < (BLOCK_SIZE * ITEMS_PER_THREAD)) ? neighbor_count : (BLOCK_SIZE * ITEMS_PER_THREAD);
    BlockRadixTopKT{sort_tmp_storage}.radixTopKToStriped(weight_keys, neighbor_idxs, max_sample_count, valid_count);
    __syncthreads();
    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD - max_sample_count;

    for (int idx_offset = ITEMS_PER_THREAD * BLOCK_SIZE; idx_offset < neighbor_count; idx_offset += stride) {

#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int local_idx = BLOCK_SIZE * i + tx - max_sample_count;
        // [0,BLOCK_SIZE*ITEMS_PER_THREAD-max_sample_count)
        int target_idx = idx_offset + local_idx;
        if (local_idx >= 0 && target_idx < neighbor_count) {
          WeightType thread_weight = *(csr_weight_ptr_gen.At(start + target_idx));
          weight_keys[i] = GenKeyFromWeight(thread_weight, rng);
          neighbor_idxs[i] = target_idx;
        }
      }
      const int iter_valid_count = ((neighbor_count - idx_offset) >= stride) ? (BLOCK_SIZE * ITEMS_PER_THREAD) : (max_sample_count + neighbor_count - idx_offset);
      BlockRadixTopKT{sort_tmp_storage}.radixTopKToStriped(weight_keys, neighbor_idxs, max_sample_count, iter_valid_count);
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int idx = i * BLOCK_SIZE + tx;
      if (idx < max_sample_count) {
        src_lid[offset + idx] = (LocalIdType) input_idx;
        LocalIdType local_original_idx = wm_csr_local_sorted_map_indices_ptr ? (*wm_csr_local_sorted_map_indices_ptr_gen.At(start + neighbor_idxs[i])) : neighbor_idxs[i];
        output[offset + idx] = *csr_col_ptr_gen.At(start + local_original_idx);
      }
    }
  }
}

template<typename IdType, typename LocalIdType, typename WeightType, typename WeightKeyType,
         typename WMIdType, typename WMOffsetType, typename WMWeightType, typename WMLocalIdType,
         unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void WeightSampleWithoutReplacementLargeKernel(IdType *output,
                                                                                        LocalIdType *src_lid,
                                                                                        const int *sample_offset,
                                                                                        const int *target_neighbor_offset,
                                                                                        WeightKeyType *weight_keys_buff,
                                                                                        const IdType *input_nodes,
                                                                                        int input_node_count,
                                                                                        WMOffsetType *wm_csr_row_ptr,
                                                                                        WMIdType *wm_csr_col_ptr,
                                                                                        WMWeightType *wm_csr_weight_ptr,
                                                                                        WMLocalIdType *wm_csr_local_sorted_map_indices_ptr,
                                                                                        int max_sample_count,
                                                                                        unsigned long long random_seed) {

  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  PtrGen<WMOffsetType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
  PtrGen<WMIdType, IdType> csr_col_ptr_gen(wm_csr_col_ptr);
  PtrGen<WMWeightType, WeightType> csr_weight_ptr_gen(
      wm_csr_weight_ptr);
  PtrGen<WMLocalIdType, LocalIdType> wm_csr_local_sorted_map_indices_ptr_gen(wm_csr_local_sorted_map_indices_ptr);
  IdType nid = input_nodes[input_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);

  WeightKeyType *weight_keys_local_buff = weight_keys_buff + target_neighbor_offset[input_idx];
  int offset = sample_offset[input_idx];
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count;
         sample_id += BLOCK_SIZE) {
      int neighbor_idx = sample_id;
      IdType gid = *csr_col_ptr_gen.At(start + neighbor_idx);
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = (LocalIdType) input_idx;
    }
    return;
  }

  RandomNumGen rng(gidx, random_seed);
  for (int id = threadIdx.x; id < neighbor_count; id += BLOCK_SIZE) {
    WeightType thread_weight = *(csr_weight_ptr_gen.At(start + id));
    weight_keys_local_buff[id] = static_cast<WeightKeyType>(GenKeyFromWeight(thread_weight, rng));
  }

  __syncthreads();

  WeightKeyType topk_val;
  bool topk_is_unique;

  using BlockRadixSelectT = BlockRadixTopKGlobalMemory<WeightKeyType, BLOCK_SIZE, true>;
  __shared__ typename BlockRadixSelectT::TempStorage share_storage;

  BlockRadixSelectT{share_storage}.radixTopKGetThreshold(weight_keys_local_buff, max_sample_count, neighbor_count, topk_val, topk_is_unique);
  __shared__ int cnt;

  if (threadIdx.x == 0) {
    cnt = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < max_sample_count; i += BLOCK_SIZE) {
    src_lid[offset + i] = (LocalIdType) input_idx;
  }

  // We use atomicAdd 1 operations instead of binaryScan to calculate the write index,
  // since we do not need to keep the relative positions of element.

  if (topk_is_unique) {
    for (int neighbor_idx = threadIdx.x; neighbor_idx < neighbor_count; neighbor_idx += BLOCK_SIZE) {
      WeightKeyType key = weight_keys_local_buff[neighbor_idx];
      bool has_topk = (key >= topk_val);

      if (has_topk) {
        int write_index = atomicAdd(&cnt, 1);
        LocalIdType local_original_idx = wm_csr_local_sorted_map_indices_ptr ? (*wm_csr_local_sorted_map_indices_ptr_gen.At(start + neighbor_idx)) : neighbor_idx;
        output[offset + write_index] = *csr_col_ptr_gen.At(start + local_original_idx);
      }
    }
  } else {
    for (int neighbor_idx = threadIdx.x; neighbor_idx < neighbor_count; neighbor_idx += BLOCK_SIZE) {
      WeightKeyType key = weight_keys_local_buff[neighbor_idx];
      bool has_topk = (key > topk_val);

      if (has_topk) {
        int write_index = atomicAdd(&cnt, 1);
        LocalIdType local_original_idx = wm_csr_local_sorted_map_indices_ptr ? (*wm_csr_local_sorted_map_indices_ptr_gen.At(start + neighbor_idx)) : neighbor_idx;
        output[offset + write_index] = *csr_col_ptr_gen.At(start + local_original_idx);
      }
    }
    __syncthreads();
    for (int neighbor_idx = threadIdx.x; neighbor_idx < neighbor_count; neighbor_idx += BLOCK_SIZE) {
      WeightKeyType key = weight_keys_local_buff[neighbor_idx];
      bool has_topk = (key == topk_val);

      if (has_topk) {
        int write_index = atomicAdd(&cnt, 1);
        if (write_index >= max_sample_count)
          break;
        LocalIdType local_original_idx = wm_csr_local_sorted_map_indices_ptr ? (*wm_csr_local_sorted_map_indices_ptr_gen.At(start + neighbor_idx)) : neighbor_idx;
        output[offset + write_index] = *csr_col_ptr_gen.At(start + local_original_idx);
      }
    }
  }
}

template<typename IdType, typename WMOffsetType, bool NeedNeighbor = false>
__global__ void GetSampleCountAndNeighborCountWithoutReplacementKernel(int *sample_offset, int *neighbor_counts,
                                                                       const IdType *input_nodes,
                                                                       int input_node_count,
                                                                       WMOffsetType *wm_csr_row_ptr,
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
  int sample_count = neighbor_count;
  if (max_sample_count > 0) {
    sample_count = min(neighbor_count, max_sample_count);
  }
  sample_offset[input_idx] = sample_count;
  if (NeedNeighbor) {
    neighbor_counts[input_idx] = (neighbor_count <= max_sample_count) ? 0 : neighbor_count;
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

template<typename IdType, typename WeightType, typename WMIdType, typename WMOffsetType, typename WMWeightType, typename WMLocalIdType>
void WeightedSampleWithoutReplacementCommon(const std::function<void *(size_t)> &sample_output_allocator,
                                            const std::function<void *(size_t)> &center_localid_allocator,
                                            int *sample_offset,
                                            void *wm_csr_row_ptr,
                                            void *wm_csr_col_ptr,
                                            void *wm_csr_weight_ptr,
                                            void *wm_csr_local_sorted_map_indices_ptr,
                                            const void *center_nodes,
                                            int center_node_count,
                                            int max_sample_count,
                                            const CUDAEnvFns &cuda_env_fns,
                                            cudaStream_t stream) {
  constexpr int sample_count_threshold = 1024;

  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  thread_local std::uniform_int_distribution<unsigned long long> distrib;
  unsigned long long random_seed = distrib(gen);
  const bool need_neighbor_count = max_sample_count > sample_count_threshold;
  whole_graph::TempMemoryHandle tmh;
  whole_graph::TempMemoryHandle tmh_neighbour;
  size_t tmp_size = (center_node_count + 1);
  cuda_env_fns.allocate_temp_fn(sizeof(int) * (tmp_size), &tmh);
  int *sample_count = (int *) tmh.ptr;
  int *neighbor_counts = nullptr;

  if (need_neighbor_count) {
    cuda_env_fns.allocate_temp_fn(sizeof(int) * (tmp_size), &tmh_neighbour);
    neighbor_counts = static_cast<int *>(tmh_neighbour.ptr);
    GetSampleCountAndNeighborCountWithoutReplacementKernel<IdType, WMOffsetType, true><<<DivUp(center_node_count,
                                                                                               128),
                                                                                         128, 0, stream>>>(
        sample_count,
        neighbor_counts,
        (const IdType *) center_nodes,
        center_node_count,
        (WMOffsetType *) wm_csr_row_ptr,
        max_sample_count);

  } else {
    GetSampleCountAndNeighborCountWithoutReplacementKernel<IdType, WMOffsetType, false><<<DivUp(center_node_count,
                                                                                                128),
                                                                                          128, 0, stream>>>(
        sample_count,
        neighbor_counts,
        (const IdType *) center_nodes,
        center_node_count,
        (WMOffsetType *) wm_csr_row_ptr,
        max_sample_count);
  }

  WM_CUDA_CHECK(cudaGetLastError());
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
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
  if (max_sample_count > sample_count_threshold) {
    WMThrustAllocator scan_allocator(cuda_env_fns);
    thrust::exclusive_scan(thrust::cuda::par(scan_allocator).on(stream),
                           neighbor_counts,
                           neighbor_counts + center_node_count + 1,
                           neighbor_counts);
    int *neighbor_offset = neighbor_counts;
    int target_neighbor_counts;
    WM_CUDA_CHECK(cudaMemcpyAsync(&target_neighbor_counts,
                                  neighbor_offset + center_node_count,
                                  sizeof(int),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    CUDA_STREAM_SYNC(cuda_env_fns, stream);

    whole_graph::TempMemoryHandle weights_tmh;
    cuda_env_fns.allocate_temp_fn(sizeof(WeightType) * (target_neighbor_counts), &weights_tmh);
    WeightType *target_weights_key_buf_ptr = static_cast<WeightType *>(weights_tmh.ptr);

    constexpr int BLOCK_SIZE = 256;

    WeightSampleWithoutReplacementLargeKernel<IdType,
                                              int,
                                              WeightType,
                                              WeightType,
                                              WMIdType,
                                              WMOffsetType,
                                              WMWeightType,
                                              WMLocalIdType,
                                              BLOCK_SIZE>
        <<<center_node_count, BLOCK_SIZE, 0, stream>>>(sample_output,
                                                       src_lid,
                                                       sample_offset,
                                                       neighbor_offset,
                                                       target_weights_key_buf_ptr,
                                                       (const IdType *) center_nodes,
                                                       center_node_count,
                                                       (WMOffsetType *) wm_csr_row_ptr,
                                                       (WMIdType *) wm_csr_col_ptr,
                                                       (WMWeightType *) wm_csr_weight_ptr,
                                                       (WMLocalIdType *) wm_csr_local_sorted_map_indices_ptr,
                                                       max_sample_count,
                                                       random_seed);

    WM_CUDA_CHECK(cudaGetLastError());
    CUDA_STREAM_SYNC(cuda_env_fns, stream);
    scan_allocator.deallocate_all();
    cuda_env_fns.free_temp_fn(&weights_tmh);
    cuda_env_fns.free_temp_fn(&tmh_neighbour);

    return;
  }

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
  using WeightedSampleFunType = void (*)(IdType *, int *, const int *,
                                         const IdType *, int,
                                         WMOffsetType *, WMIdType *,
                                         WMWeightType *, WMLocalIdType *, int,
                                         unsigned long long);

  static const WeightedSampleFunType func_array[7] = {
      WeightSampleWithoutReplacementKernel<IdType,
                                           int,
                                           WeightType,
                                           WMIdType,
                                           WMOffsetType,
                                           WMWeightType,
                                           WMLocalIdType,
                                           4,
                                           128>,
      WeightSampleWithoutReplacementKernel<IdType,
                                           int,
                                           WeightType,
                                           WMIdType,
                                           WMOffsetType,
                                           WMWeightType,
                                           WMLocalIdType,
                                           6,
                                           128>,
      WeightSampleWithoutReplacementKernel<IdType,
                                           int,
                                           WeightType,
                                           WMIdType,
                                           WMOffsetType,
                                           WMWeightType,
                                           WMLocalIdType,
                                           4,
                                           256>,
      WeightSampleWithoutReplacementKernel<IdType,
                                           int,
                                           WeightType,
                                           WMIdType,
                                           WMOffsetType,
                                           WMWeightType,
                                           WMLocalIdType,
                                           5,
                                           256>,
      WeightSampleWithoutReplacementKernel<IdType,
                                           int,
                                           WeightType,
                                           WMIdType,
                                           WMOffsetType,
                                           WMWeightType,
                                           WMLocalIdType,
                                           6,
                                           256>,
      WeightSampleWithoutReplacementKernel<IdType,
                                           int,
                                           WeightType,
                                           WMIdType,
                                           WMOffsetType,
                                           WMWeightType,
                                           WMLocalIdType,
                                           8,
                                           256>,
      WeightSampleWithoutReplacementKernel<IdType,
                                           int,
                                           WeightType,
                                           WMIdType,
                                           WMOffsetType,
                                           WMWeightType,
                                           WMLocalIdType,
                                           8,
                                           512>,
  };

  //  512,768,1024,1280,1536,2048,4096
  // 128,192,256,320,384,512,1024
  // Maximum one-fourth ratio , however it  may not be a good way to choose a fun.

  const int block_sizes[7] = {128, 128, 256, 256, 256, 256, 512};
  auto choose_fun_idx = [](int max_sample_count) {
    if (max_sample_count <= 128) {
      // return (max_sample_count - 1) / 32;
      return 0;
    }
    if (max_sample_count <= 384) {
      // return 4;
      return (max_sample_count - 129) / 64 + 4;
    }

    if (max_sample_count <= 512) {
      return 5;
    } else {
      return 6;
    }
  };
  int func_idx = choose_fun_idx(max_sample_count);

  int block_size = block_sizes[func_idx];

  func_array[func_idx]<<<center_node_count, block_size, 0, stream>>>(sample_output,
                                                                     src_lid,
                                                                     sample_offset,
                                                                     (const IdType *) center_nodes,
                                                                     center_node_count,
                                                                     (WMOffsetType *) wm_csr_row_ptr,
                                                                     (WMIdType *) wm_csr_col_ptr,
                                                                     (WMWeightType *) wm_csr_weight_ptr,
                                                                     (WMLocalIdType *) wm_csr_local_sorted_map_indices_ptr,
                                                                     max_sample_count,
                                                                     random_seed);

  WM_CUDA_CHECK(cudaGetLastError());
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
}

template<typename IdType, typename WeightType>
void WeightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                      const std::function<void *(size_t)> &center_localid_allocator,
                                      int *sample_offset,
                                      void *wm_csr_row_ptr,
                                      void *wm_csr_col_ptr,
                                      void *wm_csr_weight_ptr,
                                      void *wm_csr_local_sorted_map_indices_ptr,
                                      const void *center_nodes,
                                      int center_node_count,
                                      int max_sample_count,
                                      const CUDAEnvFns &cuda_env_fns,
                                      cudaStream_t stream) {
  WeightedSampleWithoutReplacementCommon<IdType, WeightType, IdType, int64_t, WeightType, int32_t>(
      sample_output_allocator,
      center_localid_allocator,
      sample_offset,
      wm_csr_row_ptr,
      wm_csr_col_ptr,
      wm_csr_weight_ptr,
      wm_csr_local_sorted_map_indices_ptr,
      center_nodes,
      center_node_count,
      max_sample_count,
      cuda_env_fns,
      stream);
}
REGISTER_DISPATCH_TWO_TYPES(WeightedSampleWithoutReplacement, WeightedSampleWithoutReplacement, SINT3264, FLOAT_DOUBLE);

void WmmpWeightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                          const std::function<void *(size_t)> &center_localid_allocator,
                                          int *sample_offset,
                                          void *wm_csr_row_ptr,
                                          void *wm_csr_col_ptr,
                                          void *wm_csr_weight_ptr,
                                          void *wm_csr_local_sorted_map_indices_ptr,
                                          WMType id_type,
                                          WMType weight_type,
                                          const void *center_nodes,
                                          int center_node_count,
                                          int max_sample_count,
                                          const CUDAEnvFns &cuda_env_fns,
                                          cudaStream_t stream) {

  DISPATCH_TWO_TYPES(id_type,
                     weight_type,
                     WeightedSampleWithoutReplacement,
                     sample_output_allocator,
                     center_localid_allocator,
                     sample_offset,
                     wm_csr_row_ptr,
                     wm_csr_col_ptr,
                     wm_csr_weight_ptr,
                     wm_csr_local_sorted_map_indices_ptr,
                     center_nodes,
                     center_node_count,
                     max_sample_count,
                     cuda_env_fns,
                     stream);
}

template<typename IdType, typename WeightType>
void ChunkedWeightedSampleWithoutReaplcement(const std::function<void *(size_t)> &sample_output_allocator,
                                             const std::function<void *(size_t)> &center_localid_allocator,
                                             int *sample_offset,
                                             void *wm_csr_row_ptr,
                                             void *wm_csr_col_ptr,
                                             void *wm_csr_weight_ptr,
                                             void *wm_csr_local_sorted_map_indices_ptr,
                                             const void *center_nodes,
                                             int center_node_count,
                                             int max_sample_count,
                                             const CUDAEnvFns &cuda_env_fns,
                                             cudaStream_t stream) {

  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle *wm_csr_row_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_row_ptr, dev_id);
  WholeChunkedMemoryHandle *wm_csr_col_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_col_ptr, dev_id);
  WholeChunkedMemoryHandle
      *wm_csr_weight_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_weight_ptr, dev_id);
  WholeChunkedMemoryHandle
      *wm_csr_data_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_local_sorted_map_indices_ptr, dev_id);
  WeightedSampleWithoutReplacementCommon<IdType,
                                         WeightType,
                                         const whole_graph::WholeChunkedMemoryHandle,
                                         const whole_graph::WholeChunkedMemoryHandle,
                                         const whole_graph::WholeChunkedMemoryHandle,
                                         const whole_graph::WholeChunkedMemoryHandle>(
      sample_output_allocator,
      center_localid_allocator,
      sample_offset,
      wm_csr_row_handle,
      wm_csr_col_handle,
      wm_csr_weight_handle,
      wm_csr_data_handle,
      center_nodes,
      center_node_count,
      max_sample_count,
      cuda_env_fns,
      stream);
}
REGISTER_DISPATCH_TWO_TYPES(ChunkedWeightedSampleWithoutReaplcement,
                            ChunkedWeightedSampleWithoutReaplcement,
                            SINT3264,
                            FLOAT_DOUBLE);

void WmmpChunkedWeightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                                 const std::function<void *(size_t)> &center_localid_allocator,
                                                 int *sample_offset,
                                                 void *wm_csr_row_ptr,
                                                 void *wm_csr_col_ptr,
                                                 void *wm_csr_weight_ptr,
                                                 void *wm_csr_local_sorted_map_indices_ptr,
                                                 WMType id_type,
                                                 WMType weight_type,
                                                 const void *center_nodes,
                                                 int center_node_count,
                                                 int max_sample_count,
                                                 const CUDAEnvFns &cuda_env_fns,
                                                 cudaStream_t stream) {

  DISPATCH_TWO_TYPES(id_type,
                     weight_type,
                     ChunkedWeightedSampleWithoutReaplcement,
                     sample_output_allocator,
                     center_localid_allocator,
                     sample_offset,
                     wm_csr_row_ptr,
                     wm_csr_col_ptr,
                     wm_csr_weight_ptr,
                     wm_csr_local_sorted_map_indices_ptr,
                     center_nodes,
                     center_node_count,
                     max_sample_count,
                     cuda_env_fns,
                     stream);
}

}// namespace whole_graph
