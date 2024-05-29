/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cub/device/device_radix_sort.cuh>
#include <random>
#include <thrust/scan.h>

#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/integer_utils.hpp>
#include <wholememory/device_reference.cuh>
#include <wholememory/env_func_ptrs.h>
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>

#include "wholememory_ops/output_memory_handle.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

#include "cuda_macros.hpp"
#include "error.hpp"
#include "sample_comm.cuh"

namespace wholegraph_ops {

template <typename IdType, typename WMOffsetType>
__global__ void get_sample_count_without_replacement_kernel(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  const IdType* input_nodes,
  const int input_node_count,
  int* tmp_sample_count_mem_pointer,
  const int max_sample_count)
{
  int gidx      = threadIdx.x + blockIdx.x * blockDim.x;
  int input_idx = gidx;
  if (input_idx >= input_node_count) return;
  IdType nid = input_nodes[input_idx];
  wholememory::device_reference<WMOffsetType> wm_csr_row_ptr_dev_ref(wm_csr_row_ptr);
  int64_t start      = wm_csr_row_ptr_dev_ref[nid];
  int64_t end        = wm_csr_row_ptr_dev_ref[nid + 1];
  int neighbor_count = (int)(end - start);
  // sample_count <= 0 means sample all.
  if (max_sample_count > 0) { neighbor_count = min(neighbor_count, max_sample_count); }
  tmp_sample_count_mem_pointer[input_idx] = neighbor_count;
}

template <typename IdType, typename LocalIdType, typename WMIdType, typename WMOffsetType>
__global__ void large_sample_kernel(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  const IdType* input_nodes,
  const int input_node_count,
  const int max_sample_count,
  raft::random::detail::DeviceState<raft::random::detail::PCGenerator> rngstate,
  const int* sample_offset,
  wholememory_array_description_t sample_offset_desc,
  WMIdType* output,
  int* src_lid,
  int64_t* output_edge_gid_ptr)
{
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  raft::random::detail::PCGenerator rng(rngstate, (uint64_t)gidx);
  wholememory::device_reference<WMOffsetType> csr_row_ptr_gen(wm_csr_row_ptr);
  wholememory::device_reference<WMIdType> csr_col_ptr_gen(wm_csr_col_ptr);

  IdType nid         = input_nodes[input_idx];
  int64_t start      = csr_row_ptr_gen[nid];
  int64_t end        = csr_row_ptr_gen[nid + 1];
  int neighbor_count = (int)(end - start);
  int offset         = sample_offset[input_idx];
  // sample all
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += blockDim.x) {
      int neighbor_idx           = sample_id;
      IdType gid                 = csr_col_ptr_gen[start + neighbor_idx];
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = (LocalIdType)input_idx;
      if (output_edge_gid_ptr) {
        output_edge_gid_ptr[offset + sample_id] = (int64_t)(start + neighbor_idx);
      }
    }
    return;
  }
  for (int sample_id = threadIdx.x; sample_id < max_sample_count; sample_id += blockDim.x) {
    output[offset + sample_id] = (IdType)sample_id;
    if (src_lid) src_lid[offset + sample_id] = (LocalIdType)input_idx;
  }
  __syncthreads();
  for (int idx = max_sample_count + threadIdx.x; idx < neighbor_count; idx += blockDim.x) {
    raft::random::detail::UniformDistParams<int32_t> params;
    params.start = 0;
    params.end   = 1;
    int32_t rand_num;
    raft::random::detail::custom_next(rng, &rand_num, params, 0, 0);
    rand_num %= idx + 1;
    if (rand_num < max_sample_count) { atomicMax((int*)(output + offset + rand_num), idx); }
  }
  __syncthreads();
  for (int sample_id = threadIdx.x; sample_id < max_sample_count; sample_id += blockDim.x) {
    int neighbor_idx           = *(int*)(output + offset + sample_id);
    output[offset + sample_id] = csr_col_ptr_gen[start + neighbor_idx];
    if (output_edge_gid_ptr) {
      output_edge_gid_ptr[offset + sample_id] = (int64_t)(start + neighbor_idx);
    }
  }
}

template <typename IdType,
          typename LocalIdType,
          typename WMIdType,
          typename WMOffsetType,
          int BLOCK_DIM        = 32,
          int ITEMS_PER_THREAD = 1>
__global__ void unweighted_sample_without_replacement_kernel(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  const IdType* input_nodes,
  const int input_node_count,
  const int max_sample_count,
  raft::random::detail::DeviceState<raft::random::detail::PCGenerator> rngstate,
  const int* sample_offset,
  wholememory_array_description_t sample_offset_desc,
  WMIdType* output,
  int* src_lid,
  int64_t* output_edge_gid_ptr)
{
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  raft::random::detail::PCGenerator rng(rngstate, (uint64_t)gidx);
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;

  wholememory::device_reference<WMOffsetType> csr_row_ptr_gen(wm_csr_row_ptr);
  wholememory::device_reference<WMIdType> csr_col_ptr_gen(wm_csr_col_ptr);

  IdType nid         = input_nodes[input_idx];
  int64_t start      = csr_row_ptr_gen[nid];
  int64_t end        = csr_row_ptr_gen[nid + 1];
  int neighbor_count = (int)(end - start);
  if (neighbor_count <= 0) return;
  int offset = sample_offset[input_idx];
  // use all neighbors if neighbors less than max_sample_count
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += blockDim.x) {
      IdType gid                 = csr_col_ptr_gen[start + sample_id];
      output[offset + sample_id] = gid;
      if (src_lid) src_lid[offset + sample_id] = input_idx;
      if (output_edge_gid_ptr) { output_edge_gid_ptr[offset + sample_id] = start + sample_id; }
    }
    return;
  }
  uint64_t sa_p[ITEMS_PER_THREAD];
  int M = max_sample_count;
  int N = neighbor_count;
  // UnWeightedIndexSampleWithOutReplacement<BLOCK_DIM, ITEMS_PER_THREAD>(M, N,
  // sa_p, rng);
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
    int idx = i * BLOCK_DIM + threadIdx.x;
    raft::random::detail::UniformDistParams<int32_t> params;
    params.start = 0;
    params.end   = 1;
    int32_t rand_num;
    raft::random::detail::custom_next(rng, &rand_num, params, 0, 0);
    int32_t r = idx < M ? rand_num % (N - idx) : N;
    sa_p[i]   = ((uint64_t)r << 32UL) | idx;
  }
  __syncthreads();
  BlockRadixSort(shared_data.temp_storage).SortBlockedToStriped(sa_p);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx                                     = i * BLOCK_DIM + threadIdx.x;
    int s                                       = (int)(sa_p[i] >> 32UL);
    shared_data.sample_shared_data.s.value[idx] = s;
    int p                                       = sa_p[i] & 0xFFFFFFFF;
    shared_data.sample_shared_data.p.value[idx] = p;
    if (idx < M) shared_data.sample_shared_data.q.value[p] = idx;
    shared_data.sample_shared_data.chain.value[idx] = idx;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_DIM + threadIdx.x;
    int si  = shared_data.sample_shared_data.s.value[idx];
    int si1 = shared_data.sample_shared_data.s.value[idx + 1];
    if (idx < M && (idx == M - 1 || si != si1) && si >= N - M) {
      shared_data.sample_shared_data.chain.value[N - si - 1] =
        shared_data.sample_shared_data.p.value[idx];
    }
  }
  __syncthreads();
  for (int step = 0; step < log2_up_device(M); ++step) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int idx = i * BLOCK_DIM + threadIdx.x;
      shared_data.sample_shared_data.last_chain_tmp.value[idx] =
        shared_data.sample_shared_data.chain.value[idx];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int idx = i * BLOCK_DIM + threadIdx.x;
      if (idx < M) {
        shared_data.sample_shared_data.chain.value[idx] =
          shared_data.sample_shared_data.last_chain_tmp
            .value[shared_data.sample_shared_data.last_chain_tmp.value[idx]];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_DIM + threadIdx.x;
    shared_data.sample_shared_data.last_chain_tmp.value[idx] =
      N - shared_data.sample_shared_data.chain.value[idx] - 1;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_DIM + threadIdx.x;
    int ai;
    if (idx < M) {
      int qi = shared_data.sample_shared_data.q.value[idx];
      if (idx == 0 || qi == 0 ||
          shared_data.sample_shared_data.s.value[qi] !=
            shared_data.sample_shared_data.s.value[qi - 1]) {
        ai = shared_data.sample_shared_data.s.value[qi];
      } else {
        int prev_i = shared_data.sample_shared_data.p.value[qi - 1];
        ai         = shared_data.sample_shared_data.last_chain_tmp.value[prev_i];
      }
      sa_p[i] = ai;
    }
  }
  // Output
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int idx = i * BLOCK_DIM + threadIdx.x;
    int ai  = sa_p[i];
    if (idx < M) {
      IdType gid           = csr_col_ptr_gen[start + ai];
      output[offset + idx] = gid;
      if (src_lid) src_lid[offset + idx] = (LocalIdType)input_idx;
      if (output_edge_gid_ptr) { output_edge_gid_ptr[offset + idx] = (int64_t)(start + ai); }
    }
  }
}

template <typename IdType, typename WMIdType>
void wholegraph_csr_unweighted_sample_without_replacement_func(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  void* center_nodes,
  wholememory_array_description_t center_nodes_desc,
  int max_sample_count,
  void* output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  int center_node_count = center_nodes_desc.size;

  WHOLEMEMORY_EXPECTS(wm_csr_row_ptr_desc.dtype == WHOLEMEMORY_DT_INT64,
                      "wholegraph_csr_unweighted_sample_without_replacement_func(). "
                      "wm_csr_row_ptr_desc.dtype != WHOLEMEMORY_DT_INT64, "
                      "wm_csr_row_ptr_desc.dtype = %d",
                      wm_csr_row_ptr_desc.dtype);

  WHOLEMEMORY_EXPECTS(output_sample_offset_desc.dtype == WHOLEMEMORY_DT_INT,
                      "wholegraph_csr_unweighted_sample_without_replacement_func(). "
                      "output_sample_offset_desc.dtype != WHOLEMEMORY_DT_INT, "
                      "output_sample_offset_desc.dtype = %d",
                      output_sample_offset_desc.dtype);

  wholememory_ops::temp_memory_handle gen_buffer_tmh(p_env_fns);
  int* tmp_sample_count_mem_pointer =
    (int*)gen_buffer_tmh.device_malloc(center_node_count + 1, WHOLEMEMORY_DT_INT);

  int thread_x    = 32;
  int block_count = raft::div_rounding_up_safe<int>(center_node_count, thread_x);

  get_sample_count_without_replacement_kernel<IdType, int64_t>
    <<<block_count, thread_x, 0, stream>>>(wm_csr_row_ptr,
                                           wm_csr_row_ptr_desc,
                                           (const IdType*)center_nodes,
                                           center_node_count,
                                           tmp_sample_count_mem_pointer,
                                           max_sample_count);
  WM_CUDA_CHECK(cudaGetLastError());

  // prefix sum
  wholememory_ops::wm_thrust_allocator thrust_allocator(p_env_fns);
  thrust::exclusive_scan(thrust::cuda::par_nosync(thrust_allocator).on(stream),
                         tmp_sample_count_mem_pointer,
                         tmp_sample_count_mem_pointer + center_node_count + 1,
                         (int*)output_sample_offset);

  int count;
  WM_CUDA_CHECK(cudaMemcpyAsync(&count,
                                ((int*)output_sample_offset) + center_node_count,
                                sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));

  wholememory_ops::output_memory_handle gen_output_dest_buffer_mh(p_env_fns,
                                                                  output_dest_memory_context);
  WMIdType* output_dest_node_ptr =
    (WMIdType*)gen_output_dest_buffer_mh.device_malloc(count, wm_csr_col_ptr_desc.dtype);

  int* output_center_localid_ptr = nullptr;
  if (output_center_localid_memory_context) {
    wholememory_ops::output_memory_handle gen_output_center_localid_buffer_mh(
      p_env_fns, output_center_localid_memory_context);
    output_center_localid_ptr =
      (int*)gen_output_center_localid_buffer_mh.device_malloc(count, WHOLEMEMORY_DT_INT);
  }

  int64_t* output_edge_gid_ptr = nullptr;
  if (output_edge_gid_memory_context) {
    wholememory_ops::output_memory_handle gen_output_edge_gid_buffer_mh(
      p_env_fns, output_edge_gid_memory_context);
    output_edge_gid_ptr =
      (int64_t*)gen_output_edge_gid_buffer_mh.device_malloc(count, WHOLEMEMORY_DT_INT64);
  }
  // sample node
  raft::random::RngState _rngstate(random_seed, 0, raft::random::GeneratorType::GenPC);
  raft::random::detail::DeviceState<raft::random::detail::PCGenerator> rngstate(_rngstate);
  if (max_sample_count <= 0) {
    sample_all_kernel<IdType, int, WMIdType, int64_t>
      <<<center_node_count, 64, 0, stream>>>(wm_csr_row_ptr,
                                             wm_csr_row_ptr_desc,
                                             wm_csr_col_ptr,
                                             wm_csr_col_ptr_desc,
                                             (const IdType*)center_nodes,
                                             center_node_count,
                                             (const int*)output_sample_offset,
                                             output_sample_offset_desc,
                                             (WMIdType*)output_dest_node_ptr,
                                             (int*)output_center_localid_ptr,
                                             (int64_t*)output_edge_gid_ptr);

    WM_CUDA_CHECK(cudaGetLastError());
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    return;
  }

  if (max_sample_count > 1024) {
    large_sample_kernel<IdType, int, WMIdType, int64_t>
      <<<center_node_count, 32, 0, stream>>>(wm_csr_row_ptr,
                                             wm_csr_row_ptr_desc,
                                             wm_csr_col_ptr,
                                             wm_csr_col_ptr_desc,
                                             (const IdType*)center_nodes,
                                             center_node_count,
                                             max_sample_count,
                                             rngstate,
                                             (const int*)output_sample_offset,
                                             output_sample_offset_desc,
                                             (WMIdType*)output_dest_node_ptr,
                                             (int*)output_center_localid_ptr,
                                             (int64_t*)output_edge_gid_ptr);
    WM_CUDA_CHECK(cudaGetLastError());
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    return;
  }

  typedef void (*unweighted_sample_func_type)(
    wholememory_gref_t wm_csr_row_ptr,
    wholememory_array_description_t wm_csr_row_ptr_desc,
    wholememory_gref_t wm_csr_col_ptr,
    wholememory_array_description_t wm_csr_col_ptr_desc,
    const IdType* input_nodes,
    const int input_node_count,
    const int max_sample_count,
    raft::random::detail::DeviceState<raft::random::detail::PCGenerator> rngstate,
    const int* sample_offset,
    wholememory_array_description_t sample_offset_desc,
    WMIdType* output,
    int* src_lid,
    int64_t* output_edge_gid_ptr);
  static const unweighted_sample_func_type func_array[32] = {
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 32, 1>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 32, 2>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 32, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 64, 2>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 64, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 64, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 128, 2>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 128, 2>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 128, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 128, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 128, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 128, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 2>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 2>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 2>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 2>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 3>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 4>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 4>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 4>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 4>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 4>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 4>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 4>,
    unweighted_sample_without_replacement_kernel<IdType, int, WMIdType, int64_t, 256, 4>};
  static const int warp_count_array[32] = {1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8,
                                           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
  int func_idx                          = (max_sample_count - 1) / 32;
  func_array[func_idx]<<<center_node_count, warp_count_array[func_idx] * 32, 0, stream>>>(
    wm_csr_row_ptr,
    wm_csr_row_ptr_desc,
    wm_csr_col_ptr,
    wm_csr_col_ptr_desc,
    (const IdType*)center_nodes,
    center_node_count,
    max_sample_count,
    rngstate,
    (const int*)output_sample_offset,
    output_sample_offset_desc,
    (WMIdType*)output_dest_node_ptr,
    (int*)output_center_localid_ptr,
    (int64_t*)output_edge_gid_ptr);
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
}
}  // namespace wholegraph_ops
