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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>

#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/integer_utils.hpp>
#include <wholememory/device_reference.cuh>
#include <wholememory/env_func_ptrs.h>
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory_tensor.h>

#include "wholememory_ops/output_memory_handle.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

#include "cuda_macros.hpp"
#include "error.hpp"
#include "sample_comm.cuh"

#include "wholememory_ops/gather_op_impl.h"
using wholememory_ops::wholememory_gather_nccl;
#define WARP_SIZE 32

namespace wholegraph_ops {

template <typename IdType,
          typename LocalIdType,
          typename WMIdType,
          typename DegreeType,
          int BLOCK_DIM        = 32,
          int ITEMS_PER_THREAD = 1>
__global__ void unweighted_sample_without_replacement_nccl_kernel(
  const IdType* input_nodes,
  const WMIdType* csr_row_ptr_sta,
  const DegreeType* in_degree,
  const int input_node_count,
  const int max_sample_count,
  raft::random::detail::DeviceState<raft::random::detail::PCGenerator> rngstate,
  const int* sample_offset,
  wholememory_array_description_t sample_offset_desc,
  int* src_lid,
  int64_t* output_edge_gid_ptr)
{
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  raft::random::detail::PCGenerator rng(rngstate, (uint64_t)gidx);
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;

  WMIdType start     = csr_row_ptr_sta[input_idx];
  WMIdType end       = start + in_degree[input_idx];
  int neighbor_count = (int)(in_degree[input_idx]);
  if (neighbor_count <= 0) return;
  int offset = sample_offset[input_idx];
  // use all neighbors if neighbors less than max_sample_count
  if (neighbor_count <= max_sample_count) {
    for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += blockDim.x) {
      output_edge_gid_ptr[offset + sample_id] = start + sample_id;
      if (src_lid) src_lid[offset + sample_id] = input_idx;
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
      output_edge_gid_ptr[offset + idx] = (int64_t)(start + ai);
      if (src_lid) src_lid[offset + idx] = (LocalIdType)input_idx;
    }
  }
}

template <typename IdType, typename WMIdType>
void wholegraph_csr_unweighted_sample_without_replacement_nccl_func(
  wholememory_handle_t csr_row_wholememory_handle,
  wholememory_handle_t csr_col_wholememory_handle,
  wholememory_tensor_description_t wm_csr_row_ptr_desc,
  wholememory_tensor_description_t wm_csr_col_ptr_desc,
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
                      "wholegraph_csr_unweighted_sample_without_replacement_nccl_func(). "
                      "wm_csr_row_ptr_desc.dtype != WHOLEMEMORY_DT_INT64, "
                      "wm_csr_row_ptr_desc.dtype = %d",
                      wm_csr_row_ptr_desc.dtype);

  WHOLEMEMORY_EXPECTS(output_sample_offset_desc.dtype == WHOLEMEMORY_DT_INT,
                      "wholegraph_csr_unweighted_sample_without_replacement_nccl_func(). "
                      "output_sample_offset_desc.dtype != WHOLEMEMORY_DT_INT, "
                      "output_sample_offset_desc.dtype = %d",
                      output_sample_offset_desc.dtype);

  auto double_center_node_count = center_node_count + center_node_count;
  wholememory_ops::temp_memory_handle center_nodes_buf(p_env_fns);
  IdType* center_nodes_expandshift_one = static_cast<IdType*>(
    center_nodes_buf.device_malloc(double_center_node_count, center_nodes_desc.dtype));
  // fill center_nodes_shift_one with [center_nodes, center_nodes+1]
  wholememory_ops::wm_thrust_allocator thrust_allocator(p_env_fns);
  thrust::counting_iterator<int64_t> iota(0);
  thrust::for_each(thrust::cuda::par_nosync(thrust_allocator).on(stream),
                   iota,
                   iota + double_center_node_count,
                   ExpandWithOffsetFunc<IdType>{
                     (const IdType*)center_nodes, center_nodes_expandshift_one, center_node_count});

  // gathering [rowoffsets, rowoffsets+1]
  wholememory_ops::temp_memory_handle output_buf(p_env_fns);
  int64_t* center_nodes_indptr =
    static_cast<int64_t*>(output_buf.device_malloc(double_center_node_count, WHOLEMEMORY_DT_INT64));
  wholememory_array_description_t center_nodes_expandshift_one_desc{
    double_center_node_count, 0, center_nodes_desc.dtype};
  wholememory_matrix_description_t center_nodes_indptr_desc{
    {double_center_node_count, 1}, 1, 0, wm_csr_row_ptr_desc.dtype};
  wholememory_matrix_description_t wm_csr_row_ptr_mat_desc;
  wholememory_convert_tensor_desc_to_matrix(&wm_csr_row_ptr_mat_desc, &wm_csr_row_ptr_desc);
  wholememory_ops::wholememory_gather_nccl(csr_row_wholememory_handle,
                                           wm_csr_row_ptr_mat_desc,
                                           center_nodes_expandshift_one,
                                           center_nodes_expandshift_one_desc,
                                           center_nodes_indptr,
                                           center_nodes_indptr_desc,
                                           p_env_fns,
                                           stream,
                                           -1);
  // find the in_degree (subtraction) and sample count
  // temporarily store sampled_csr_ptr_buf (# of degrees/samples per node) in int32;
  // can be changed to int8_t/16_t later
  wholememory_ops::temp_memory_handle sampled_csr_ptr_buf(p_env_fns);
  int* in_degree =
    static_cast<int*>(sampled_csr_ptr_buf.device_malloc(center_node_count + 1, WHOLEMEMORY_DT_INT));
  thrust::for_each(
    thrust::cuda::par_nosync(thrust_allocator).on(stream),
    iota,
    iota + center_node_count,
    ReduceForDegrees<int64_t, int>{center_nodes_indptr, in_degree, center_node_count});
  // prefix sum to get the output_sample_offset (depending on min(max_sample_count and in_degree))
  int sampled_count = max_sample_count <= 0 ? std::numeric_limits<int>::max() : max_sample_count;
  thrust::transform_exclusive_scan(thrust::cuda::par_nosync(thrust_allocator).on(stream),
                                   in_degree,
                                   in_degree + center_node_count + 1,
                                   (int*)output_sample_offset,
                                   MinInDegreeFanout<int>{sampled_count},
                                   0,
                                   thrust::plus<int>());
  // start local sampling
  int count;
  WM_CUDA_CHECK(cudaMemcpyAsync(&count,
                                ((int*)output_sample_offset) + center_node_count,
                                sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));

  int64_t* output_edge_gid_ptr = nullptr;
  wholememory_ops::temp_memory_handle edge_gid_buffer_mh(p_env_fns);
  if (output_edge_gid_memory_context) {
    wholememory_ops::output_memory_handle gen_output_edge_gid_buffer_mh(
      p_env_fns, output_edge_gid_memory_context);
    output_edge_gid_ptr =
      (int64_t*)gen_output_edge_gid_buffer_mh.device_malloc(count, WHOLEMEMORY_DT_INT64);
  } else {
    output_edge_gid_ptr = (int64_t*)edge_gid_buffer_mh.device_malloc(count, WHOLEMEMORY_DT_INT64);
  }

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
  raft::random::RngState _rngstate(random_seed, 0, raft::random::GeneratorType::GenPC);
  raft::random::detail::DeviceState<raft::random::detail::PCGenerator> rngstate(_rngstate);
  {
    typedef void (*unweighted_sample_func_type)(
      const IdType* input_nodes,
      const int64_t* center_nodes_indptr,
      const int* in_degree,
      const int input_node_count,
      const int max_sample_count,
      raft::random::detail::DeviceState<raft::random::detail::PCGenerator> rngstate,
      const int* sample_offset,
      wholememory_array_description_t sample_offset_desc,
      int* src_lid,
      int64_t* output_edge_gid_ptr);
    static const unweighted_sample_func_type func_array[32] = {
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 32, 1>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 32, 2>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 32, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 64, 2>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 64, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 64, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 128, 2>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 128, 2>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 128, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 128, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 128, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 128, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 2>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 2>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 2>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 2>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 3>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 4>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 4>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 4>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 4>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 4>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 4>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 4>,
      unweighted_sample_without_replacement_nccl_kernel<IdType, int, int64_t, int, 256, 4>};
    static const int warp_count_array[32] = {1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8,
                                             8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
    int func_idx                          = (max_sample_count - 1) / 32;
    func_array[func_idx]<<<center_node_count, warp_count_array[func_idx] * 32, 0, stream>>>(
      (const IdType*)center_nodes,
      (const int64_t*)center_nodes_indptr,
      (const int*)in_degree,
      center_node_count,
      sampled_count,
      rngstate,
      (const int*)output_sample_offset,
      output_sample_offset_desc,
      (int*)output_center_localid_ptr,
      (int64_t*)output_edge_gid_ptr);

    wholememory_matrix_description_t wm_csr_col_ptr_mat_desc;
    wholememory_matrix_description_t output_dest_node_ptr_desc{
      {count, 1}, 1, 0, wm_csr_col_ptr_desc.dtype};
    wholememory_array_description_t output_edge_gid_ptr_desc{count, 0, WHOLEMEMORY_DT_INT64};
    wholememory_convert_tensor_desc_to_matrix(&wm_csr_col_ptr_mat_desc, &wm_csr_col_ptr_desc);
    wholememory_ops::wholememory_gather_nccl(csr_col_wholememory_handle,
                                             wm_csr_col_ptr_mat_desc,
                                             output_edge_gid_ptr,
                                             output_edge_gid_ptr_desc,
                                             output_dest_node_ptr,
                                             output_dest_node_ptr_desc,
                                             p_env_fns,
                                             stream,
                                             -1);
  }
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
}
}  // namespace wholegraph_ops
