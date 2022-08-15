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

#include "data_type.h"
#include "macros.h"
#include "whole_chunked_memory.cuh"
#include "whole_memory.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/scan.h>

namespace whole_graph {

template<typename ValueType>
__device__ __forceinline__ bool ShouldKeep(ValueType edge_value, ValueType target_value, unsigned extract_type) {
  unsigned type_mask;
  if (edge_value == target_value) {
    type_mask = 1;
  } else if (edge_value < target_value) {
    type_mask = 2;
  } else {
    type_mask = 4;
  }
  return (type_mask & extract_type) != 0;
}

template<typename IdType, typename ValueType, typename CSRRowHandle, typename CSRColHandle, typename EdgeValueHandle>
__global__ void StatisticSubGraphDegreeKernel(int extract_type,
                                              int64_t *sample_count,
                                              const IdType *target_gid_ptr,
                                              const ValueType *filter_target_value_ptr,
                                              CSRRowHandle *edges_csr_row,
                                              CSRColHandle *edges_csr_col,
                                              EdgeValueHandle *edges_value) {
  int row_idx = blockIdx.x;
  __shared__ int extracted_count;
  if (threadIdx.x == 0) {
    extracted_count = 0;
  }
  __syncthreads();
  PtrGen<CSRRowHandle, int64_t> csr_row_ptr_gen(edges_csr_row);
  PtrGen<CSRColHandle, IdType> csr_col_ptr_gen(edges_csr_col);
  PtrGen<EdgeValueHandle, ValueType> edge_value_ptr_gen(edges_value);
  IdType nid = target_gid_ptr[row_idx];
  ValueType target_value = filter_target_value_ptr[row_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);
  int thread_extracted_count = 0;
  for (int neighbor_idx = threadIdx.x; neighbor_idx < neighbor_count; neighbor_idx += blockDim.x) {
    IdType dst_nid = *csr_col_ptr_gen.At(start + neighbor_idx);
    ValueType dst_value = *edge_value_ptr_gen.At(start + neighbor_idx);
    if (ShouldKeep<ValueType>(dst_value, target_value, extract_type)) thread_extracted_count++;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    thread_extracted_count += __shfl_down_sync(0xffffffff, thread_extracted_count, offset);
  }
  if (threadIdx.x % 32 == 0 && thread_extracted_count > 0) {
    atomicAdd_block(&extracted_count, thread_extracted_count);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    sample_count[row_idx] = extracted_count;
  }
}

template<typename IdType, typename ValueType, typename CSRRowHandle, typename CSRColHandle, typename EdgeValueHandle>
__global__ void ExtractSubGraphKernel(int extract_type,
                                      IdType *result_ids,
                                      ValueType *result_values,
                                      const int64_t *subgraph_row_ptr,
                                      const IdType *target_gid_ptr,
                                      const ValueType *filter_target_value_ptr,
                                      CSRRowHandle *edges_csr_row,
                                      CSRColHandle *edges_csr_col,
                                      EdgeValueHandle *edges_value) {
  int row_idx = blockIdx.x;
  __shared__ int shared_warp_start[32];
  __shared__ int shared_warp_count[32];
  __shared__ int shared_block_start;
  if (threadIdx.x == 0) shared_block_start = 0;
  if (threadIdx.x / 32 == 0) {
    shared_warp_start[threadIdx.x] = 0;
    shared_warp_count[threadIdx.x] = 0;
  }
  unsigned tmask = (1U << (threadIdx.x % 32)) - 1U;
  __syncthreads();
  int64_t subgraph_start = subgraph_row_ptr[row_idx];
  //int64_t subgraph_end = subgraph_row_ptr[row_idx + 1];
  PtrGen<CSRRowHandle, int64_t> csr_row_ptr_gen(edges_csr_row);
  PtrGen<CSRColHandle, IdType> csr_col_ptr_gen(edges_csr_col);
  PtrGen<EdgeValueHandle, ValueType> edge_value_ptr_gen(edges_value);
  IdType nid = target_gid_ptr[row_idx];
  ValueType target_value = filter_target_value_ptr[row_idx];
  int64_t start = *csr_row_ptr_gen.At(nid);
  int64_t end = *csr_row_ptr_gen.At(nid + 1);
  int neighbor_count = (int) (end - start);
  int step_count = DivUp(neighbor_count, blockDim.x);
  for (int s = 0; s < step_count; s++) {
    int neighbor_idx = threadIdx.x + s * blockDim.x;
    bool should_keep = false;
    IdType dst_nid;
    ValueType dst_value;
    if (neighbor_idx < neighbor_count) {
      dst_nid = *csr_col_ptr_gen.At(start + neighbor_idx);
      dst_value = *edge_value_ptr_gen.At(start + neighbor_idx);
      should_keep = ShouldKeep<ValueType>(dst_value, target_value, extract_type);
    }
    __syncwarp();
    int warp_mask = __ballot_sync(0xffffffff, should_keep);
    int warp_count = __popc(warp_mask);
    if (threadIdx.x % 32 == 0) {
      shared_warp_count[threadIdx.x / 32] = warp_count;
    }
    int idx_in_warp = __popc(warp_mask & tmask);
    __syncthreads();
    if (threadIdx.x == 0) {
      int block_start = shared_block_start;
      int result_count_in_step = 0;
      for (int widx = 0; widx < DivUp(blockDim.x, 32); widx++) {
        shared_warp_start[widx] = block_start + result_count_in_step;
        result_count_in_step += shared_warp_count[widx];
      }
      shared_block_start = block_start + result_count_in_step;
      if (s == step_count - 1) {
        assert(shared_block_start == (int) (subgraph_row_ptr[row_idx + 1] - subgraph_start));
      }
    }
    __syncthreads();
    int save_idx = shared_warp_start[threadIdx.x / 32] + idx_in_warp;
    if (should_keep) {
      result_ids[subgraph_start + save_idx] = dst_nid;
      if (result_values != nullptr) {
        result_values[subgraph_start + save_idx] = dst_value;
      }
    }
  }
}

template<typename IdType, typename ValueType, typename CSRRowHandle, typename CSRColHandle, typename EdgeValueHandle>
void WmmpExtractSubGraphWithFilterCommon(int extract_type,
                                         bool need_value_output,
                                         int64_t target_count,
                                         int64_t total_node_count,
                                         int64_t *subgraph_row_ptr,
                                         const std::function<void *(size_t)> &subgraph_col_allocator,
                                         const std::function<void *(size_t)> &subgraph_edge_value_allocator,
                                         const IdType *target_gid_ptr,
                                         const ValueType *filter_target_value_ptr,
                                         CSRRowHandle *edges_csr_row,
                                         CSRColHandle *edges_csr_col,
                                         EdgeValueHandle *edges_value,
                                         const CUDAEnvFns &cuda_env_fns,
                                         cudaStream_t stream) {
  whole_graph::TempMemoryHandle sample_count_tmh;
  auto *sample_count = (int64_t *) cuda_env_fns.allocate_temp_fn(sizeof(int64_t) * (target_count + 1), &sample_count_tmh);
  StatisticSubGraphDegreeKernel<IdType, ValueType, CSRRowHandle, CSRColHandle, EdgeValueHandle>
      <<<target_count, 64, 0, stream>>>(extract_type,
                                        sample_count,
                                        target_gid_ptr,
                                        filter_target_value_ptr,
                                        edges_csr_row,
                                        edges_csr_col,
                                        edges_value);
  WM_CUDA_CHECK(cudaGetLastError());
  WMThrustAllocator allocator(cuda_env_fns);
  thrust::exclusive_scan(thrust::cuda::par(allocator).on(stream),
                         sample_count,
                         sample_count + target_count + 1,
                         subgraph_row_ptr);
  int64_t count;
  WM_CUDA_CHECK(cudaMemcpyAsync(&count,
                                subgraph_row_ptr + target_count,
                                sizeof(int64_t),
                                cudaMemcpyDeviceToHost,
                                stream));
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  cuda_env_fns.free_temp_fn(&sample_count_tmh);
  allocator.deallocate_all();
  auto *subgraph_col_ptr = (IdType *) subgraph_col_allocator(count);
  ValueType *subgraph_edge_value_ptr = nullptr;
  if (need_value_output) subgraph_edge_value_ptr = (ValueType *) subgraph_edge_value_allocator(count);
  ExtractSubGraphKernel<IdType, ValueType, CSRRowHandle, CSRColHandle, EdgeValueHandle>
      <<<target_count, 64, 0, stream>>>(extract_type,
                                        subgraph_col_ptr,
                                        subgraph_edge_value_ptr,
                                        subgraph_row_ptr,
                                        target_gid_ptr,
                                        filter_target_value_ptr,
                                        edges_csr_row,
                                        edges_csr_col,
                                        edges_value);
  WM_CUDA_CHECK(cudaGetLastError());
}

template<typename IdType, typename ValueType>
void WmmpExtractSubGraphWithFilterFunc(int extract_type,
                                       bool need_value_output,
                                       int64_t target_count,
                                       int64_t total_node_count,
                                       int64_t *subgraph_row_ptr,
                                       const std::function<void *(size_t)> &subgraph_col_allocator,
                                       const std::function<void *(size_t)> &subgraph_edge_value_allocator,
                                       const void *target_gid_ptr,
                                       const void *filter_target_value_ptr,
                                       const int64_t *edges_csr_row,
                                       const void *edges_csr_col,
                                       const void *edges_value,
                                       const CUDAEnvFns &cuda_env_fns,
                                       cudaStream_t stream) {
  WmmpExtractSubGraphWithFilterCommon<IdType, ValueType, const int64_t, const IdType, const ValueType>(
      extract_type,
      need_value_output,
      target_count,
      total_node_count,
      subgraph_row_ptr,
      subgraph_col_allocator,
      subgraph_edge_value_allocator,
      (const IdType *) target_gid_ptr,
      (const ValueType *) filter_target_value_ptr,
      (const int64_t *) edges_csr_row,
      (const IdType *) edges_csr_col,
      (const ValueType *) edges_value,
      cuda_env_fns,
      stream);
}

REGISTER_DISPATCH_TWO_TYPES(WmmpExtractSubGraphWithFilterFuncInt,
                            WmmpExtractSubGraphWithFilterFunc,
                            SINT3264,
                            ALLINT)
REGISTER_DISPATCH_TWO_TYPES(WmmpExtractSubGraphWithFilterFuncFloat,
                            WmmpExtractSubGraphWithFilterFunc,
                            SINT3264,
                            HALF_FLOAT_DOUBLE)

void WmmpExtractSubGraphWithFilter(WMType id_type,
                                   WMType edge_value_type,
                                   int extract_type,
                                   bool need_value_output,
                                   int64_t target_count,
                                   int64_t total_node_count,
                                   int64_t *subgraph_row_ptr,
                                   const std::function<void *(size_t)> &subgraph_col_allocator,
                                   const std::function<void *(size_t)> &subgraph_edge_value_allocator,
                                   const void *target_gid_ptr,
                                   const void *filter_target_value_ptr,
                                   const int64_t *edges_csr_row,
                                   const void *edges_csr_col,
                                   const void *edges_value,
                                   const CUDAEnvFns &cuda_env_fns,
                                   cudaStream_t stream) {
  bool is_value_int = edge_value_type == WMT_Int8 || edge_value_type == WMT_Int16 || edge_value_type == WMT_Int32
      || edge_value_type == WMT_Int64;
  bool is_value_float = edge_value_type == WMT_Half || edge_value_type == WMT_Float || edge_value_type == WMT_Double;
  WM_CHECK(is_value_float || is_value_int);
  if (is_value_float) {
    DISPATCH_TWO_TYPES(id_type, edge_value_type,
                       WmmpExtractSubGraphWithFilterFuncFloat,
                       extract_type,
                       need_value_output,
                       target_count,
                       total_node_count,
                       subgraph_row_ptr,
                       subgraph_col_allocator,
                       subgraph_edge_value_allocator,
                       target_gid_ptr,
                       filter_target_value_ptr,
                       edges_csr_row,
                       edges_csr_col,
                       edges_value,
                       cuda_env_fns,
                       stream);
  } else {
    DISPATCH_TWO_TYPES(id_type, edge_value_type,
                       WmmpExtractSubGraphWithFilterFuncInt,
                       extract_type,
                       need_value_output,
                       target_count,
                       total_node_count,
                       subgraph_row_ptr,
                       subgraph_col_allocator,
                       subgraph_edge_value_allocator,
                       target_gid_ptr,
                       filter_target_value_ptr,
                       edges_csr_row,
                       edges_csr_col,
                       edges_value,
                       cuda_env_fns,
                       stream);
  }
}

template<typename IdType, typename ValueType>
void WmmpExtractSubGraphWithFilterChunkedFunc(int extract_type,
                                              bool need_value_output,
                                              int64_t target_count,
                                              int64_t total_node_count,
                                              int64_t *subgraph_row_ptr,
                                              const std::function<void *(size_t)> &subgraph_col_allocator,
                                              const std::function<void *(size_t)> &subgraph_edge_value_allocator,
                                              const void *target_gid_ptr,
                                              const void *filter_target_value_ptr,
                                              WholeChunkedMemory_t edges_csr_row,
                                              WholeChunkedMemory_t edges_csr_col,
                                              WholeChunkedMemory_t edges_value,
                                              const CUDAEnvFns &cuda_env_fns,
                                              cudaStream_t stream) {
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle
      *wm_edges_csr_row_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) edges_csr_row, dev_id);
  WholeChunkedMemoryHandle
      *wm_edges_csr_col_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) edges_csr_col, dev_id);
  WholeChunkedMemoryHandle *wm_edges_value_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) edges_value, dev_id);
  WmmpExtractSubGraphWithFilterCommon<IdType,
                                      ValueType,
                                      const WholeChunkedMemoryHandle,
                                      const WholeChunkedMemoryHandle,
                                      const WholeChunkedMemoryHandle>(
      extract_type,
      need_value_output,
      target_count,
      total_node_count,
      subgraph_row_ptr,
      subgraph_col_allocator,
      subgraph_edge_value_allocator,
      (const IdType *) target_gid_ptr,
      (const ValueType *) filter_target_value_ptr,
      wm_edges_csr_row_handle,
      wm_edges_csr_col_handle,
      wm_edges_value_handle,
      cuda_env_fns,
      stream);
}

REGISTER_DISPATCH_TWO_TYPES(WmmpExtractSubGraphWithFilterChunkedFuncInt,
                            WmmpExtractSubGraphWithFilterChunkedFunc,
                            SINT3264,
                            ALLINT)
REGISTER_DISPATCH_TWO_TYPES(WmmpExtractSubGraphWithFilterChunkedFuncFloat,
                            WmmpExtractSubGraphWithFilterChunkedFunc,
                            SINT3264,
                            HALF_FLOAT_DOUBLE)

void WmmpExtractSubGraphWithFilterChunked(WMType id_type,
                                          WMType edge_value_type,
                                          int extract_type,
                                          bool need_value_output,
                                          int64_t target_count,
                                          int64_t total_node_count,
                                          int64_t *subgraph_row_ptr,
                                          const std::function<void *(size_t)> &subgraph_col_allocator,
                                          const std::function<void *(size_t)> &subgraph_edge_value_allocator,
                                          const void *target_gid_ptr,
                                          const void *filter_target_value_ptr,
                                          WholeChunkedMemory_t edges_csr_row,
                                          WholeChunkedMemory_t edges_csr_col,
                                          WholeChunkedMemory_t edges_value,
                                          const CUDAEnvFns &cuda_env_fns,
                                          cudaStream_t stream) {
  bool is_value_int = edge_value_type == WMT_Int8 || edge_value_type == WMT_Int16 || edge_value_type == WMT_Int32
      || edge_value_type == WMT_Int64;
  bool is_value_float = edge_value_type == WMT_Half || edge_value_type == WMT_Float || edge_value_type == WMT_Double;
  WM_CHECK(is_value_float || is_value_int);
  if (is_value_float) {
    DISPATCH_TWO_TYPES(id_type, edge_value_type,
                       WmmpExtractSubGraphWithFilterChunkedFuncFloat,
                       extract_type,
                       need_value_output,
                       target_count,
                       total_node_count,
                       subgraph_row_ptr,
                       subgraph_col_allocator,
                       subgraph_edge_value_allocator,
                       target_gid_ptr,
                       filter_target_value_ptr,
                       edges_csr_row,
                       edges_csr_col,
                       edges_value,
                       cuda_env_fns,
                       stream);
  } else {
    DISPATCH_TWO_TYPES(id_type, edge_value_type,
                       WmmpExtractSubGraphWithFilterChunkedFuncInt,
                       extract_type,
                       need_value_output,
                       target_count,
                       total_node_count,
                       subgraph_row_ptr,
                       subgraph_col_allocator,
                       subgraph_edge_value_allocator,
                       target_gid_ptr,
                       filter_target_value_ptr,
                       edges_csr_row,
                       edges_csr_col,
                       edges_value,
                       cuda_env_fns,
                       stream);
  }
}

}// namespace whole_graph