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

#include <string>
#include <unordered_set>

#include "cuda_env_fns.h"
#include "data_type.h"
#include "file_utils.h"
#include "macros.h"
#include "parallel_utils.h"
#include "whole_chunked_memory.cuh"
#include "whole_chunked_memory.h"
#include "whole_memory.h"
#include "whole_memory_communicator.h"

namespace whole_graph {

int64_t StatFilelistEltCount(const std::string &file_prefix,
                             WMType id_type) {
  std::vector<std::string> filelist;
  if (!GetPartFileListFromPrefix(file_prefix, &filelist)) {
    std::cerr << "[StatFilelistEltCount] GetPartFileListFromPrefix from prefix " << file_prefix
              << " failed.\n";
    abort();
  }
  size_t elt_size = GetWMTSize(id_type);
  int64_t total_size = 0;
  for (auto filename : filelist) {
    size_t filesize = StatFileSize(filename);
    WM_CHECK(filesize % elt_size == 0);
    total_size += filesize / elt_size;
  }
  return total_size;
}

static constexpr int kJumpCOOSpan = 32;

int64_t WmmpGetJumpCOORowSize(int64_t total_edge_count) {
  return DivUp(total_edge_count, kJumpCOOSpan);
}

template<typename IdType, typename HandleType, typename CsrRowPtrHandleType>
__global__ void GenerateJumpCOORowKernel(HandleType *jump_coo_row,
                                         CsrRowPtrHandleType *csr_row_ptr,
                                         int64_t total_node_count,
                                         int64_t total_edge_count,
                                         int64_t start_node_id,
                                         int64_t end_node_id) {
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  constexpr int kBlockSize = 256;
  __shared__ int count[kBlockSize];
  __shared__ int64_t start_jump_idx[kBlockSize];
  int warp_id = tidx / 32;
  int lane_id = tidx % 32;
  IdType block_start_idx = start_node_id + (int64_t) bidx * kBlockSize;
  IdType warp_start_idx = block_start_idx + warp_id * 32;
  IdType thread_node_id = warp_start_idx + lane_id;
  whole_graph::PtrGen<HandleType, IdType> jump_coo_row_gen(jump_coo_row);
  whole_graph::PtrGen<CsrRowPtrHandleType, int64_t> csr_row_ptr_gen(csr_row_ptr);
  int64_t node_edge_start_offset, node_edge_end_offset;
  if (thread_node_id < total_node_count && thread_node_id < end_node_id) {
    node_edge_start_offset = *csr_row_ptr_gen.At(thread_node_id);
    node_edge_end_offset = *csr_row_ptr_gen.At(thread_node_id + 1);
    assert(node_edge_start_offset <= total_edge_count);
    assert(node_edge_end_offset <= total_edge_count);
    int64_t node_start_jump_idx = DivUp(node_edge_start_offset, kJumpCOOSpan);
    int64_t node_end_jump_idx = DivUp(node_edge_end_offset, kJumpCOOSpan);
    start_jump_idx[tidx] = node_start_jump_idx;
    count[tidx] = node_end_jump_idx - node_start_jump_idx;
  } else {
    count[tidx] = 0;
  }
  __syncthreads();
  for (int target_lane = 0; target_lane < 32; target_lane++) {
    int target_tid = target_lane + warp_id * 32;
    if (count[target_tid] == 0) continue;
    IdType target_tid_node_id = warp_start_idx + target_lane;
    for (int idx = lane_id; idx < count[target_tid]; idx += 32) {
      *jump_coo_row_gen.At(start_jump_idx[target_tid] + idx) = target_tid_node_id;
    }
    __syncwarp();
  }
}

template<typename IdType, bool IsChunked>
void GenerateJumpCOORowCommon(void *wm_jump_coo_row,
                              void *wm_csr_row_ptr,
                              int64_t total_node_count,
                              int64_t total_edge_count,
                              BootstrapCommunicator *bootstrap_communicator,
                              cudaStream_t stream) {
  int rank_idx = bootstrap_communicator->Rank();
  int size = bootstrap_communicator->Size();
  int64_t start_node_id = AlignUp(total_node_count * rank_idx / size, 256);
  int64_t end_node_id = AlignUp(total_node_count * (rank_idx + 1) / size, 256);
  int64_t local_node_count = end_node_id - start_node_id;
  int block_count = local_node_count / 256;
  if (block_count == 0) {
    WmmpBarrier(bootstrap_communicator);
    return;
  }
  if (!IsChunked) {
    GenerateJumpCOORowKernel<IdType, IdType, const int64_t><<<block_count, 256, 0, stream>>>((IdType *) wm_jump_coo_row,
                                                                                             (const int64_t *) wm_csr_row_ptr,
                                                                                             total_node_count,
                                                                                             total_edge_count,
                                                                                             start_node_id,
                                                                                             end_node_id);
  } else {
    GenerateJumpCOORowKernel<IdType,
                             WholeChunkedMemoryHandle,
                             const WholeChunkedMemoryHandle><<<block_count, 256, 0, stream>>>((WholeChunkedMemoryHandle *) wm_jump_coo_row,
                                                                                              (const WholeChunkedMemoryHandle *) wm_csr_row_ptr,
                                                                                              total_node_count,
                                                                                              total_edge_count,
                                                                                              start_node_id,
                                                                                              end_node_id);
  }
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  WmmpBarrier(bootstrap_communicator);
}

template<typename IdType>
void GenerateJumpCOORow(void *wm_jump_coo_row,
                        void *wm_csr_row_ptr,
                        int64_t total_node_count,
                        int64_t total_edge_count,
                        BootstrapCommunicator *bootstrap_communicator,
                        cudaStream_t stream) {
  GenerateJumpCOORowCommon<IdType, false>(wm_jump_coo_row,
                                          wm_csr_row_ptr,
                                          total_node_count,
                                          total_edge_count,
                                          bootstrap_communicator,
                                          stream);
}

REGISTER_DISPATCH_ONE_TYPE(GenerateJumpCOORow, GenerateJumpCOORow, SINT3264)

void WmmpGenerateJumpCOORow(void *wm_jump_coo_row,
                            void *wm_csr_row_ptr,
                            int64_t total_node_count,
                            int64_t total_edge_count,
                            WMType id_type,
                            cudaStream_t stream) {
  auto *bootstrap_communicator = WmmpGetBootstrapCommunicator(wm_csr_row_ptr);
  WM_CHECK(id_type == WMT_Int32 || id_type == WMT_Int64);
  DISPATCH_ONE_TYPE(id_type, GenerateJumpCOORow,
                    wm_jump_coo_row,
                    wm_csr_row_ptr,
                    total_node_count,
                    total_edge_count,
                    bootstrap_communicator,
                    stream);
}

template<typename IdType>
void GenerateChunkedJumpCOORow(void *wm_jump_coo_row,
                               void *wm_csr_row_ptr,
                               int64_t total_node_count,
                               int64_t total_edge_count,
                               BootstrapCommunicator *bootstrap_communicator,
                               cudaStream_t stream) {
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle
      *wm_jump_coo_row_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_jump_coo_row, dev_id);
  WholeChunkedMemoryHandle
      *wm_csr_row_ptr_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_row_ptr, dev_id);
  GenerateJumpCOORowCommon<IdType, true>(wm_jump_coo_row_handle,
                                         wm_csr_row_ptr_handle,
                                         total_node_count,
                                         total_edge_count,
                                         bootstrap_communicator,
                                         stream);
}

REGISTER_DISPATCH_ONE_TYPE(GenerateChunkedJumpCOORow, GenerateChunkedJumpCOORow, SINT3264)

void WmmpGenerateChunkedJumpCOORow(WholeChunkedMemory_t wm_jump_coo_row,
                                   WholeChunkedMemory_t wm_csr_row_ptr,
                                   int64_t total_node_count,
                                   int64_t total_edge_count,
                                   WMType id_type,
                                   cudaStream_t stream) {
  auto *bootstrap_communicator = WcmmpGetBootstrapCommunicator(wm_csr_row_ptr);
  WM_CHECK(id_type == WMT_Int32 || id_type == WMT_Int64);
  DISPATCH_ONE_TYPE(id_type, GenerateChunkedJumpCOORow,
                    wm_jump_coo_row,
                    wm_csr_row_ptr,
                    total_node_count,
                    total_edge_count,
                    bootstrap_communicator,
                    stream);
}

template<typename IdType, typename NodeIDHandleType, typename OffsetHandleType>
__global__ void GetEdgeNodesFromEidKernel(OffsetHandleType *wm_csr_row_ptr,
                                          NodeIDHandleType *wm_csr_col_idx,
                                          NodeIDHandleType *wm_jump_coo_row,
                                          const int64_t *edge_idx_list,
                                          int64_t total_src_node_count,
                                          int64_t total_edge_count,
                                          IdType *src_ptr,
                                          IdType *dst_ptr,
                                          int64_t edge_list_count) {
  int warp_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int lane_idx = threadIdx.x;
  if (warp_idx >= edge_list_count) return;
  int64_t edge_id = edge_idx_list[warp_idx];
  if (src_ptr != nullptr) {
    int find = 0;
    int64_t edge_jump_id = edge_id / kJumpCOOSpan;
    whole_graph::PtrGen<NodeIDHandleType, IdType> jump_coo_row_gen(wm_jump_coo_row);
    IdType jump_src_node_id = *jump_coo_row_gen.At(edge_jump_id);
    whole_graph::PtrGen<OffsetHandleType, int64_t> csr_row_ptr_gen(wm_csr_row_ptr);
    IdType search_node_id = jump_src_node_id + lane_idx;
    for (; search_node_id < total_src_node_count; search_node_id += 32) {
      int64_t start = *csr_row_ptr_gen.At(search_node_id);
      int64_t end = *csr_row_ptr_gen.At(search_node_id + 1);
      if (start <= edge_id && end > edge_id) find = 1;
      if (end > edge_id) break;
    }
    __syncwarp();
    unsigned find_mask = __ballot_sync(0xffffffff, find);
    assert(find_mask != 0 && __popc(find_mask) == 1);
    int find_lane = __ffs(find_mask) - 1;
    if (lane_idx == find_lane) {
      src_ptr[warp_idx] = search_node_id;
    }
  }
  if (dst_ptr != nullptr) {
    whole_graph::PtrGen<NodeIDHandleType, IdType> csr_col_idx_gen(wm_csr_col_idx);
    if (lane_idx == 0) {
      dst_ptr[warp_idx] = *csr_col_idx_gen.At(edge_id);
    }
  }
}

template<typename IdType>
void GetEdgeNodesFromEid(void *wm_csr_row_ptr,
                         void *wm_csr_col_idx,
                         void *wm_jump_coo_row,
                         const int64_t *edge_idx_list,
                         int64_t total_src_node_count,
                         int64_t total_edge_count,
                         void *src_ptr,
                         void *dst_ptr,
                         int64_t edge_list_count,
                         cudaStream_t stream) {
  int block_count = DivUp(edge_list_count, 4);
  dim3 block_size(32, 4);
  GetEdgeNodesFromEidKernel<IdType, const IdType, const int64_t><<<block_count, block_size, 0, stream>>>(
      (const int64_t *) wm_csr_row_ptr,
      (const IdType *) wm_csr_col_idx,
      (const IdType *) wm_jump_coo_row,
      edge_idx_list,
      total_src_node_count,
      total_edge_count,
      (IdType *) src_ptr,
      (IdType *) dst_ptr,
      edge_list_count);
}

REGISTER_DISPATCH_ONE_TYPE(GetEdgeNodesFromEid, GetEdgeNodesFromEid, SINT3264)

void WmmpGetEdgeNodesFromEid(void *wm_csr_row_ptr,
                             void *wm_csr_col_idx,
                             void *wm_jump_coo_row,
                             const int64_t *edge_idx_list,
                             WMType id_type,
                             int64_t total_src_node_count,
                             int64_t total_edge_count,
                             void *src_ptr,
                             void *dst_ptr,
                             int64_t edge_list_count,
                             cudaStream_t stream) {
  WM_CHECK(id_type == WMT_Int32 || id_type == WMT_Int64);
  DISPATCH_ONE_TYPE(id_type, GetEdgeNodesFromEid,
                    wm_csr_row_ptr,
                    wm_csr_col_idx,
                    wm_jump_coo_row,
                    edge_idx_list,
                    total_src_node_count,
                    total_edge_count,
                    src_ptr,
                    dst_ptr,
                    edge_list_count,
                    stream);
}

template<typename IdType>
void GetEdgeNodesFromEidChunked(WholeChunkedMemory_t wm_csr_row_ptr,
                                WholeChunkedMemory_t wm_csr_col_idx,
                                WholeChunkedMemory_t wm_jump_coo_row,
                                const int64_t *edge_idx_list,
                                int64_t total_src_node_count,
                                int64_t total_edge_count,
                                void *src_ptr,
                                void *dst_ptr,
                                int64_t edge_list_count,
                                cudaStream_t stream) {
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle
      *wm_csr_row_ptr_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_row_ptr, dev_id);
  WholeChunkedMemoryHandle
      *wm_csr_col_idx_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_csr_col_idx, dev_id);
  WholeChunkedMemoryHandle
      *wm_jump_coo_row_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) wm_jump_coo_row, dev_id);
  int block_count = DivUp(edge_list_count, 4);
  dim3 block_size(32, 4);
  GetEdgeNodesFromEidKernel<IdType,
                            const WholeChunkedMemoryHandle,
                            const WholeChunkedMemoryHandle><<<block_count, block_size, 0, stream>>>(
      wm_csr_row_ptr_handle,
      wm_csr_col_idx_handle,
      wm_jump_coo_row_handle,
      edge_idx_list,
      total_src_node_count,
      total_edge_count,
      (IdType *) src_ptr,
      (IdType *) dst_ptr,
      edge_list_count);
}

REGISTER_DISPATCH_ONE_TYPE(GetEdgeNodesFromEidChunked, GetEdgeNodesFromEidChunked, SINT3264)

void WmmpGetEdgeNodesFromEidChunked(WholeChunkedMemory_t wm_csr_row_ptr,
                                    WholeChunkedMemory_t wm_csr_col_idx,
                                    WholeChunkedMemory_t wm_jump_coo_row,
                                    const int64_t *edge_idx_list,
                                    WMType id_type,
                                    int64_t total_src_node_count,
                                    int64_t total_edge_count,
                                    void *src_ptr,
                                    void *dst_ptr,
                                    int64_t edge_list_count,
                                    cudaStream_t stream) {
  WM_CHECK(id_type == WMT_Int32 || id_type == WMT_Int64);
  DISPATCH_ONE_TYPE(id_type, GetEdgeNodesFromEidChunked,
                    wm_csr_row_ptr,
                    wm_csr_col_idx,
                    wm_jump_coo_row,
                    edge_idx_list,
                    total_src_node_count,
                    total_edge_count,
                    src_ptr,
                    dst_ptr,
                    edge_list_count,
                    stream);
}

}// namespace whole_graph