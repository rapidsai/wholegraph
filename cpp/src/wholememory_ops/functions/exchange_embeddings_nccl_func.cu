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
#include "exchange_embeddings_nccl_func.h"

#include <vector>

#include <cub/device/device_radix_sort.cuh>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <wholememory/communicator.hpp>

#include "cuda_macros.hpp"
#include "logger.hpp"
#include "wholememory_ops/register.hpp"

namespace wholememory_ops {

wholememory_error_code_t exchange_embeddings_nccl_func(const void* dev_local_gather_buffer_ptr,
                                                       const int64_t* host_send_to_rank_count_ptr,
                                                       const int64_t* host_recv_from_rank_count_ptr,
                                                       void* dev_embedding_recv_buffer_ptr,
                                                       size_t embedding_size,
                                                       wholememory_comm_t wm_comm,
                                                       cudaStream_t stream)
{
  try {
    int world_size;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));
    std::vector<size_t> embedding_send_counts(world_size), embedding_send_displs(world_size);
    std::vector<size_t> embedding_recv_counts(world_size), embedding_recv_displs(world_size);
    size_t send_disp = 0, recv_disp = 0;
    for (int i = 0; i < world_size; i++) {
      embedding_send_displs[i] = send_disp;
      embedding_recv_displs[i] = recv_disp;
      size_t send_count        = host_send_to_rank_count_ptr[i] * embedding_size;
      size_t recv_count        = host_recv_from_rank_count_ptr[i] * embedding_size;
      embedding_send_counts[i] = send_count;
      embedding_recv_counts[i] = recv_count;
      send_disp += send_count;
      recv_disp += recv_count;
    }
    wm_comm->alltoallv(dev_local_gather_buffer_ptr,
                       dev_embedding_recv_buffer_ptr,
                       embedding_send_counts.data(),
                       embedding_send_displs.data(),
                       embedding_recv_counts.data(),
                       embedding_recv_displs.data(),
                       WHOLEMEMORY_DT_INT8,
                       stream);
    WM_CUDA_DEBUG_SYNC_STREAM(stream);
    // WHOLEMEMORY_EXPECTS(wm_comm->sync_stream(stream) == WHOLEMEMORY_SUCCESS,
    //                    "Embedding AllToAllV failed.");
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("exchange_embeddings_nccl_func LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

__global__ void DedupIndiceAndGradientsKernel(int raw_count,
                                              int unique_count,
                                              const int* start_pos,
                                              const int* mapping_array,
                                              const float* grads,
                                              float* dedup_grads,
                                              int embedding_dim,
                                              int embedding_stride)
{
  int bidx         = blockIdx.x;
  int start_offset = start_pos[bidx];
  int end_offset   = raw_count;
  if (bidx != unique_count - 1) end_offset = start_pos[bidx + 1];
  for (int idx = start_offset; idx < end_offset; idx++) {
    int map_idx                    = mapping_array[idx];
    const float* current_grads_ptr = grads + map_idx * embedding_stride;
    float* current_dedup_grads_ptr = dedup_grads + bidx * embedding_stride;
    if (idx == start_offset) {
      for (int dim = threadIdx.x; dim < embedding_dim; dim += blockDim.x) {
        current_dedup_grads_ptr[dim] = current_grads_ptr[dim];
      }
    } else {
      for (int dim = threadIdx.x; dim < embedding_dim; dim += blockDim.x) {
        current_dedup_grads_ptr[dim] += current_grads_ptr[dim];
      }
    }
  }
}

template <typename IndexT>
void dedup_indice_and_gradients_temp_func(int64_t* run_count,
                                          const void* indices_ptr,
                                          wholememory_array_description_t indice_desc,
                                          const float* grads,
                                          wholememory_matrix_description_t grads_desc,
                                          void* dedup_indice_ptr,
                                          float* dedup_grads,
                                          wholememory_env_func_t* p_env_fn,
                                          cudaStream_t stream)
{
  const IndexT* indice = static_cast<const IndexT*>(indices_ptr);
  IndexT* dedup_indice = static_cast<IndexT*>(dedup_indice_ptr);
  int raw_count        = indice_desc.size;
  if (raw_count == 0) {
    *run_count = 0;
    return;
  }
  IndexT* sorted_indice = dedup_indice;
  wm_thrust_allocator allocator(p_env_fn);
  wholememory_ops::temp_memory_handle mapping_sequence_handle(p_env_fn);
  int* dev_mapping_sequence =
    static_cast<int*>(mapping_sequence_handle.device_malloc(raw_count * 2, WHOLEMEMORY_DT_INT));
  int* dev_indice_mapping = dev_mapping_sequence + raw_count;
  thrust::sequence(thrust::cuda::par_nosync(allocator).on(stream),
                   dev_mapping_sequence,
                   dev_mapping_sequence + raw_count,
                   0);
  void* cub_temp_storage    = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indice,
                                  sorted_indice,
                                  dev_mapping_sequence,
                                  dev_indice_mapping,
                                  raw_count,
                                  0,
                                  sizeof(IndexT) * 8,
                                  stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indice,
                                  sorted_indice,
                                  dev_mapping_sequence,
                                  dev_indice_mapping,
                                  raw_count,
                                  0,
                                  sizeof(IndexT) * 8,
                                  stream);
  auto thrust_ret      = thrust::unique_by_key(thrust::cuda::par(allocator).on(stream),
                                          sorted_indice,
                                          sorted_indice + raw_count,
                                          dev_mapping_sequence);
  *run_count           = thrust_ret.first - sorted_indice;
  int embedding_dim    = grads_desc.sizes[1];
  int embedding_stride = grads_desc.stride;
  int thread_count     = std::min<int>(embedding_dim, 256);
  DedupIndiceAndGradientsKernel<<<*run_count, thread_count, 0, stream>>>(raw_count,
                                                                         *run_count,
                                                                         dev_mapping_sequence,
                                                                         dev_indice_mapping,
                                                                         grads,
                                                                         dedup_grads,
                                                                         embedding_dim,
                                                                         embedding_stride);
  WM_CUDA_CHECK_NO_THROW(cudaGetLastError());
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
}

REGISTER_DISPATCH_ONE_TYPE(DedupIndiceAndGradientsTempFunc,
                           dedup_indice_and_gradients_temp_func,
                           SINT3264)

int64_t dedup_indice_and_gradients(const void* indices,
                                   wholememory_array_description_t indice_desc,
                                   const float* grads,
                                   wholememory_matrix_description_t grads_desc,
                                   void* dedup_indice,
                                   float* dedup_grads,
                                   wholememory_env_func_t* p_env_fn,
                                   cudaStream_t stream)
{
  WHOLEMEMORY_CHECK_NOTHROW(indice_desc.dtype == WHOLEMEMORY_DT_INT ||
                            indice_desc.dtype == WHOLEMEMORY_DT_INT64);
  WHOLEMEMORY_CHECK_NOTHROW(indice_desc.size == grads_desc.sizes[0]);
  WHOLEMEMORY_CHECK_NOTHROW(grads_desc.dtype == WHOLEMEMORY_DT_FLOAT);
  int64_t run_count = 0;
  DISPATCH_ONE_TYPE(indice_desc.dtype,
                    DedupIndiceAndGradientsTempFunc,
                    &run_count,
                    indices,
                    indice_desc,
                    grads,
                    grads_desc,
                    dedup_indice,
                    dedup_grads,
                    p_env_fn,
                    stream);
  return run_count;
}

}  // namespace wholememory_ops
