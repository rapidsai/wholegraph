/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#ifdef WITH_NVSHMEM_SUPPORT
#pragma once

#include "gather_scatter_func.cuh"
#include "nvshmem_device_reference.cuh"
#include "wholememory/communicator.hpp"
#include "wholememory/device_reference.cuh"
#include "wholememory/global_reference.h"
#include "wholememory/memory_handle.hpp"
#include "wholememory/tensor_description.h"
#include "wholememory_ops/register.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
namespace wholememory_ops {
template <typename InputIteratorT, typename OffsetT, typename T>
__device__ __forceinline__ OffsetT UpperBound(InputIteratorT input, OffsetT num_items, T val)
{
  OffsetT retval = 0;
  while (num_items > 0) {
    OffsetT half = num_items >> 1;
    if (val < input[retval + half]) {
      num_items = half;
    } else {
      retval    = retval + (half + 1);
      num_items = num_items - (half + 1);
    }
  }
  return retval;
}

template <typename InputIteratorT, typename OffsetT, typename T>
__device__ __forceinline__ OffsetT LowerBound(InputIteratorT input, OffsetT num_items, T val)
{
  OffsetT retval = 0;
  while (num_items > 0) {
    OffsetT half = num_items >> 1;
    if (input[retval + half] < val) {
      retval    = retval + (half + 1);
      num_items = num_items - (half + 1);
    } else {
      num_items = half;
    }
  }
  return retval;
}

template <typename IndexT>
void sort_index_in_pair(const void* indices_before_sort,
                        int64_t indice_count,
                        void* indices_after_sort,
                        IndexT* raw_indices,  // output
                        wholememory_comm_t wm_comm,
                        wm_thrust_allocator* p_thrust_allocator,
                        cudaStream_t stream)
{
  wm_thrust_allocator& allocator = *p_thrust_allocator;

  IndexT* seq_indices =
    reinterpret_cast<IndexT*>(allocator.allocate(indice_count * sizeof(IndexT)));
  thrust::sequence(
    thrust::cuda::par_nosync(allocator).on(stream), seq_indices, seq_indices + indice_count, 0);
  // TODO: use unsigned type (wm_ops::UTypeT) can put all negative indices at last. But maybe
  // later... using UTypeT = typename UnsignedType<IndexT>::UType;
  auto indices_to_sort      = static_cast<const IndexT*>(indices_before_sort);
  auto sorted_indice        = static_cast<IndexT*>(indices_after_sort);
  void* cub_temp_storage    = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indices_to_sort,
                                  sorted_indice,
                                  seq_indices,
                                  raw_indices,
                                  indice_count,
                                  0,
                                  sizeof(IndexT) * 8,
                                  stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(cub_temp_storage,
                                  temp_storage_bytes,
                                  indices_to_sort,
                                  sorted_indice,
                                  seq_indices,
                                  raw_indices,
                                  indice_count,
                                  0,
                                  sizeof(IndexT) * 8,
                                  stream);
  allocator.deallocate(reinterpret_cast<char*>(seq_indices), indice_count * sizeof(IndexT));
  allocator.deallocate(static_cast<char*>(cub_temp_storage), temp_storage_bytes);
}

template <typename EmbeddingT, typename IndexT, int ALIGNMENT = 1, bool USE_IBGDA = true>
__global__ void gather_func_with_nvshmem_sort_idxs_kernel(
  wholememory_nvshmem_ref_t embeding_nvshmem_ref,
  wholememory_matrix_description_t embedding_desc,
  const IndexT* __restrict__ sorted_index,  // sorted (input) index to select
  const IndexT* __restrict__ output_index,  // output index to drop in output array
  int64_t indice_count,
  const int max_blocks_for_local,
  const int intra_node_ranks,
  const int node_rank,
  size_t* embedding_entry_offsets,
  EmbeddingT* __restrict__ temp_output,
  wholememory_matrix_description_t output_desc,
  const int threads_per_group)
{
  const int64_t local_index_lowerbound = embedding_entry_offsets[node_rank * intra_node_ranks];
  const int64_t local_index_upperbound =
    embedding_entry_offsets[(node_rank + 1) * intra_node_ranks];
  const int64_t local_index_start  = LowerBound(sorted_index, indice_count, local_index_lowerbound);
  const int64_t local_index_length = UpperBound(
    sorted_index + local_index_start, indice_count - local_index_start, local_index_upperbound - 1);
  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;
  extern __shared__ char shmem[];
  nvshmem_device_reference<EmbeddingT> embedding_nvshmem_device_ref{embeding_nvshmem_ref};
  embedding_nvshmem_device_ref.mov_offsets_to_shmem(shmem);
  if (blockIdx.x >= max_blocks_for_local) {
    const int64_t thread_id = (blockIdx.x - max_blocks_for_local) * blockDim.x + threadIdx.x;
    for (int64_t row_id = thread_id; row_id < indice_count - local_index_length;
         row_id += (gridDim.x - max_blocks_for_local) * blockDim.x) {
      const int64_t scaled_row_id =
        row_id < local_index_start ? row_id : row_id + local_index_length;
      int64_t embedding_table_idx = sorted_index[scaled_row_id];
      const int64_t output_idx    = output_index[scaled_row_id];

      if (embedding_table_idx < 0) continue;
      EmbeddingT* temp_output_ptr =
        temp_output + output_stride * output_idx + output_desc.storage_offset;
      int64_t embedding_offset =
        embedding_desc.storage_offset + embedding_table_idx * embedding_stride;
      int dest_rank = embedding_nvshmem_device_ref.dest_rank(embedding_offset);
      EmbeddingT* symmetric_address =
        embedding_nvshmem_device_ref.symmetric_address(embedding_offset);
      if (USE_IBGDA) {
        nvshmem_getmem(temp_output_ptr,
                       const_cast<const EmbeddingT*>(symmetric_address),
                       embedding_size * sizeof(EmbeddingT),
                       dest_rank);
      } else {
        nvshmem_getmem_nbi(temp_output_ptr,
                           const_cast<const EmbeddingT*>(symmetric_address),
                           embedding_size * sizeof(EmbeddingT),
                           dest_rank);
      }
    }
  }

  else {
    //  one embedding per block
    const int thread_id_in_group = threadIdx.x % threads_per_group;
    const int group_id_in_block  = threadIdx.x / threads_per_group;
    const int groups_per_block   = blockDim.x / threads_per_group;
    const int group_id           = blockIdx.x * groups_per_block + group_id_in_block;
    for (int64_t row_id = local_index_start + group_id;
         row_id < local_index_start + local_index_length;
         row_id += max_blocks_for_local * groups_per_block) {
      IndexT embedding_table_idx = sorted_index[row_id];
      IndexT output_idx          = output_index[row_id];
      if (embedding_table_idx < 0) continue;
      // printf("*********in kernel : idx_id =%ld , embedding_table_idx:%ld, output_idx:%ld
      // ,\n",idx_id, int64_t(embedding_table_idx),int64_t( output_idx));
      EmbeddingT* temp_output_ptr =
        temp_output + output_stride * output_idx + output_desc.storage_offset;
      int64_t embedding_offset =
        embedding_desc.storage_offset + embedding_table_idx * embedding_stride;
      EmbeddingT* peer_embedding_ptr = static_cast<EmbeddingT*>(
        nvshmem_ptr(embedding_nvshmem_device_ref.symmetric_address(embedding_offset),
                    embedding_nvshmem_device_ref.dest_rank(embedding_offset)));
      if (peer_embedding_ptr == nullptr) {
        printf("Error: Could not find peer NVSHMEM array.\n");
        __trap();
      }
      for (int emb_idx = thread_id_in_group * ALIGNMENT; emb_idx < embedding_size;
           emb_idx += ALIGNMENT * threads_per_group) {
        mov_data<sizeof(EmbeddingT) * ALIGNMENT>(temp_output_ptr + emb_idx,
                                                 peer_embedding_ptr + emb_idx);
      }
    }
  }
}

template <typename EmbeddingT, typename OutputT, int ALIGNMENT = 1>

__global__ void embedding_output_convert_date_type_kernel(
  OutputT* output,
  const EmbeddingT* input,
  int64_t embedding_count,
  int64_t embedding_dim,
  wholememory_matrix_description_t output_desc,
  wholememory_matrix_description_t input_desc)
{
  int thread_idx        = threadIdx.x;
  int64_t output_stride = output_desc.stride;
  int64_t input_stride  = input_desc.stride;
  typed_data_vector<EmbeddingT, ALIGNMENT> input_vector;
  typed_data_vector<OutputT, ALIGNMENT> output_vector;
  for (int64_t output_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       output_idx < embedding_count;
       output_idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    OutputT* output_ptr  = output + output_desc.storage_offset + output_stride * output_idx;
    int64_t input_offset = output_idx * input_stride + input_desc.storage_offset;
    for (int emb_idx = thread_idx * ALIGNMENT; emb_idx < embedding_dim;
         emb_idx += ALIGNMENT * blockDim.x) {
      mov_data<sizeof(EmbeddingT) * ALIGNMENT>(&input_vector, input + input_offset + emb_idx);

#pragma unroll
      for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
        typed_data_vector_at(output_vector, sub_idx) =
          convert_type<EmbeddingT, OutputT>(typed_data_vector_at(input_vector, sub_idx));
      }
      mov_data<sizeof(OutputT) * ALIGNMENT>(output_ptr + emb_idx, &output_vector);
    }
  }
}

template <typename EmbeddingT, typename OutputT>
void embedding_output_convert_date_type_temp_func(OutputT* output,
                                                  const EmbeddingT* input,
                                                  int64_t embedding_count,
                                                  int64_t embedding_dim,
                                                  wholememory_matrix_description_t output_desc,
                                                  wholememory_matrix_description_t input_desc,
                                                  cudaStream_t stream)
{
  if (embedding_count == 0 || input_desc.sizes[1] == 0 || output_desc.sizes[0] == 0) return;

  int im_alignment   = determine_memory_alignment_elt_count(input, input_desc);
  int om_alignment   = determine_memory_alignment_elt_count(output, output_desc);
  int alignment      = std::min<int>(im_alignment, om_alignment);
  int embedding_size = embedding_dim;
  int thread_x       = wholememory::div_rounding_up_safe<int>(embedding_size, alignment);
  thread_x           = std::min(thread_x, 256);
  int thread_y       = 1;
  if (thread_x < 64) {
    int power2_thread_x = 1;
    for (; power2_thread_x < thread_x; power2_thread_x *= 2)
      ;
    thread_x = power2_thread_x;
    thread_y = 64 / thread_x;
  }
  int64_t block_count_64 = (embedding_count + thread_y - 1) / thread_y;
  int block_count = block_count_64 >= INT_MAX ? INT_MAX / 4 : static_cast<int>(block_count_64);
  dim3 block_dim(thread_x, thread_y, 1);

  void (*kernel_fn)(OutputT*,
                    const EmbeddingT*,
                    int64_t,
                    int64_t,
                    wholememory_matrix_description_t,
                    wholememory_matrix_description_t) = nullptr;

  switch (alignment) {
    case 16: {
      kernel_fn = embedding_output_convert_date_type_kernel<EmbeddingT, OutputT, 16>;
      break;
    }
    case 8: {
      kernel_fn = embedding_output_convert_date_type_kernel<EmbeddingT, OutputT, 8>;
      break;
    }
    case 4: {
      kernel_fn = embedding_output_convert_date_type_kernel<EmbeddingT, OutputT, 4>;
      break;
    }
    case 2: {
      kernel_fn = embedding_output_convert_date_type_kernel<EmbeddingT, OutputT, 2>;
      break;
    }
    case 1: {
      kernel_fn = embedding_output_convert_date_type_kernel<EmbeddingT, OutputT, 1>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("embedding_output_convert_date_type func alignment=%d.", alignment);
      return;
    }
  }
  kernel_fn<<<block_count, block_dim, 0, stream>>>(
    output, input, embedding_count, embedding_dim, output_desc, input_desc);
  WM_CUDA_CHECK(cudaGetLastError());
}

template <typename EmbeddingT, typename IndexT, typename OutputT>
void nvshmem_gather_temp_get_mem_sort_idx_func(wholememory_comm_t wm_comm,
                                               wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                                               wholememory_matrix_description_t embedding_desc,
                                               const void* indices,
                                               int64_t indice_count,
                                               void* output,
                                               void* temp_output,
                                               wholememory_matrix_description_t output_desc,
                                               size_t* embedding_entry_offsets,
                                               wholememory_env_func_t* p_env_fns,
                                               cudaStream_t stream,
                                               int gather_sms)
{
  wm_thrust_allocator thrust_allocator(p_env_fns);

  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;
  temp_memory_handle dev_raw_indice(p_env_fns);
  IndexT* dev_raw_indice_ptr = static_cast<IndexT*>(
    dev_raw_indice.device_malloc(indice_count, get_wholememory_dtype<IndexT>()));
  temp_memory_handle dev_sorted_indice(p_env_fns);
  void* dev_sorted_indice_ptr =
    dev_sorted_indice.device_malloc(indice_count, get_wholememory_dtype<IndexT>());
  wholememory_matrix_description_t temp_output_desc = output_desc;
  temp_output_desc.storage_offset                   = 0;
  sort_index_in_pair(indices,
                     indice_count,
                     dev_sorted_indice_ptr,
                     dev_raw_indice_ptr,
                     wm_comm,
                     &thrust_allocator,
                     stream);
  int intra_node_rank_num = wm_comm->intra_node_rank_num;
  int node_id             = wm_comm->world_rank / wm_comm->intra_node_rank_num;

  auto ret_data     = static_cast<EmbeddingT*>(temp_output);
  auto sorted_index = static_cast<IndexT*>(dev_sorted_indice_ptr);

  int wm_alignment = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment = determine_memory_alignment_elt_count(temp_output, temp_output_desc);
  int alignment    = std::min<int>(wm_alignment, mm_alignment);

  const char* use_ibgda = std::getenv("NVSHMEM_IB_ENABLE_IBGDA");
  bool use_ibgda_flag   = ((use_ibgda != nullptr) && (strcmp(use_ibgda, "1") == 0));
  void (*gather_nvshmem_kernel_fn)(wholememory_nvshmem_ref_t,
                                   wholememory_matrix_description_t,
                                   const IndexT*,
                                   const IndexT*,
                                   int64_t,
                                   const int,
                                   const int,
                                   const int,
                                   size_t*,
                                   EmbeddingT*,
                                   wholememory_matrix_description_t,
                                   const int) = nullptr;
  switch (alignment) {
    case 16: {
      gather_nvshmem_kernel_fn =
        use_ibgda_flag ? gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 16, true>
                       : gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 16, false>;
      break;
    }
    case 8: {
      gather_nvshmem_kernel_fn =
        use_ibgda_flag ? gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 8, true>
                       : gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 8, false>;
      break;
    }
    case 4: {
      gather_nvshmem_kernel_fn =
        use_ibgda_flag ? gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 4, true>
                       : gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 4, false>;
      break;
    }
    case 2: {
      gather_nvshmem_kernel_fn =
        use_ibgda_flag ? gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 2, true>
                       : gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 2, false>;
      break;
    }
    case 1: {
      gather_nvshmem_kernel_fn =
        use_ibgda_flag ? gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 1, true>
                       : gather_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 1, false>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("gather func alignment=%d.", alignment);
      return;
    }
  }

  int block_threshold     = 64;
  int max_blocks          = 1024;
  constexpr int WARP_SIZE = 32;

  const int max_threads_per_block   = 256;
  const int num_threads_per_feature = std::min<int64_t>(
    max_threads_per_block,
    ((embedding_desc.sizes[1] / alignment) + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
  const int block_size =
    (max_threads_per_block / num_threads_per_feature) * num_threads_per_feature;
  const int ngroup_per_block = block_size / num_threads_per_feature;
  int num_blocks =
    std::min(max_blocks, static_cast<int>(indice_count + ngroup_per_block) / ngroup_per_block);
  // re-adjust num_blocks/block_threshold to make at least 1 block available for nvshmem remote get
  if (num_blocks < block_threshold) {
    block_threshold = 1;
    if (num_blocks == 1) num_blocks = 2;
  }
  size_t shared_mem_size =
    embeding_nvshmem_ptr.same_chunk ? 0 : ((embeding_nvshmem_ptr.world_size + 1) * sizeof(size_t));
  gather_nvshmem_kernel_fn<<<num_blocks, block_size, shared_mem_size, stream>>>(
    embeding_nvshmem_ptr,
    embedding_desc,
    sorted_index,
    dev_raw_indice_ptr,
    indice_count,
    block_threshold,
    intra_node_rank_num,
    node_id,
    embedding_entry_offsets,
    ret_data,
    temp_output_desc,
    num_threads_per_feature);
  if (!use_ibgda_flag) {
    nvshmemx_quiet_on_stream(stream);  // wait transfer
  }

  int64_t output_ele_size = wholememory_get_memory_element_count_from_matrix(&output_desc);
  if constexpr (sizeof(EmbeddingT) == sizeof(OutputT)) {
    WM_CUDA_CHECK(cudaMemcpyAsync(static_cast<OutputT*>(output) + output_desc.storage_offset,
                                  temp_output,
                                  output_ele_size * sizeof(OutputT),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
  } else {
    embedding_output_convert_date_type_temp_func<EmbeddingT, OutputT>(
      static_cast<OutputT*>(output),
      static_cast<EmbeddingT*>(temp_output),
      indice_count,
      embedding_desc.sizes[1],
      output_desc,
      temp_output_desc,
      stream);
  }

  WM_CUDA_CHECK(cudaGetLastError());
  (void)gather_sms;
}

template <typename EmbeddingT, typename IndexT, int ALIGNMENT = 1, bool USE_IBGDA = true>
__global__ void scatter_func_with_nvshmem_sort_idxs_kernel(
  EmbeddingT* __restrict__ temp_input,
  wholememory_matrix_description_t temp_input_desc,
  wholememory_nvshmem_ref_t embeding_nvshmem_ref,
  wholememory_matrix_description_t embedding_desc,
  const IndexT* __restrict__ sorted_index,
  const IndexT* __restrict__ input_index,
  int64_t indice_count,
  const int max_blocks_for_local,
  const int intra_node_ranks,
  const int node_rank,
  size_t* embedding_entry_offsets,
  const int threads_per_group)
{
  const int64_t local_index_lowerbound = embedding_entry_offsets[node_rank * intra_node_ranks];
  const int64_t local_index_upperbound =
    embedding_entry_offsets[(node_rank + 1) * intra_node_ranks];
  const int64_t local_index_start  = LowerBound(sorted_index, indice_count, local_index_lowerbound);
  const int64_t local_index_length = UpperBound(
    sorted_index + local_index_start, indice_count - local_index_start, local_index_upperbound - 1);

  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t input_stride     = temp_input_desc.stride;
  extern __shared__ char shmem[];
  nvshmem_device_reference<EmbeddingT> embedding_nvshmem_device_ref{embeding_nvshmem_ref};
  embedding_nvshmem_device_ref.mov_offsets_to_shmem(shmem);
  if (blockIdx.x >= max_blocks_for_local) {
    const int64_t thread_id = (blockIdx.x - max_blocks_for_local) * blockDim.x + threadIdx.x;
    for (int64_t row_id = thread_id; row_id < indice_count - local_index_length;
         row_id += (gridDim.x - max_blocks_for_local) * blockDim.x) {
      const int64_t scaled_row_id =
        row_id < local_index_start ? row_id : row_id + local_index_length;
      int64_t embedding_table_idx = sorted_index[scaled_row_id];
      const int64_t input_idx     = input_index[scaled_row_id];

      if (embedding_table_idx < 0) continue;
      EmbeddingT* temp_input_ptr =
        temp_input + input_stride * input_idx + temp_input_desc.storage_offset;
      int64_t embedding_offset =
        embedding_desc.storage_offset + embedding_table_idx * embedding_stride;
      int dest_rank = embedding_nvshmem_device_ref.dest_rank(embedding_offset);
      EmbeddingT* symmetric_address =
        embedding_nvshmem_device_ref.symmetric_address(embedding_offset);
      if (USE_IBGDA) {
        nvshmem_putmem((symmetric_address),
                       const_cast<const EmbeddingT*>(temp_input_ptr),
                       embedding_size * sizeof(EmbeddingT),
                       dest_rank);
      } else {
        nvshmem_putmem_nbi((symmetric_address),
                           const_cast<const EmbeddingT*>(temp_input_ptr),
                           embedding_size * sizeof(EmbeddingT),
                           dest_rank);
      }
    }
  }

  else {
    //  one embedding per block
    const int thread_id_in_group = threadIdx.x % threads_per_group;
    const int group_id_in_block  = threadIdx.x / threads_per_group;
    const int groups_per_block   = blockDim.x / threads_per_group;
    const int group_id           = blockIdx.x * groups_per_block + group_id_in_block;
    for (int64_t row_id = local_index_start + group_id;
         row_id < local_index_start + local_index_length;
         row_id += max_blocks_for_local * groups_per_block) {
      IndexT embedding_table_idx = sorted_index[row_id];
      IndexT input_idx           = input_index[row_id];
      if (embedding_table_idx < 0) continue;
      // printf("*********in kernel : idx_id =%ld , embedding_table_idx:%ld, output_idx:%ld
      // ,\n",idx_id, int64_t(embedding_table_idx),int64_t( output_idx));
      EmbeddingT* temp_input_ptr = temp_input + input_stride * input_idx;
      int64_t embedding_offset =
        embedding_desc.storage_offset + embedding_table_idx * embedding_stride;
      EmbeddingT* peer_embedding_ptr = static_cast<EmbeddingT*>(
        nvshmem_ptr(embedding_nvshmem_device_ref.symmetric_address(embedding_offset),
                    embedding_nvshmem_device_ref.dest_rank(embedding_offset)));
      if (peer_embedding_ptr == nullptr) {
        printf("Error: Could not find peer NVSHMEM array.\n");
        __trap();
      }
      for (int emb_idx = thread_id_in_group * ALIGNMENT; emb_idx < embedding_size;
           emb_idx += ALIGNMENT * threads_per_group) {
        mov_data<sizeof(EmbeddingT) * ALIGNMENT>(peer_embedding_ptr + emb_idx,
                                                 temp_input_ptr + emb_idx);
      }
    }
  }
}

template <typename InputT, typename IndexT, typename EmbeddingT>
void nvshmem_scatter_temp_put_mem_sort_idx_func(wholememory_comm_t wm_comm,
                                                void* input,
                                                void* temp_input,
                                                wholememory_matrix_description_t input_desc,
                                                const void* indices,
                                                int64_t indice_count,
                                                wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                                                wholememory_matrix_description_t embedding_desc,
                                                size_t* embedding_entry_offsets,
                                                wholememory_env_func_t* p_env_fns,
                                                cudaStream_t stream,
                                                int scatter_sms)
{
  wm_thrust_allocator thrust_allocator(p_env_fns);

  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;
  temp_memory_handle dev_raw_indice(p_env_fns);
  IndexT* dev_raw_indice_ptr = static_cast<IndexT*>(
    dev_raw_indice.device_malloc(indice_count, get_wholememory_dtype<IndexT>()));
  temp_memory_handle dev_sorted_indice(p_env_fns);
  void* dev_sorted_indice_ptr =
    dev_sorted_indice.device_malloc(indice_count, get_wholememory_dtype<IndexT>());

  sort_index_in_pair(indices,
                     indice_count,
                     dev_sorted_indice_ptr,
                     dev_raw_indice_ptr,
                     wm_comm,
                     &thrust_allocator,
                     stream);

  int intra_node_rank_num = wm_comm->intra_node_rank_num;
  int node_id             = wm_comm->world_rank / wm_comm->intra_node_rank_num;

  auto temp_input_data = static_cast<EmbeddingT*>(temp_input);
  auto sorted_index    = static_cast<IndexT*>(dev_sorted_indice_ptr);

  wholememory_matrix_description_t temp_input_desc = input_desc;
  temp_input_desc.storage_offset                   = 0;

  int64_t input_ele_size = wholememory_get_memory_element_count_from_matrix(&input_desc);
  if constexpr (sizeof(EmbeddingT) == sizeof(InputT)) {
    WM_CUDA_CHECK(cudaMemcpyAsync(temp_input,
                                  static_cast<InputT*>(input) + input_desc.storage_offset,
                                  input_ele_size * sizeof(InputT),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
  } else {
    embedding_output_convert_date_type_temp_func<InputT, EmbeddingT>(
      static_cast<EmbeddingT*>(temp_input),
      static_cast<InputT*>(input),
      input_desc.sizes[0],
      embedding_desc.sizes[1],
      temp_input_desc,
      input_desc,
      stream);
  }

  int wm_alignment = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment = determine_memory_alignment_elt_count(temp_input, temp_input_desc);
  int alignment    = std::min<int>(wm_alignment, mm_alignment);

  const char* use_ibgda = std::getenv("NVSHMEM_IB_ENABLE_IBGDA");
  bool use_ibgda_flag   = ((use_ibgda != nullptr) && (strcmp(use_ibgda, "1") == 0));
  void (*scatter_nvshmem_kernel_fn)(EmbeddingT*,
                                    wholememory_matrix_description_t,
                                    wholememory_nvshmem_ref_t,
                                    wholememory_matrix_description_t,
                                    const IndexT*,
                                    const IndexT*,
                                    int64_t,
                                    const int,
                                    const int,
                                    const int,
                                    size_t*,
                                    const int) = nullptr;

  switch (alignment) {
    case 16: {
      scatter_nvshmem_kernel_fn =
        use_ibgda_flag ? scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 16, true>
                       : scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 16, false>;
      break;
    }
    case 8: {
      scatter_nvshmem_kernel_fn =
        use_ibgda_flag ? scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 8, true>
                       : scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 8, false>;
      break;
    }
    case 4: {
      scatter_nvshmem_kernel_fn =
        use_ibgda_flag ? scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 4, true>
                       : scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 4, false>;
      break;
    }
    case 2: {
      scatter_nvshmem_kernel_fn =
        use_ibgda_flag ? scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 2, true>
                       : scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 2, false>;
      break;
    }
    case 1: {
      scatter_nvshmem_kernel_fn =
        use_ibgda_flag ? scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 1, true>
                       : scatter_func_with_nvshmem_sort_idxs_kernel<EmbeddingT, IndexT, 1, false>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("gather func alignment=%d.", alignment);
      return;
    }
  }

  int block_threshold     = 64;
  int max_blocks          = 1024;
  constexpr int WARP_SIZE = 32;

  const int max_threads_per_block   = 256;
  const int num_threads_per_feature = std::min<int64_t>(
    max_threads_per_block,
    ((embedding_desc.sizes[1] / alignment) + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
  const int block_size =
    (max_threads_per_block / num_threads_per_feature) * num_threads_per_feature;
  const int ngroup_per_block = block_size / num_threads_per_feature;
  int num_blocks =
    std::min(max_blocks, static_cast<int>(indice_count + ngroup_per_block) / ngroup_per_block);
  // re-adjust num_blocks/block_threshold to make at least 1 block available for nvshmem remote get
  if (num_blocks < block_threshold) {
    block_threshold = 1;
    if (num_blocks == 1) num_blocks = 2;
  }

  size_t shared_mem_size =
    embeding_nvshmem_ptr.same_chunk ? 0 : ((embeding_nvshmem_ptr.world_size + 1) * sizeof(size_t));
  scatter_nvshmem_kernel_fn<<<num_blocks, block_size, shared_mem_size, stream>>>(
    temp_input_data,
    temp_input_desc,
    embeding_nvshmem_ptr,
    embedding_desc,
    sorted_index,
    dev_raw_indice_ptr,
    indice_count,
    block_threshold,
    intra_node_rank_num,
    node_id,
    embedding_entry_offsets,
    num_threads_per_feature);
  if (!use_ibgda_flag) {
    nvshmemx_quiet_on_stream(stream);  // wait transfer
  }

  WM_CUDA_CHECK(cudaGetLastError());
  (void)scatter_sms;
}

};  // namespace wholememory_ops

#endif
