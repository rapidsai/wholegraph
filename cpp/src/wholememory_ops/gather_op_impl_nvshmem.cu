
#ifdef WITH_NVSHMEM_SUPPORT
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <wholememory/wholememory.h>

#include <wholememory/env_func_ptrs.h>

#include "logger.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/integer_utils.hpp"
#include "wholememory/memory_handle.hpp"
#include "wholememory_ops/functions/bucket_ids_func.h"
#include "wholememory_ops/functions/exchange_embeddings_nccl_func.h"
#include "wholememory_ops/functions/exchange_ids_nccl_func.h"
#include "wholememory_ops/functions/gather_scatter_func.cuh"
#include "wholememory_ops/functions/gather_scatter_func.h"

#include "wholememory_ops/register.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"
#include <wholememory/tensor_description.h>

#include "functions/gather_scatter_func.h"
#include "wholememory/device_reference.cuh"
#include "wholememory/global_reference.h"
#include "wholememory/nvshmem_template.cuh"
#include <cub/cub.cuh>
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

template <typename EmbeddingT, typename IndexT, typename OutputT>
__global__ void gather_func_with_nvshmem_convert_date_type_kernel(
  OutputT* output,
  const EmbeddingT* temp_output,
  int64_t indice_count,
  wholememory_matrix_description_t embedding_desc,

  wholememory_matrix_description_t output_desc)
{
  int thread_idx        = threadIdx.x;
  int64_t output_stride = output_desc.stride;
  int embedding_size    = embedding_desc.sizes[1];

  for (int64_t output_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       output_idx < indice_count;
       output_idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    OutputT* output_ptr = output + output_desc.storage_offset + output_stride * output_idx;

    int64_t temp_output_offset = output_idx * output_stride;
    for (int emb_idx = thread_idx; emb_idx < embedding_size; emb_idx += blockDim.x) {
      output_ptr[emb_idx] =
        convert_type<EmbeddingT, OutputT>(temp_output[temp_output_offset + emb_idx]);
    }
  }
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
  size_t embedding_entry_per_rank,
  EmbeddingT* __restrict__ temp_output,
  wholememory_matrix_description_t output_desc,
  const int threads_per_group)
{
  const int64_t local_index_lowerbound = node_rank * intra_node_ranks * embedding_entry_per_rank;
  const int64_t local_index_upperbound =
    (node_rank + 1) * intra_node_ranks * embedding_entry_per_rank;
  const int64_t local_index_start  = LowerBound(sorted_index, indice_count, local_index_lowerbound);
  const int64_t local_index_length = UpperBound(
    sorted_index + local_index_start, indice_count - local_index_start, local_index_upperbound - 1);

  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;
  wholememory::nvshmem_device_reference<EmbeddingT> embedding_nvshmem_device_ref{
    embeding_nvshmem_ref};
  if (blockIdx.x >= max_blocks_for_local) {
    const int64_t thread_id = (blockIdx.x - max_blocks_for_local) * blockDim.x + threadIdx.x;
    for (int64_t row_id = thread_id; row_id < indice_count - local_index_length;
         row_id += (gridDim.x - max_blocks_for_local) * blockDim.x) {
      const int64_t scaled_row_id =
        row_id < local_index_start ? row_id : row_id + local_index_length;
      int64_t embedding_table_idx = sorted_index[scaled_row_id];
      const int64_t output_idx    = output_index[scaled_row_id];

      if (embedding_table_idx < 0) continue;
      EmbeddingT* temp_output_ptr = temp_output + output_stride * output_idx;
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
      EmbeddingT* temp_output_ptr = temp_output + output_stride * output_idx;
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
    thrust::cuda::par(allocator).on(stream), seq_indices, seq_indices + indice_count, 0);
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

template <typename EmbeddingT, typename IndexT, typename OutputT>
void nvshmem_gather_temp_get_mem_sort_idx_func(wholememory_comm_t wm_comm,
                                               wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                                               wholememory_matrix_description_t embedding_desc,
                                               const void* indices,
                                               int64_t indice_count,
                                               void* output,
                                               void* temp_output,
                                               wholememory_matrix_description_t output_desc,
                                               size_t embedding_entry_count_per_rank,

                                               wholememory_env_func_t* p_env_fns,
                                               cudaStream_t stream

)
{
  int embedding_size = embedding_desc.sizes[1];
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

  auto ret_data     = static_cast<EmbeddingT*>(temp_output);
  auto sorted_index = static_cast<IndexT*>(dev_sorted_indice_ptr);

  int wm_alignment = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment = determine_memory_alignment_elt_count(temp_output, output_desc);
  int alignment    = std::min<int>(wm_alignment, mm_alignment);

  const char* use_ibgda = std::getenv("NVSHMEM_IB_ENABLE_IBGDA");
  bool use_ibgda_flag   = ((use_ibgda != nullptr) && (strcmp(use_ibgda, "1") == 0));
  void (*gather_nvshmem_kernel_fn)(wholememory_nvshmem_ref_t,
                                   wholememory_matrix_description_t,
                                   const IndexT*,  // sorted (input) index to select
                                   const IndexT*,  // output index to drop in output array
                                   int64_t,
                                   const int,
                                   const int,
                                   const int,
                                   size_t,
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

  int block_threshold     = 32;
  int max_blocks          = 512;
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

  gather_nvshmem_kernel_fn<<<num_blocks, block_size, 0, stream>>>(embeding_nvshmem_ptr,
                                                                  embedding_desc,
                                                                  sorted_index,
                                                                  dev_raw_indice_ptr,
                                                                  indice_count,
                                                                  block_threshold,
                                                                  intra_node_rank_num,
                                                                  node_id,
                                                                  embedding_entry_count_per_rank,
                                                                  ret_data,
                                                                  output_desc,
                                                                  num_threads_per_feature);
  if (!use_ibgda_flag) {
    nvshmemx_quiet_on_stream(stream);  // wait transfer
  }

  int64_t output_ele_size = indice_count * output_desc.stride;
  if constexpr (sizeof(EmbeddingT) == sizeof(OutputT)) {
    WM_CUDA_CHECK(cudaMemcpyAsync(static_cast<OutputT*>(output) + output_desc.storage_offset,
                                  temp_output,
                                  output_ele_size * sizeof(OutputT),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
  } else {
    int thread_x = std::min(raft::bound_by_power_of_two(embedding_size), 256);
    int thread_y = 1;
    if (thread_x < 64) {
      int power2_thread_x = 1;
      for (; power2_thread_x < thread_x; power2_thread_x *= 2)
        ;
      thread_x = power2_thread_x;
      thread_y = 64 / thread_x;
    }
    int64_t block_count_64 = (indice_count + thread_y - 1) / thread_y;
    int block_count = block_count_64 >= INT_MAX ? INT_MAX / 4 : static_cast<int>(block_count_64);
    dim3 block_dim(thread_x, thread_y, 1);

    gather_func_with_nvshmem_convert_date_type_kernel<EmbeddingT, IndexT, OutputT>
      <<<block_count, block_dim, 0, stream>>>(static_cast<OutputT*>(output),
                                              static_cast<EmbeddingT*>(temp_output),
                                              indice_count,
                                              embedding_desc,
                                              output_desc);
  }

  WM_CUDA_CHECK(cudaGetLastError());
}
// TWO kernel .

// 1. read 2. convert type

REGISTER_DISPATCH_THREE_TYPES(GatherFuncNvshmemGetSortIdx,
                              nvshmem_gather_temp_get_mem_sort_idx_func,
                              ALLSINT_ALLFLOAT,
                              SINT3264,
                              ALLSINT_ALLFLOAT)

wholememory_error_code_t wholememory_gather_nvshmem(
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  try {
    bool embedding_is_float = wholememory_dtype_is_floating_number(wholememory_desc.dtype);
    WHOLEMEMORY_CHECK(embedding_is_float ||
                      wholememory_dtype_is_integer_number(wholememory_desc.dtype));
    bool output_is_float = wholememory_dtype_is_floating_number(output_desc.dtype);
    WHOLEMEMORY_CHECK(output_is_float || wholememory_dtype_is_integer_number(output_desc.dtype));
    WHOLEMEMORY_EXPECTS(
      embedding_is_float == output_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indice_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
    if (wholememory_desc.storage_offset < 0 ||
        wholememory_desc.storage_offset + wholememory_desc.sizes[1] > wholememory_desc.stride) {
      return WHOLEMEMORY_INVALID_INPUT;
    }

    size_t embedding_size_per_rank;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_partition_plan(&embedding_size_per_rank, wholememory_handle));

    size_t element_size         = wholememory_dtype_get_element_size(wholememory_desc.dtype);
    size_t embedding_entry_size = element_size * wholememory_desc.stride;

    WHOLEMEMORY_EXPECTS_NOTHROW(
      embedding_size_per_rank % embedding_entry_size == 0,
      "embedding_size_per_rank=%ld is not multiple of embedding_entry_size=%ldx%ld",
      embedding_size_per_rank,
      element_size,
      wholememory_desc.stride);

    size_t embedding_entry_count_per_rank = embedding_size_per_rank / embedding_entry_size;

    wholememory_comm_t wm_comm;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_communicator(&wm_comm, wholememory_handle));

    int world_size;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_size(&world_size, wm_comm));
    int world_rank;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_communicator_get_rank(&world_rank, wm_comm));
    wholememory_nvshmem_ref_t embedding_nvshmem_ref;
    WHOLEMEMORY_RETURN_ON_FAIL(
      wholememory_get_nvshmem_reference(&embedding_nvshmem_ref, wholememory_handle));

    temp_memory_handle device_temp_output_handle(p_env_fns);
    size_t temp_output_ele_size = indice_desc.size * output_desc.stride;
    void* temp_output_ptr =
      device_temp_output_handle.device_malloc(temp_output_ele_size, wholememory_desc.dtype);
    size_t temp_output_byte_size =
      temp_output_ele_size * wholememory_dtype_get_element_size(wholememory_desc.dtype);
    // register
    if (nvshmemx_buffer_register(temp_output_ptr, temp_output_byte_size) != 0) {
      WHOLEMEMORY_ERROR("nvshmemx_buffer_register error in wholememory_gather_nvshmem");
    }

    DISPATCH_THREE_TYPES(wholememory_desc.dtype,
                         indice_desc.dtype,
                         output_desc.dtype,
                         GatherFuncNvshmemGetSortIdx,
                         wm_comm,
                         embedding_nvshmem_ref,
                         wholememory_desc,
                         indices,
                         indice_desc.size,
                         output,
                         temp_output_ptr,
                         output_desc,
                         embedding_entry_count_per_rank,
                         p_env_fns,
                         stream);
    // ungistre
    if (nvshmemx_buffer_unregister(temp_output_ptr) != 0) {
      WHOLEMEMORY_ERROR("nvshmemx_buffer_unregister error in wholememory_gather_nvshmem");
    }

    WM_CUDA_CHECK(cudaGetLastError());

  } catch (wholememory::cuda_error& wce) {
    WHOLEMEMORY_ERROR("CUDA logic Error %s\n", wce.what());
    return WHOLEMEMORY_CUDA_ERROR;
  } catch (wholememory::logic_error& wle) {
    WHOLEMEMORY_ERROR("LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }

  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops

#endif
