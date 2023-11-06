
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
#include "functions/nvshmem_gather_func.cuh"
#include "wholememory/device_reference.cuh"
#include "wholememory/global_reference.h"
#include "wholememory/nvshmem_template.cuh"
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

template <typename AlignedEmbeddingType, typename IndexT, bool USE_IBGDA>
__global__ void nvshmem_gather_func_kernel(
  const AlignedEmbeddingType* __restrict__ embedding_dev_pointer,  // data already in
                                                                   // AlignedEmbeddingType type
  const int64_t num_feat_elements,  // AlignedEmbeddingType * num_feat = feat size per row
  const int64_t embedding_stride_elements,
  const int64_t output_stride_elements,
  const IndexT* __restrict__ sorted_index,  // sorted (input) index to select
  const IndexT* __restrict__ output_index,  // output index to drop in output array
  int64_t indice_count,
  size_t partition_size,
  const int world_size,  // num_ranks,
  const int local_size,  // num_local_pes
  const int node_rank,   // node_id
  const int block_threshold,
  const int td_per_group,  // number of threads used for fetch each feature tensor (per indx)
  AlignedEmbeddingType* const __restrict__ output)
{
  const int64_t local_index_lowerbound = node_rank * local_size * partition_size;
  const int64_t local_index_upperbound = (node_rank + 1) * local_size * partition_size;
  const int64_t local_index_start  = LowerBound(sorted_index, indice_count, local_index_lowerbound);
  const int64_t local_index_length = UpperBound(
    sorted_index + local_index_start, indice_count - local_index_start, local_index_upperbound - 1);
  //   if(threadIdx.x==0 && blockIdx.x==0) printf("call kernel
  //   nvshmem_gather_func_kernel*********\n");
  // remote fetch
  if (blockIdx.x >= block_threshold) {
    const int64_t thread_id = (blockIdx.x - block_threshold) * blockDim.x + threadIdx.x;
    for (int64_t out_row = thread_id; out_row < indice_count - local_index_length;
         out_row += (gridDim.x - block_threshold) * blockDim.x) {
      const int64_t scaled_out_row =
        out_row < local_index_start ? out_row : out_row + local_index_length;
      int64_t in_row          = sorted_index[scaled_out_row];
      const int64_t out_index = output_index[scaled_out_row];
      const int dest_rank     = in_row / partition_size;
      in_row                  = in_row % partition_size;
      if (USE_IBGDA) {
        nvshmem_getmem(&output[out_index * output_stride_elements],
                       &embedding_dev_pointer[in_row * embedding_stride_elements],
                       num_feat_elements * sizeof(AlignedEmbeddingType),
                       dest_rank);
      } else {
        nvshmem_getmem_nbi(&output[out_index * output_stride_elements],
                           &embedding_dev_pointer[in_row * embedding_stride_elements],
                           num_feat_elements * sizeof(AlignedEmbeddingType),
                           dest_rank);
      }
    }
  }
  // local fetch
  else {
    const int thread_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const int group_id     = thread_id / td_per_group;
    const int thread_col   = thread_id % td_per_group;
    const int total_groups = (block_threshold * blockDim.x) /
                             td_per_group;  // total remote features can be fetched per iteration
    for (int64_t out_row = local_index_start + group_id;
         out_row < local_index_start + local_index_length;
         out_row += total_groups) {
      int64_t in_row          = sorted_index[out_row];
      const int64_t out_index = output_index[out_row];
      const int dest_rank     = in_row / partition_size;
      in_row                  = in_row % partition_size;
      const AlignedEmbeddingType* const __restrict__ peer_array =
        (AlignedEmbeddingType*)nvshmem_ptr(embedding_dev_pointer, dest_rank);
      if (peer_array == nullptr) {
        printf("Error: Could not find peer NVSHMEM array.\n");
        __trap();
      }
      int64_t col = thread_col;

      // const int64_t start_align_offset =
      // ((uintptr_t)(&peer_array[in_row*embedding_stride_elements]) % CACHE_LINE_SIZE) /
      // sizeof(AlignedEmbeddingType); if (start_align_offset != 0) {
      //   col -= start_align_offset;
      //   const int64_t end_align_offset =
      //   ((uintptr_t)(&peer_array[in_row*embedding_stride_elements+num_feat_elements]) %
      //   CACHE_LINE_SIZE) / sizeof(AlignedEmbeddingType); if (col < 0) {
      //     col += num_feat_elements + start_align_offset - end_align_offset;
      //   }
      // } // the logic is error
      while (col < num_feat_elements) {
        output[out_index * output_stride_elements + col] =
          peer_array[in_row * embedding_stride_elements + col];
        col += td_per_group;
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

template <typename DType, typename IdType, typename AlignedType>
void CallMultiNodeAlignedKernel(int alignment,
                                int max_threads,
                                int max_blocks,
                                int64_t num_feat_bytes,
                                const int64_t embedding_stride_bytes,
                                const int64_t output_stride_bytes,
                                int64_t len,
                                cudaStream_t stream,
                                const DType* array_data,
                                IdType* sorted_index,
                                IdType* output_index,
                                size_t partition_size,
                                int num_ranks,
                                int node_id,
                                int num_local_ranks,
                                int block_threshold,
                                DType* ret_data)
{
  const int num_threads =
    std::min<int64_t>(max_threads, (((num_feat_bytes / alignment) + 31) / 32) * 32);
  const int block_size       = (max_threads / num_threads) * num_threads;
  const int ngroup_per_block = block_size / num_threads;
  int num_blocks =
    std::min(max_blocks, static_cast<int>(len + ngroup_per_block) / (block_size / num_threads));
  const char* use_ibgda = std::getenv("NVSHMEM_IB_ENABLE_IBGDA");
  // re-adjust num_blocks/block_threshold to make at least 1 block available for nvshmem remote get
  if (num_blocks < block_threshold) {
    block_threshold = 1;
    if (num_blocks == 1) num_blocks = 2;
  }

  if ((use_ibgda != nullptr) && (strcmp(use_ibgda, "1") == 0)) {
    nvshmem_gather_func_kernel<AlignedType, IdType, true>
      <<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<const AlignedType*>(array_data),
                                              num_feat_bytes / alignment,
                                              embedding_stride_bytes / alignment,
                                              output_stride_bytes / alignment,
                                              sorted_index,
                                              output_index,
                                              len,
                                              partition_size,
                                              num_ranks,
                                              num_local_ranks,
                                              node_id,
                                              block_threshold,
                                              num_threads,
                                              reinterpret_cast<AlignedType*>(ret_data));
  } else {
    nvshmem_gather_func_kernel<AlignedType, IdType, false>
      <<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<const AlignedType*>(array_data),
                                              num_feat_bytes / alignment,
                                              embedding_stride_bytes / alignment,
                                              output_stride_bytes / alignment,
                                              sorted_index,
                                              output_index,
                                              len,
                                              partition_size,
                                              num_ranks,
                                              num_local_ranks,
                                              node_id,
                                              block_threshold,
                                              num_threads,
                                              reinterpret_cast<AlignedType*>(ret_data));
    // Wait for transfer to complete
    nvshmemx_quiet_on_stream(stream);
  }
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
                                               size_t embedding_entry_count_per_rank,
                                               wholememory_env_func_t* p_env_fns,
                                               cudaStream_t stream

)
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

  int intra_node_rank_num                = wm_comm->intra_node_rank_num;
  int node_id                            = wm_comm->world_rank / wm_comm->intra_node_rank_num;
  int block_threshold                    = 16;
  int max_blocks                         = 256;
  int max_threads                        = 512;
  const int64_t num_feat                 = embedding_desc.sizes[1];
  const int64_t num_feat_bytes           = num_feat * sizeof(EmbeddingT);
  const int64_t embedding_stride_bytes   = embedding_desc.stride * sizeof(EmbeddingT);
  const int64_t temp_output_stride_bytes = output_desc.stride * sizeof(EmbeddingT);
  size_t partition_size                  = embedding_entry_count_per_rank;
  auto embedding_nvshmem_pointer = static_cast<const EmbeddingT*>(embeding_nvshmem_ptr.pointer);
  auto ret_data                  = static_cast<EmbeddingT*>(temp_output);
  auto sorted_index              = static_cast<IndexT*>(dev_sorted_indice_ptr);
  if ((num_feat_bytes % 16 == 0) && (embedding_stride_bytes % 16 == 0) &&
      (temp_output_stride_bytes % 16 == 0) && (uintptr_t)embedding_nvshmem_pointer % 16 == 0 &&
      (uintptr_t)ret_data % 16 == 0) {
    CallMultiNodeAlignedKernel<EmbeddingT, IndexT, int4>(16,
                                                         max_threads,
                                                         max_blocks,
                                                         num_feat_bytes,
                                                         embedding_stride_bytes,
                                                         temp_output_stride_bytes,
                                                         indice_count,
                                                         stream,
                                                         embedding_nvshmem_pointer,
                                                         sorted_index,
                                                         dev_raw_indice_ptr,
                                                         partition_size,
                                                         wm_comm->world_size,
                                                         node_id,
                                                         intra_node_rank_num,
                                                         block_threshold,
                                                         ret_data);
  } else if (num_feat_bytes % 8 == 0 && (embedding_stride_bytes % 8 == 0) &&
             (temp_output_stride_bytes % 8 == 0) && (uintptr_t)embedding_nvshmem_pointer % 8 == 0 &&
             (uintptr_t)ret_data % 8 == 0) {
    CallMultiNodeAlignedKernel<EmbeddingT, IndexT, int2>(8,
                                                         max_threads,
                                                         max_blocks,
                                                         num_feat_bytes,
                                                         embedding_stride_bytes,
                                                         temp_output_stride_bytes,
                                                         indice_count,
                                                         stream,
                                                         embedding_nvshmem_pointer,
                                                         sorted_index,
                                                         dev_raw_indice_ptr,
                                                         partition_size,
                                                         wm_comm->world_size,
                                                         node_id,
                                                         intra_node_rank_num,
                                                         block_threshold,
                                                         ret_data);
  } else if (num_feat_bytes % 4 == 0 && (embedding_stride_bytes % 4 == 0) &&
             (temp_output_stride_bytes % 4 == 0) && (uintptr_t)embedding_nvshmem_pointer % 4 == 0 &&
             (uintptr_t)ret_data % 4 == 0) {
    CallMultiNodeAlignedKernel<EmbeddingT, IndexT, int>(4,
                                                        max_threads,
                                                        max_blocks,
                                                        num_feat_bytes,
                                                        embedding_stride_bytes,
                                                        temp_output_stride_bytes,
                                                        indice_count,
                                                        stream,
                                                        embedding_nvshmem_pointer,
                                                        sorted_index,
                                                        dev_raw_indice_ptr,
                                                        partition_size,
                                                        wm_comm->world_size,
                                                        node_id,
                                                        intra_node_rank_num,
                                                        block_threshold,
                                                        ret_data);
  } else if (num_feat_bytes % 2 == 0 && (embedding_stride_bytes % 2 == 0) &&
             (temp_output_stride_bytes % 2 == 0) && (uintptr_t)embedding_nvshmem_pointer % 2 == 0 &&
             (uintptr_t)ret_data % 2 == 0) {
    CallMultiNodeAlignedKernel<EmbeddingT, IndexT, short>(2,
                                                          max_threads,
                                                          max_blocks,
                                                          num_feat_bytes,
                                                          embedding_stride_bytes,
                                                          temp_output_stride_bytes,
                                                          indice_count,
                                                          stream,
                                                          embedding_nvshmem_pointer,
                                                          sorted_index,
                                                          dev_raw_indice_ptr,
                                                          partition_size,
                                                          wm_comm->world_size,
                                                          node_id,
                                                          intra_node_rank_num,
                                                          block_threshold,
                                                          ret_data);
  } else {
    CallMultiNodeAlignedKernel<EmbeddingT, IndexT, EmbeddingT>(1,
                                                               max_threads,
                                                               max_blocks,
                                                               num_feat_bytes,
                                                               embedding_stride_bytes,
                                                               temp_output_stride_bytes,
                                                               indice_count,
                                                               stream,
                                                               embedding_nvshmem_pointer,
                                                               sorted_index,
                                                               dev_raw_indice_ptr,
                                                               partition_size,
                                                               wm_comm->world_size,
                                                               node_id,
                                                               intra_node_rank_num,
                                                               block_threshold,
                                                               ret_data);
  }
  WM_CUDA_CHECK(cudaGetLastError());

  int64_t output_ele_size = indice_count * output_desc.stride;
  if constexpr (sizeof(EmbeddingT) == sizeof(OutputT)) {
    WM_CUDA_CHECK(cudaMemcpyAsync(static_cast<OutputT*>(output) + output_desc.storage_offset,
                                  temp_output,
                                  output_ele_size * sizeof(OutputT),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
  } else {
    int embedding_size = embedding_desc.sizes[1];

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

REGISTER_DISPATCH_THREE_TYPES(GatherFuncNvshmemGetSortIdx,
                              nvshmem_gather_temp_get_mem_sort_idx_func,
                              ALLSINT_ALLFLOAT,
                              SINT3264,
                              ALLSINT_ALLFLOAT);

void call_nvshmem_gather_temp_get_mem_sort_idx_func(wholememory_comm_t wm_comm,
                                                    wholememory_nvshmem_ref_t embedding_nvshmem_ref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    const void* indices,
                                                    wholememory_array_description_t indice_desc,
                                                    void* output,
                                                    void* temp_output,
                                                    wholememory_matrix_description_t output_desc,
                                                    size_t embedding_entry_count_per_rank,
                                                    wholememory_env_func_t* p_env_fns,
                                                    cudaStream_t stream)
{
  DISPATCH_THREE_TYPES(embedding_desc.dtype,
                       indice_desc.dtype,
                       output_desc.dtype,
                       GatherFuncNvshmemGetSortIdx,
                       wm_comm,
                       embedding_nvshmem_ref,
                       embedding_desc,
                       indices,
                       indice_desc.size,
                       output,
                       temp_output,
                       output_desc,
                       embedding_entry_count_per_rank,
                       p_env_fns,
                       stream);
}

}  // namespace wholememory_ops

#endif
