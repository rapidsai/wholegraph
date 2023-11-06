
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
                                                    cudaStream_t stream);
template <typename EmbeddingT, typename IndexT, typename OutputT>
__global__ void gather_func_with_nvshmem_kernel(wholememory_nvshmem_ref_t embeding_nvshmem_ref,
                                                wholememory_matrix_description_t embedding_desc,
                                                const IndexT* indices,
                                                int64_t indice_count,
                                                OutputT* output,
                                                wholememory_matrix_description_t output_desc,
                                                size_t embedding_entry_count_per_rank,
                                                int world_rank,
                                                int world_size)
{
  //   int64_t output_idx         = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  //   IndexT embedding_table_idx = indices[output_idx];
  //   if (embedding_table_idx < 0) return;
  int thread_idx           = threadIdx.x;
  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;
  wholememory::nvshmem_device_reference<EmbeddingT> embedding_nvshmem_device_ref{
    embeding_nvshmem_ref};
  for (int64_t output_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       output_idx < indice_count;
       output_idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    IndexT embedding_table_idx = indices[output_idx];
    if (embedding_table_idx < 0) continue;
    OutputT* output_ptr = output + output_desc.storage_offset + output_stride * output_idx;

    int64_t embedding_offset =
      embedding_desc.storage_offset + embedding_table_idx * embedding_stride;
    for (int emb_idx = thread_idx; emb_idx < embedding_size; emb_idx += blockDim.x) {
      output_ptr[emb_idx] = convert_type<EmbeddingT, OutputT>(
        embedding_nvshmem_device_ref.load(embedding_offset + emb_idx));
    }
  }
}

// TWO kernel .

// 1. read 2. convert type

template <typename EmbeddingT, typename IndexT>
__global__ void gather_func_with_nvshmem_get_kernel(wholememory_nvshmem_ref_t embeding_nvshmem_ref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    const IndexT* indices,
                                                    int64_t indice_count,
                                                    EmbeddingT* temp_output,
                                                    wholememory_matrix_description_t output_desc
                                                    )
{
  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;
  wholememory::nvshmem_device_reference<EmbeddingT> embedding_nvshmem_device_ref{
    embeding_nvshmem_ref};
  for (int64_t output_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output_idx < indice_count;
       output_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    IndexT embedding_table_idx = indices[output_idx];
    if (embedding_table_idx < 0) continue;
    EmbeddingT* temp_output_ptr = temp_output + output_stride * output_idx;
    int64_t embedding_offset =
      embedding_desc.storage_offset + embedding_table_idx * embedding_stride;
    int dest_rank = embedding_nvshmem_device_ref.dest_rank(embedding_offset);
    EmbeddingT* symmetric_address =
      embedding_nvshmem_device_ref.symmetric_address(embedding_offset);
    nvshmem_getmem_nbi(temp_output_ptr,
                       const_cast<const EmbeddingT*>(symmetric_address),
                       embedding_size * sizeof(EmbeddingT),
                       dest_rank);
  }
}

template <typename EmbeddingT, typename IndexT, int ALIGNMENT = 1, bool USE_IBGDA = true>
__global__ void gather_func_with_nvshmem_mix_kernel(wholememory_nvshmem_ref_t embeding_nvshmem_ref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    const pair<IndexT>* indices_pair,
                                                    int64_t remote_idx_count,
                                                    int64_t indice_count,
                                                    int64_t max_block_for_remote,
                                                    EmbeddingT* temp_output,
                                                    wholememory_matrix_description_t output_desc,
                                                    const int threads_per_group)
{
  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;
  wholememory::nvshmem_device_reference<EmbeddingT> embedding_nvshmem_device_ref{
    embeding_nvshmem_ref};
  int threadx = threadIdx.x + blockIdx.x * blockDim.x;
  if (blockIdx.x < max_block_for_remote) {
    for (int64_t remote_idx_id = threadx; remote_idx_id < remote_idx_count;
         remote_idx_id += max_block_for_remote * blockDim.x) {
      pair<IndexT> target_indices_pair = indices_pair[remote_idx_id];
      IndexT embedding_table_idx       = target_indices_pair.value;
      IndexT output_idx                = target_indices_pair.raw_idx;
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
  } else {
    // local
    //  one embedding per block
    const int thread_id_in_group = threadIdx.x % threads_per_group;
    const int group_id_in_block  = threadIdx.x / threads_per_group;
    const int groups_per_block   = blockDim.x / threads_per_group;
    for (int64_t idx_id = (blockIdx.x - max_block_for_remote) * groups_per_block +
                          group_id_in_block + remote_idx_count;
         idx_id < indice_count;
         idx_id += ((gridDim.x - max_block_for_remote) * groups_per_block)) {
      pair<IndexT> target_indices_pair = indices_pair[idx_id];
      IndexT embedding_table_idx       = target_indices_pair.value;
      IndexT output_idx                = target_indices_pair.raw_idx;
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
template <typename EmbeddingT, typename IndexT, typename OutputT>
void nvshmem_gather_temp_get_mem_func(wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                                      wholememory_matrix_description_t embedding_desc,
                                      const void* indices,
                                      int64_t indice_count,
                                      void* output,
                                      void* temp_output,
                                      wholememory_matrix_description_t output_desc,
                                      cudaStream_t stream

)
{
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;

  int embedding_size      = embedding_desc.sizes[1];
  int thread_x            = 128;
  int64_t block_count_128 = (indice_count + thread_x - 1) / thread_x;
  int block_count =
    ((block_count_128 >= INT_MAX) ? INT_MAX / 4 : static_cast<int>(block_count_128));
  gather_func_with_nvshmem_get_kernel<EmbeddingT, IndexT>
    <<<block_count, thread_x, 0, stream>>>(embeding_nvshmem_ptr,
                                           embedding_desc,
                                           static_cast<const IndexT*>(indices),
                                           indice_count,
                                           static_cast<EmbeddingT*>(temp_output),
                                           output_desc);

  nvshmemx_quiet_on_stream(stream);  // wait transfer

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
}

template <typename EmbeddingT, typename IndexT, typename OutputT>
void nvshmem_gather_temp_func(wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                              wholememory_matrix_description_t embedding_desc,
                              const void* indices,
                              int64_t indice_count,
                              void* output,
                              wholememory_matrix_description_t output_desc,
                              size_t embedding_entry_count_per_rank,
                              int world_rank,
                              int world_size,
                              cudaStream_t stream

)
{
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;

  int embedding_size = embedding_desc.sizes[1];
  // int thread_x       = wholememory::div_rounding_up_safe<int>(embedding_size, alignment);
  // thread_x           = std::min(thread_x, 256);
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

  gather_func_with_nvshmem_kernel<EmbeddingT, IndexT, OutputT>
    <<<block_count, block_dim, 0, stream>>>(embeding_nvshmem_ptr,
                                            embedding_desc,
                                            static_cast<const IndexT*>(indices),
                                            indice_count,
                                            static_cast<OutputT*>(output),
                                            output_desc,
                                            embedding_entry_count_per_rank,
                                            world_rank,
                                            world_size);
}

template <typename EmbeddingT, typename IndexT, typename OutputT>
void nvshmem_gather_temp_get_mem_mix_func(wholememory_comm_t wm_comm,
                                          wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                                          wholememory_matrix_description_t embedding_desc,
                                          const void* indices,
                                          int64_t indice_count,
                                          void* output,
                                          void* temp_output,
                                          wholememory_matrix_description_t output_desc,
                                          size_t embedding_entry_count_per_rank,
                                          int world_rank,
                                          int world_size,
                                          wholememory_env_func_t* p_env_fns,
                                          cudaStream_t stream

)
{
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;

  int embedding_size        = embedding_desc.sizes[1];
  int intra_node_first_rank = wm_comm->intra_node_first_rank;
  int intra_node_end_rank   = wm_comm->intra_node_rank_num + intra_node_first_rank;
  int64_t partition_min =
    std::min(static_cast<int64_t>(intra_node_first_rank * embedding_entry_count_per_rank),
             embedding_desc.sizes[0]);
  int64_t partition_max =
    std::min(static_cast<int64_t>(intra_node_end_rank * embedding_entry_count_per_rank),
             embedding_desc.sizes[0]);
  wm_thrust_allocator thrust_allocator(p_env_fns);

  temp_memory_handle device_temp_indices_pair_handle(p_env_fns);
  IndexT* device_temp_indices_pair_ptr =
    static_cast<IndexT*>(device_temp_indices_pair_handle.device_malloc(
      indice_count * 2, get_wholememory_dtype<IndexT>()));

  int64_t remote_idxs = 0;

  generate_partition_pair_index<IndexT>(
    static_cast<const IndexT*>(indices),
    indice_count,
    partition_min,
    partition_max,
    reinterpret_cast<pair<IndexT>*>(device_temp_indices_pair_ptr),
    &remote_idxs,
    &thrust_allocator,
    p_env_fns,
    stream);
  // [0,remote_idx]
  if (remote_idxs <= 0) {
    return nvshmem_gather_temp_func<EmbeddingT, IndexT, OutputT>(embeding_nvshmem_ptr,
                                                                 embedding_desc,
                                                                 indices,
                                                                 indice_count,
                                                                 output,
                                                                 output_desc,
                                                                 embedding_entry_count_per_rank,
                                                                 world_rank,
                                                                 world_size,
                                                                 stream);
  } else {
    int64_t local_idxs = indice_count - remote_idxs;
    // int64_t local_fetch_blocks = std::max<int64_t>(0, local_idxs);
    // printf("local_idxs is %d,  the block _count is %d *************\n",int(local_idxs), int
    // (block_count));
    int wm_alignment        = determine_wholememory_alignment_elt_count(embedding_desc);
    int mm_alignment        = determine_memory_alignment_elt_count(temp_output, output_desc);
    int alignment           = std::min<int>(wm_alignment, mm_alignment);
    constexpr int WARP_SIZE = 32;

    const int max_threads_per_block   = 256;
    const int num_threads_per_feature = std::min<int64_t>(
      max_threads_per_block,
      ((embedding_desc.sizes[1] / alignment) + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const int num_threads_per_block =
      (max_threads_per_block / num_threads_per_feature) * num_threads_per_feature;
    const int ngroup_per_block = num_threads_per_block / num_threads_per_feature;
    int local_fetch_blocks     = (local_idxs + ngroup_per_block - 1) / ngroup_per_block;
    int max_block_for_remote   = 16;
    int remote_blocks          = (remote_idxs + num_threads_per_block - 1) / num_threads_per_block;
    max_block_for_remote       = std::max<int>(max_block_for_remote, remote_blocks);
    int64_t block_count        = max_block_for_remote + local_fetch_blocks;
    block_count = ((block_count >= INT_MAX) ? INT_MAX / 4 : static_cast<int64_t>(block_count));

    const char* use_ibgda = std::getenv("NVSHMEM_IB_ENABLE_IBGDA");

    bool use_ibgda_flag = ((use_ibgda != nullptr) && (strcmp(use_ibgda, "1") == 0));

    void (*gather_nvshmem_kernel_fn)(wholememory_nvshmem_ref_t,
                                     wholememory_matrix_description_t,
                                     const pair<IndexT>*,
                                     int64_t,
                                     int64_t,
                                     int64_t,
                                     EmbeddingT*,
                                     wholememory_matrix_description_t,
                                     const int) = nullptr;
    switch (alignment) {
      case 16: {
        gather_nvshmem_kernel_fn =
          use_ibgda_flag ? gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 16, true>
                         : gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 16, false>;
        break;
      }
      case 8: {
        gather_nvshmem_kernel_fn =
          use_ibgda_flag ? gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 8, true>
                         : gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 8, false>;
        break;
      }
      case 4: {
        gather_nvshmem_kernel_fn =
          use_ibgda_flag ? gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 4, true>
                         : gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 4, false>;
        break;
      }
      case 2: {
        gather_nvshmem_kernel_fn =
          use_ibgda_flag ? gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 2, true>
                         : gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 2, false>;
        break;
      }
      case 1: {
        gather_nvshmem_kernel_fn =
          use_ibgda_flag ? gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 1, true>
                         : gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, 1, false>;
        break;
      }
      default: {
        WHOLEMEMORY_FAIL("gather func alignment=%d.", alignment);
        return;
      }
    }
    gather_nvshmem_kernel_fn<<<block_count, num_threads_per_block, 0, stream>>>(
      embeding_nvshmem_ptr,
      embedding_desc,
      reinterpret_cast<const pair<IndexT>*>(device_temp_indices_pair_ptr),
      remote_idxs,
      indice_count,
      max_block_for_remote,
      static_cast<EmbeddingT*>(temp_output),
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
  }
}

REGISTER_DISPATCH_THREE_TYPES(
  GatherFuncNvshmem, nvshmem_gather_temp_func, ALLSINT_ALLFLOAT, SINT3264, ALLSINT_ALLFLOAT)

REGISTER_DISPATCH_THREE_TYPES(GatherFuncNvshmemGet,
                              nvshmem_gather_temp_get_mem_func,
                              ALLSINT_ALLFLOAT,
                              SINT3264,
                              ALLSINT_ALLFLOAT)

REGISTER_DISPATCH_THREE_TYPES(GatherFuncNvshmemGetMIX,
                              nvshmem_gather_temp_get_mem_mix_func,
                              ALLSINT_ALLFLOAT,
                              SINT3264,
                              ALLSINT_ALLFLOAT)

// #ifdef WHOLEGRAPH_USE_NVSHMEM

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
    // if (wholememory::is_intranode_communicator(wm_comm)) {
    //   DISPATCH_THREE_TYPES(wholememory_desc.dtype,
    //                        indice_desc.dtype,
    //                        output_desc.dtype,
    //                        GatherFuncNvshmem,
    //                        embedding_nvshmem_ref,
    //                        wholememory_desc,
    //                        indices,
    //                        indice_desc.size,
    //                        output,
    //                        output_desc,
    //                        embedding_entry_count_per_rank,
    //                        world_rank,
    //                        world_size,
    //                        stream);
    //   WM_CUDA_CHECK(cudaGetLastError());

    //   return WHOLEMEMORY_SUCCESS;
    // }
#if 1

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
    // GatherFuncNvshmemGetMIX
    // GatherFuncNvshmemGetSortIdx

    const char* use_dlfw        = std::getenv("USE_DLFW");
    const char* use_NVSHMEM_GET = std::getenv("USE_NVSHMEM_GET");
    // re-adjust num_blocks/block_threshold to make at least 1 block available for nvshmem remote
    // get

    if ((use_dlfw != nullptr) && (strcmp(use_dlfw, "1") == 0)) {
      // printf("*******use dlfw ******************\n");

      call_nvshmem_gather_temp_get_mem_sort_idx_func(wm_comm,
                                                     embedding_nvshmem_ref,
                                                     wholememory_desc,
                                                     indices,
                                                     indice_desc,
                                                     output,
                                                     temp_output_ptr,
                                                     output_desc,
                                                     embedding_entry_count_per_rank,
                                                     p_env_fns,
                                                     stream);

    } else if ((use_NVSHMEM_GET != nullptr) && (strcmp(use_NVSHMEM_GET, "1") == 0)) {
      // printf("******* use_NVSHMEM_GET ******************\n");

      DISPATCH_THREE_TYPES(wholememory_desc.dtype,
                           indice_desc.dtype,
                           output_desc.dtype,
                           GatherFuncNvshmemGet,
                           embedding_nvshmem_ref,
                           wholememory_desc,
                           indices,
                           indice_desc.size,
                           output,
                           temp_output_ptr,
                           output_desc,
                           stream);

    } else {
      DISPATCH_THREE_TYPES(wholememory_desc.dtype,
                           indice_desc.dtype,
                           output_desc.dtype,
                           GatherFuncNvshmemGetMIX,
                           wm_comm,
                           embedding_nvshmem_ref,
                           wholememory_desc,
                           indices,
                           indice_desc.size,
                           output,
                           temp_output_ptr,
                           output_desc,
                           embedding_entry_count_per_rank,
                           world_rank,
                           world_size,
                           p_env_fns,
                           stream);
    }

    // ungistre
    if (nvshmemx_buffer_unregister(temp_output_ptr) != 0) {
      WHOLEMEMORY_ERROR("nvshmemx_buffer_unregister error in wholememory_gather_nvshmem");
    }

#else
    DISPATCH_THREE_TYPES(wholememory_desc.dtype,
                         indice_desc.dtype,
                         output_desc.dtype,
                         GatherFuncNvshmem,
                         embedding_nvshmem_ref,
                         wholememory_desc,
                         indices,
                         indice_desc.size,
                         output,
                         output_desc,
                         embedding_entry_count_per_rank,
                         world_rank,
                         world_size,
                         stream);
#endif
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
