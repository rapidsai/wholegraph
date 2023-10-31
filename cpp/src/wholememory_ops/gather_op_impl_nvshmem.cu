#ifdef WITH_NVSHMEM_SUPPORT
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

#include "functions/partition_indices.cuh"
#include "wholememory/device_reference.cuh"
#include "wholememory/global_reference.h"
#include "wholememory/nvshmem_template.cuh"
#include <nvshmem.h>
#include <nvshmemx.h>

namespace wholememory_ops {
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
                                                    wholememory_matrix_description_t output_desc,
                                                    size_t embedding_entry_count_per_rank,
                                                    int world_rank,
                                                    int world_size)
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

template <typename EmbeddingT, typename IndexT, typename OutputT>
__global__ void gather_func_with_nvshmem_convert_date_type_kernel(
  OutputT* output,
  const EmbeddingT* temp_output,
  int64_t indice_count,
  wholememory_matrix_description_t embedding_desc,

  wholememory_matrix_description_t output_desc,
  size_t embedding_entry_count_per_rank,
  int world_rank,
  int world_size)
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

template <typename EmbeddingT, typename IndexT, typename OutputT>
__global__ void gather_func_with_nvshmem_mix_kernel(wholememory_nvshmem_ref_t embeding_nvshmem_ref,
                                                    wholememory_matrix_description_t embedding_desc,
                                                    const pair<IndexT>* indices_pair,
                                                    int64_t remote_idx,
                                                    int64_t indice_count,
                                                    int64_t max_block_for_remote,
                                                    EmbeddingT* temp_output,
                                                    wholememory_matrix_description_t output_desc)
{
  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;
  wholememory::nvshmem_device_reference<EmbeddingT> embedding_nvshmem_device_ref{
    embeding_nvshmem_ref};
  int threadx = threadIdx.x + blockIdx.x * blockDim.x;
  if (blockIdx.x < max_block_for_remote) {
    for (int64_t remote_idx_id = threadx; remote_idx_id < remote_idx;
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
      nvshmem_getmem_nbi(temp_output_ptr,
                         const_cast<const EmbeddingT*>(symmetric_address),
                         embedding_size * sizeof(EmbeddingT),
                         dest_rank);
    }
  } else {
    // local
    //  one embedding per block
    for (int64_t idx_id = (blockIdx.x - max_block_for_remote) + remote_idx; idx_id < indice_count;
         idx_id += (gridDim.x - max_block_for_remote)) {
      pair<IndexT> target_indices_pair = indices_pair[idx_id];
      IndexT embedding_table_idx       = target_indices_pair.value;
      IndexT output_idx                = target_indices_pair.raw_idx;
      if (embedding_table_idx < 0) continue;
      // printf("*********in kernel : idx_id =%ld , embedding_table_idx:%ld, output_idx:%ld
      // ,\n",idx_id, int64_t(embedding_table_idx),int64_t( output_idx));
      EmbeddingT* temp_output_ptr = temp_output + output_stride * output_idx;
      int64_t embedding_offset =
        embedding_desc.storage_offset + embedding_table_idx * embedding_stride;
      for (int emb_idx = threadIdx.x; emb_idx < embedding_size; emb_idx += blockDim.x) {
        temp_output_ptr[emb_idx] = embedding_nvshmem_device_ref.load(embedding_offset + emb_idx);
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
                                      size_t embedding_entry_count_per_rank,
                                      int world_rank,
                                      int world_size,
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
                                           output_desc,
                                           embedding_entry_count_per_rank,
                                           world_rank,
                                           world_size);

  nvshmemx_quiet_on_stream(stream);  // wait transfer

  int64_t output_ele_size = indice_count * output_desc.stride;
  if constexpr (sizeof(EmbeddingT) == sizeof(OutputT)) {
    // printf("****************GatherFuncKernel  cudaMemcpyAsync data ***********\n");

    WM_CUDA_CHECK(cudaMemcpyAsync(static_cast<OutputT*>(output) + output_desc.storage_offset,
                                  temp_output,
                                  output_ele_size * sizeof(OutputT),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
  } else {
    // printf("****************GatherFuncKernel  convert data ***********\n");

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
                                              output_desc,
                                              embedding_entry_count_per_rank,
                                              world_rank,
                                              world_size);
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
    int64_t local_idxs         = indice_count - remote_idxs;
    int max_block_for_remote   = 16;
    const int threads          = 128;
    int remote_blocks          = (remote_idxs + threads - 1) / threads;
    max_block_for_remote       = std::min<int>(max_block_for_remote, remote_blocks);
    int64_t local_fetch_blocks = std::max<int64_t>(0, local_idxs);
    int64_t block_count        = max_block_for_remote + local_fetch_blocks;
    block_count = ((block_count >= INT_MAX) ? INT_MAX / 4 : static_cast<int64_t>(block_count));
    // printf("local_idxs is %d,  the block _count is %d *************\n",int(local_idxs), int
    // (block_count));
    gather_func_with_nvshmem_mix_kernel<EmbeddingT, IndexT, OutputT>
      <<<block_count, threads, 0, stream>>>(
        embeding_nvshmem_ptr,
        embedding_desc,
        reinterpret_cast<const pair<IndexT>*>(device_temp_indices_pair_ptr),
        remote_idxs,
        indice_count,
        max_block_for_remote,
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
                                                output_desc,
                                                embedding_entry_count_per_rank,
                                                world_rank,
                                                world_size);
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
    if (wholememory::is_intranode_communicator(wm_comm)) {
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
      WM_CUDA_CHECK(cudaGetLastError());

      return WHOLEMEMORY_SUCCESS;
    }
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

#if 1
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
#else

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
                         embedding_entry_count_per_rank,
                         world_rank,
                         world_size,
                         stream);
#endif
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
