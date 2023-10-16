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

REGISTER_DISPATCH_THREE_TYPES(
  GatherFuncNvshmem, nvshmem_gather_temp_func, ALLSINT_ALLFLOAT, SINT3264, ALLSINT_ALLFLOAT)

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