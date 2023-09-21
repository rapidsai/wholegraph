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

#include <nvshmem.h>
#include <nvshmemx.h>

#include "wholememory/nvshmem_template.cuh"

namespace wholememory_ops {
template <typename InputT, typename IndexT, typename EmbeddingT>
__global__ void scatter_func_with_nvshmem_kernel(const InputT* input,
                                                 wholememory_matrix_description_t input_desc,
                                                 const IndexT* indices,
                                                 int64_t indice_count,
                                                 void* embeding_nvshmem_ptr,
                                                 wholememory_matrix_description_t embedding_desc,
                                                 size_t embedding_entry_count_per_rank,
                                                 int world_rank,
                                                 int world_size)
{
  int thread_idx                         = threadIdx.x;
  int embedding_size                     = embedding_desc.sizes[1];
  int64_t embedding_stride               = embedding_desc.stride;
  int64_t input_stride                   = input_desc.stride;
  EmbeddingT* embedding_nvshmem_type_ptr = static_cast<EmbeddingT*>(embeding_nvshmem_ptr);

  for (int64_t input_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       input_idx < indice_count;
       input_idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    const InputT* input_ptr    = input + input_desc.storage_offset + input_stride * input_idx;
    IndexT embedding_table_idx = indices[input_idx];
    if (embedding_table_idx < 0) continue;
    int target_rank                   = embedding_table_idx / embedding_entry_count_per_rank;
    int local_rank_embeding_table_idx = embedding_table_idx % embedding_entry_count_per_rank;
    int64_t embedding_offset =
      embedding_desc.storage_offset + local_rank_embeding_table_idx * embedding_stride;
    for (int emb_idx = thread_idx; emb_idx < embedding_size; emb_idx += blockDim.x) {
      EmbeddingT scatter_ele = convert_type<InputT, EmbeddingT>(input_ptr[emb_idx]);
      // TODO: nvshmem_put block
      wholememory::nvshmem_put<EmbeddingT>(
        embedding_nvshmem_type_ptr + embedding_offset + emb_idx, scatter_ele, target_rank);
    }
  }
}

template <typename InputT, typename IndexT, typename EmbeddingT>
void nvshmem_scatter_temp_func(const void* input,
                               wholememory_matrix_description_t input_desc,
                               void* indices,
                               int64_t indice_count,
                               void* embeding_nvshmem_ptr,
                               wholememory_matrix_description_t embedding_desc,
                               size_t embedding_entry_count_per_rank,
                               int world_rank,
                               int world_size,
                               cudaStream_t stream)
{
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;

  int embedding_size = embedding_desc.sizes[1];
  int thread_x       = std::min(raft::bound_by_power_of_two(embedding_size), 256);
  int thread_y       = 1;
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

  scatter_func_with_nvshmem_kernel<InputT, IndexT, EmbeddingT>
    <<<block_count, block_dim, 0, stream>>>(static_cast<const InputT*>(input),
                                            input_desc,
                                            static_cast<const IndexT*>(indices),
                                            indice_count,
                                            embeding_nvshmem_ptr,
                                            embedding_desc,
                                            embedding_entry_count_per_rank,
                                            world_rank,
                                            world_size);
}

REGISTER_DISPATCH_THREE_TYPES(
  ScatterFunNvshmem, nvshmem_scatter_temp_func, ALLSINT_ALLFLOAT, SINT3264, ALLSINT_ALLFLOAT)

wholememory_error_code_t wholememory_scatter_nvshmem(
  void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  try {
    bool embedding_is_float = wholememory_dtype_is_floating_number(wholememory_desc.dtype);
    WHOLEMEMORY_CHECK(embedding_is_float ||
                      wholememory_dtype_is_integer_number(wholememory_desc.dtype));
    bool input_is_float = wholememory_dtype_is_floating_number(input_desc.dtype);
    WHOLEMEMORY_CHECK(input_is_float || wholememory_dtype_is_integer_number(input_desc.dtype));
    WHOLEMEMORY_EXPECTS(
      embedding_is_float == input_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indices_desc.size == 0) { return WHOLEMEMORY_SUCCESS; }
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
    void* embedding_nvshmem_ptr;
    size_t local_array_size, local_array_offset;
    WHOLEMEMORY_RETURN_ON_FAIL(wholememory_get_local_memory(
      &embedding_nvshmem_ptr, &local_array_size, &local_array_offset, wholememory_handle));

    DISPATCH_THREE_TYPES(input_desc.dtype,
                         indices_desc.dtype,
                         wholememory_desc.dtype,
                         ScatterFunNvshmem,
                         input,
                         input_desc,
                         indices,
                         indices_desc.size,
                         embedding_nvshmem_ptr,
                         wholememory_desc,
                         embedding_entry_count_per_rank,
                         world_rank,
                         world_size,
                         stream);
    // use put
    nvshmemx_barrier_all_on_stream(stream);

    WM_CUDA_CHECK(cudaGetLastError());

  } catch (const wholememory::cuda_error& wle) {
    WHOLEMEMORY_ERROR("scatter CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_ERROR("scatter LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_UNKNOW_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops

#endif