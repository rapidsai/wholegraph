#ifdef WITH_NVSHMEM_SUPPORT

#include "nvshmem_gather_scatter_func.cuh"
#include <wholememory/wholememory.h>

#include "logger.hpp"
#include "wholememory_ops/register.hpp"
namespace wholememory_ops {

template <typename InputT, typename EmbeddingT>
void nvshmem_scatter_integer_int64_temp_func(wholememory_comm_t wm_comm,
                                             void* input,
                                             void* temp_input,
                                             wholememory_matrix_description_t input_desc,
                                             const void* indices,
                                             int64_t indice_count,
                                             wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                                             wholememory_matrix_description_t embedding_desc,
                                             size_t embedding_entry_count_per_rank,
                                             wholememory_env_func_t* p_env_fns,
                                             cudaStream_t stream)
{
  nvshmem_scatter_temp_put_mem_sort_idx_func<InputT, int64_t, EmbeddingT>(
    wm_comm,
    input,
    temp_input,
    input_desc,
    indices,
    indice_count,
    embeding_nvshmem_ptr,
    embedding_desc,
    embedding_entry_count_per_rank,
    p_env_fns,
    stream);
}

REGISTER_DISPATCH_TWO_TYPES(NvshmemScatterFuncIntegerInt64,
                            nvshmem_scatter_integer_int64_temp_func,
                            ALLSINT,
                            ALLSINT)

wholememory_error_code_t nvshmem_scatter_integer_int64_func(
  wholememory_comm_t wm_comm,
  void* input,
  void* temp_input,
  wholememory_matrix_description_t input_desc,
  const void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
  wholememory_matrix_description_t embedding_desc,
  size_t embedding_entry_count_per_rank,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  try {
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(embedding_desc.dtype));
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(input_desc.dtype));
    WHOLEMEMORY_CHECK(indices_desc.dtype == WHOLEMEMORY_DT_INT64);

    DISPATCH_TWO_TYPES(input_desc.dtype,
                       embedding_desc.dtype,
                       NvshmemScatterFuncIntegerInt64,
                       wm_comm,
                       input,
                       temp_input,
                       input_desc,
                       indices,
                       indices_desc.size,
                       embeding_nvshmem_ptr,
                       embedding_desc,
                       embedding_entry_count_per_rank,
                       p_env_fns,
                       stream);
  } catch (const wholememory::cuda_error& wle) {
    WHOLEMEMORY_ERROR("scatter CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_ERROR("scatter CUDA LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}
};  // namespace wholememory_ops

#endif
